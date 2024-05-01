import os
import tqdm

import torch
import numpy as np
import torch.nn as nn

from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint
from lib.losses.centernet_loss import compute_centernet3d_loss
from lib.losses.proposal_head_loss import compute_proposal_head_loss


class Trainer(object):
    def __init__(
        self,
        cfg,
        model,
        optimizer,
        train_loader,
        test_loader,
        lr_scheduler,
        warmup_lr_scheduler,
        logger,
    ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # loading pretrain/resume model
        if cfg.get("pretrain_model"):
            assert os.path.exists(cfg["pretrain_model"])
            load_checkpoint(
                model=self.model,
                optimizer=None,
                filename=cfg["pretrain_model"],
                map_location=self.device,
                logger=self.logger,
            )

        if cfg.get("resume_model", None):
            assert os.path.exists(cfg["resume_model"])
            self.epoch = load_checkpoint(
                model=self.model.to(self.device),
                optimizer=self.optimizer,
                filename=cfg["resume_model"],
                map_location=self.device,
                logger=self.logger,
            )
            self.lr_scheduler.last_epoch = self.epoch - 1

        self.gpu_ids = [
            0,1
        ]  # list(map(int, cfg["gpu_ids"].split(",")))
        self.model = torch.nn.DataParallel(model, device_ids=self.gpu_ids).to(
            self.device
        )

    def train(self):
        start_epoch = self.epoch

        progress_bar = tqdm.tqdm(
            range(start_epoch, self.cfg["max_epoch"]),
            dynamic_ncols=True,
            leave=True,
            desc="epochs",
        )
        for epoch in range(start_epoch, self.cfg["max_epoch"]):
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            # train one epoch
            self.train_one_epoch()
            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            # save trained model
            if (self.epoch % self.cfg["save_frequency"]) == 0:
                os.makedirs("checkpoints", exist_ok=True)
                ckpt_name = os.path.join(
                    "checkpoints", "checkpoint_epoch_%d" % self.epoch
                )
                save_checkpoint(
                    get_checkpoint_state(self.model, self.optimizer, self.epoch),
                    ckpt_name,
                )

            progress_bar.update()

        return None

    def train_one_epoch(self):
        self.model.train()
        progress_bar = tqdm.tqdm(
            total=len(self.train_loader),
            leave=(self.epoch + 1 == self.cfg["max_epoch"]),
            desc="iters",
        )
        center_loss = 0
        rcnn_loss = 0
        for batch_idx, (inputs, targets, _) in enumerate(self.train_loader):
            rgb, hha = inputs
            rgb = rgb.to(self.device)
            hha = hha.to(self.device)
            for key in targets.keys():
                targets[key] = targets[key].to(self.device)

            proposals_2d = targets["2d_bbox"]

            ## change b,N,5 to N,6 st first column is the index of b
            b = proposals_2d.shape[0]
            n = proposals_2d.shape[1]
            b_indices = torch.arange(b).view(b, 1, 1).expand(b, n, 1).to(self.device)

            # Concatenate the b_indices tensor with the original tensor along the last dimension
            processed_tensor = torch.cat((b_indices, proposals_2d), dim=-1)

            # Reshape the processed tensor to (b*n, 6)
            processed_tensor = processed_tensor.view(b * n, 6)
            ## remove second column from processed_tensor
            proposals_2d = processed_tensor[:, [0, 2, 3, 4, 5]]

            # train one batch
            self.optimizer.zero_grad()
            outputs, (dim, rot_cls, rot_reg, loc) = self.model(
                rgb, hha, proposals_2d.type(torch.float32).to(self.device)
            )
            loss_centernet, stats_batch = compute_centernet3d_loss(outputs, targets)
            loss_rcnn = compute_proposal_head_loss(
                (dim, rot_cls, rot_reg, loc), targets
            )
            total_loss = loss_centernet + loss_rcnn
            total_loss.backward()
            self.optimizer.step()
            center_loss += loss_centernet.item()
            rcnn_loss += loss_rcnn.item()
            progress_bar.update()
        with open("losses.txt", "a") as f:
            f.write(
                f"Epoch {self.epoch} Center Loss: {center_loss / len(self.train_loader)} RCNN Loss: {rcnn_loss / len(self.train_loader)}\n")
        progress_bar.close()
