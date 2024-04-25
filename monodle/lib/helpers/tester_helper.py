import os
import tqdm

import torch

from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs, extract_dets_from_rpn
from lib.helpers.decode_helper import decode_detections


class Tester(object):
    def __init__(self, cfg, model, dataloader, logger, eval=False):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.max_objs = (
            dataloader.dataset.max_objs
        )  # max objects per images, defined in dataset
        self.class_name = dataloader.dataset.class_name
        self.output_dir = "./outputs"
        self.dataset_type = cfg.get("type", "KITTI")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.eval = eval

    def test(self):
        assert self.cfg["mode"] in ["single", "all"]

        # test a single checkpoint
        if self.cfg["mode"] == "single":
            # assert os.path.exists(self.cfg["checkpoint"])
            # load_checkpoint(
            #     model=self.model,
            #     optimizer=None,
            #     filename=self.cfg["checkpoint"],
            #     map_location=self.device,
            #     logger=self.logger,
            # )
            self.model.to(self.device)
            self.inference()
            self.evaluate()

        # test all checkpoints in the given dir
        if self.cfg["mode"] == "all":
            checkpoints_list = []
            for _, _, files in os.walk(self.cfg["checkpoints_dir"]):
                checkpoints_list = [
                    os.path.join(self.cfg["checkpoints_dir"], f)
                    for f in files
                    if f.endswith(".pth")
                ]
            checkpoints_list.sort(key=os.path.getmtime)

            for checkpoint in checkpoints_list:
                load_checkpoint(
                    model=self.model,
                    optimizer=None,
                    filename=checkpoint,
                    map_location=self.device,
                    logger=self.logger,
                )
                self.model.to(self.device)
                self.inference()
                self.evaluate()

    def inference(self):

        ## TODO: inference helper for rpn network. Apply NMS to the output of the network with merged predictions from two heads.
        torch.set_grad_enabled(False)
        self.model.eval()

        results = {}
        progress_bar = tqdm.tqdm(
            total=len(self.dataloader), leave=True, desc="Evaluation Progress"
        )
        for batch_idx, (inputs, targets, info) in enumerate(self.dataloader):
            # load evaluation data and move data to GPU.
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

            outputs, (dim, rot_cls, rot_reg, loc) = self.model(
                rgb, hha, proposals_2d.type(torch.float32).to(self.device)
            )

            dets = extract_dets_from_outputs(outputs=outputs, K=self.max_objs)
            dets = dets.detach().cpu().numpy()

            # get corresponding calibs & transform tensor to numpy
            calibs = [
                self.dataloader.dataset.get_calib(index) for index in info["img_id"]
            ]
            info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            cls_mean_size = self.dataloader.dataset.cls_mean_size
            dets = decode_detections(
                dets=dets,
                info=info,
                calibs=calibs,
                cls_mean_size=cls_mean_size,
                threshold=self.cfg.get("threshold", 0.2),
            )

            dim = dim.detach().cpu().numpy()
            rot_cls = rot_cls.detach().cpu().numpy()
            rot_reg = rot_reg.detach().cpu().numpy()
            loc = loc.detach().cpu().numpy()

            dets_rcnn = extract_dets_from_rpn(
                dim,
                rot_cls,
                rot_reg,
                loc,
                targets["2d_bbox"],
                info,
                calibs,
                self.dataloader.dataset.cls_mean_size,
            )

            results.update(dets_rcnn)
            progress_bar.update()
            break
        progress_bar.close()

        # save the result for evaluation.
        self.logger.info("==> Saving ...")
        self.save_results(results)

    def save_results(self, results, output_dir="./outputs"):
        output_dir = os.path.join(output_dir, "data")
        os.makedirs(output_dir, exist_ok=True)

        for img_id in results.keys():
            if self.dataset_type == "KITTI":
                output_path = os.path.join(output_dir, "{:06d}.txt".format(img_id))
            else:
                os.makedirs(
                    os.path.join(
                        output_dir, self.dataloader.dataset.get_sensor_modality(img_id)
                    ),
                    exist_ok=True,
                )
                output_path = os.path.join(
                    output_dir,
                    self.dataloader.dataset.get_sensor_modality(img_id),
                    self.dataloader.dataset.get_sample_token(img_id) + ".txt",
                )

            f = open(output_path, "w")
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write("{} 0.0 0".format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(" {:.2f}".format(results[img_id][i][j]))
                f.write("\n")
            f.close()

    def evaluate(self):
        self.dataloader.dataset.eval(results_dir="./outputs/data", logger=self.logger)
