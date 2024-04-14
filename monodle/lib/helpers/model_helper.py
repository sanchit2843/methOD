from lib.models.centernet3d import CenterNet3D
from lib.models.centernet_damo import CenterNet3DDamo


def build_model(cfg):
    if cfg["type"] == "centernet3d":
        return CenterNet3D(
            backbone=cfg["backbone"], neck=cfg["neck"], num_class=cfg["num_class"]
        )
    elif cfg["type"] == "damoyolo":
        return CenterNet3DDamo(num_class=cfg["num_class"])
    else:
        raise NotImplementedError("%s model is not supported" % cfg["type"])
