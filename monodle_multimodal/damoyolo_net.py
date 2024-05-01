from lib.backbones.backbones.tinynas_mob import TinyNAS
import ast

print("imported")

with open(
    "lib/backbones/damoyolo_structure_small.txt",
    "r",
) as f:
    structure_info = f.read()

struct_str = "".join([x.strip() for x in structure_info])
struct_info = ast.literal_eval(struct_str)
for layer in struct_info:
    if "nbitsA" in layer:
        del layer["nbitsA"]
    if "nbitsW" in layer:
        del layer["nbitsW"]
backbone = TinyNAS(
    structure_info=struct_info,
    out_indices=(1, 2, 3, 4, 5),
    with_spp=True,
    depthwise=True,
)

from lib.backbones.dlaup import DLAUp
from lib.backbones.giraffe_fpn_btn import GiraffeNeckV2

# neck = GiraffeNeckV2(
#     depth=0.5,
#     hidden_ratio=0.5,
#     in_channels=[40, 80, 160],
#     out_channels=[40, 80, 160],
#     act="silu",
#     spp=False,
#     block_name="BasicBlock_3x3_Reverse",
#     depthwise=True,
# )
import torch

tensor = torch.randn(1, 3, 384, 1280)

features = backbone(tensor)
print(features[0].shape)
breakpoint()
