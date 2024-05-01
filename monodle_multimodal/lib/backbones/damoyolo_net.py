from lib.backbones.backbones.tinynas_mob import TinyNAS
import ast
print("imported")

with open("/Users/sanchittanwar/Desktop/Workspace/courses/CS7643/final_project/methOD/monodle/lib/backbones/damoyolo_structure_small.txt", "r") as f:
    structure_info = f.read()   

struct_str = ''.join([x.strip() for x in backbone_cfg.net_structure_str])
struct_info = ast.literal_eval(struct_str)
for layer in struct_info:
    if 'nbitsA' in layer:
        del layer['nbitsA']
    if 'nbitsW' in layer:
        del layer['nbitsW']
backbone = TinyNAS(structure_info=structure_info, out_indices=(2,4,5),with_spp = True, depthwise=True)

breakpoint()