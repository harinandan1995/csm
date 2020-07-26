import torch
from src.model.unet import UNet

params = torch.load('/mnt/raid/csmteam/out/2020-07-17/085508/checkpoints/model_021445_195')
unet = UNet(4, (3), 5)
print(unet)
for n, p in unet.named_parameters():
    print(n)

# print(params.keys())

for k, v in params.items():
    if "unet.model" in k:
        print(k.replace('unet.', ''))
        print(unet.state_dict()[k.replace('unet.', '')].size())
        print(v.size())
        unet.state_dict()[k.replace('unet.', '')].copy_(v)


# print(params.keys())
