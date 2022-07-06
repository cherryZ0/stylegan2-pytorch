import numpy as np
import json 
from model import Generator
import torch 


latents = np.load('np_latent.npy')

dic = json.load(open('index.json', 'r'))

keys = list(dic.keys())

g_ema = Generator(
        512, 512, 8, channel_multiplier=2).to('cuda')
checkpoint = torch.load("models/star_512p_cm_2.pt")

g_ema.load_state_dict(checkpoint["g_ema"])

for key in keys:
    index = np.array(dic[key]).reshape(-1) 
    index = index == 1
    l = latents[index]
    l = torch.from_numpy(l).to("cuda")
    l = g_ema.style(l).mean(0, keepdim=True)
    l = l.cpu().detach().numpy()
    np.save("new_latents/{}.npy".format(key), l)
    print(key)
