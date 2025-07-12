import glob

import numpy as np
import torch


files = sorted(glob.glob("./datasets/s3dis/normals/*.npy"))

for file in files:
    chunks = file.split("/")[-1].split(".")
    area = chunks[0]
    room = chunks[1]

    normal = np.load(file, allow_pickle=True)
    print('process:', area, room)
    torch.save((normal), f"./datasets/s3dis/normals/{area}_{room}.pth")
