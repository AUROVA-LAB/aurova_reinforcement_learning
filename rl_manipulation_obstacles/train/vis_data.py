import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split

from data import *
from networks_lfd import *
from train_utils import collate_fn

import matplotlib.pyplot as plt
import cv2 as cv




# =========================================================
# TRAINING
# =========================================================


def vis():

    dataset = HDF5LfDDataset(os.path.join(os.getcwd(), "../dataset"))


    for i in range(len(dataset)):
        cv.imshow("aaa", np.transpose(dataset[i]["cam_D"], (1,2,0)))
        cv.waitKey(1)


if __name__ == "__main__":
    vis()
