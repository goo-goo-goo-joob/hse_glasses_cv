import multiprocessing
import os
import pickle
import sys
import typing
import uuid
import zipfile
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models import resnet
from tqdm.auto import tqdm

class Model(torch.nn.Module):
    def __init__(self, feature_dim=128, arch="resnet18", init=True):
        super(Model, self).__init__()

        self.f = []

        if arch == "resnet18":
            w = resnet.ResNet18_Weights.DEFAULT if init else None
            module = resnet.resnet18(weights=w)
            in_size = 512
        elif arch == "resnet34":
            w = resnet.ResNet34_Weights.DEFAULT if init else None
            module = resnet.resnet34(weights=w)
            in_size = 512
        elif arch == "resnet50":
            w = resnet.ResNet50_Weights.DEFAULT if init else None
            module = resnet.resnet50(weights=w)
            in_size = 2048
        else:
            raise Exception("Unknown module {}".format(repr(arch)))
        for name, module in module.named_children():
            # if name == "conv1":
            #     module = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # if not isinstance(module, torch.nn.Linear) and not isinstance(module, torch.nn.MaxPool2d):
            if not isinstance(module, torch.nn.Linear):
                self.f.append(module)
        # encoder
        self.f = torch.nn.Sequential(*self.f)
        self.c = torch.nn.Linear(in_size, 2, bias=True)
        # projection head
        self.g = torch.nn.Sequential(
            torch.nn.Linear(in_size, 512, bias=False),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, feature_dim, bias=True),
        )

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        proba = self.c(feature)
        return F.normalize(out, dim=-1), proba