# Copyright (c) 2018-present, Royal Bank of Canada and other authors.
# See the AUTHORS.txt file for a list of contributors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from advertorch.attacks import LocalSearchAttack
from advertorch.attacks import SinglePixelAttack
from advertorch.attacks import SpatialTransformAttack
from advertorch.attacks import JacobianSaliencyMapAttack
from advertorch.attacks import LBFGSAttack
from advertorch.attacks import CarliniWagnerL2Attack
from advertorch.attacks import DDNL2Attack
from advertorch.attacks import FastFeatureAttack
from advertorch.attacks import MomentumIterativeAttack
from advertorch.attacks import LinfPGDAttack
from advertorch.attacks import SparseL1DescentAttack
from advertorch.attacks import L1PGDAttack
from advertorch.attacks import L2BasicIterativeAttack
from advertorch.attacks import GradientAttack
from advertorch.attacks import LinfBasicIterativeAttack
from advertorch.attacks import GradientSignAttack
from advertorch.attacks import ElasticNetL1Attack
from advertorch.attacks import LinfSPSAAttack
from advertorch.attacks import LinfFABAttack
from advertorch.attacks import L2FABAttack
from advertorch.attacks import L1FABAttack
from advertorch.defenses import JPEGFilter
from advertorch.defenses import BitSqueezing
from advertorch.defenses import MedianSmoothing2D
from advertorch.defenses import AverageSmoothing2D
from advertorch.defenses import GaussianSmoothing2D
from advertorch.defenses import BinaryFilter

from plnn.modules import View, Flatten

DIM_INPUT = 15
NUM_CLASS = 5
BATCH_SIZE = 16

IMAGE_SIZE = 16
COLOR_CHANNEL = 3


# ###########################################################
# model definitions for testing


class SimpleModel(nn.Module):
    def __init__(self, dim_input=DIM_INPUT, num_classes=NUM_CLASS):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(dim_input, 10)
        self.fc2 = nn.Linear(10, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class SimpleImageModel(nn.Module):

    def __init__(self, num_classes=NUM_CLASS):
        super(SimpleImageModel, self).__init__()
        self.num_classes = NUM_CLASS
        self.conv1 = nn.Conv2d(
            COLOR_CHANNEL, 8, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(4)
        self.linear1 = nn.Linear(4 * 4 * 8, self.num_classes)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        return out


class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(7 * 7 * 64, 200)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(200, 10)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out

class LeNet5_nomaxpool(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(7 * 7 * 64, 200)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(200, 10)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out


#################################################################################
########################## MODELS FOR SIMPLEX PAPER #############################

## 14*14*8 (1568) --> 14*14*8 (1568) --> 14*14*8 (1568) --> 392 --> 100 (5196 ReLUs)
class mnist_model_deep(nn.Module):

    def __init__(self):
        super(mnist_model_deep, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(8, 8, 4, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        self.linear1 = nn.Linear(8*7*7,100)
        self.relu5 = nn.ReLU()
        self.linear2 = nn.Linear(100, 10)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))
        out = self.relu4(self.conv4(out))
        out = out.view(out.size(0), -1)
        out = self.relu5(self.linear1(out))
        out = self.linear2(out)
        return out

# first medium size model 14x14x4 (784) --> 7x7x8 (392) --> 50 --> 10 ----(1226 ReLUs)
# robust error 0.068
class mnist_model_med(nn.Module):

    def __init__(self):
        super(mnist_model_med, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 8, 4, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.linear1 = nn.Linear(8*7*7,50)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(50, 10)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out

#############
#### wide mnist model

## original kw small model
## 14x14x16 (3136) --> 7x7x32 (1568) --> 100 --> 10 ----(4804 ReLUs)
class mnist_model_wide(nn.Module):

    def __init__(self):
        super(mnist_model_wide, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.linear1 = nn.Linear(32*7*7,100)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(100, 10)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out

####### CIFAR

# 16*16*8 (2048) --> 16*8*8 (1024) --> 100 
# 3172 ReLUs (base model)
class cifar_model_base(nn.Module):

    def __init__(self):
        super(cifar_model_base, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, 4, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.linear1 = nn.Linear(16*8*8,100)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(100, 10)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out

# 16*16*16 (4096) --> 32*8*8 (2048) --> 100 
# 6244 ReLUs
# wide model
class cifar_model_wide(nn.Module):

    def __init__(self):
        super(cifar_model_wide, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.linear1 = nn.Linear(32*8*8,100)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(100, 10)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out

# 16*16*8 (2048) -->  16*16*8 (2048) --> 16*16*8 (2048) --> 512 --> 100
# 6756 ReLUs
#deep model
class cifar_model_deep(nn.Module):

    def __init__(self):
        super(cifar_model_deep, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(8, 8, 4, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        self.linear1 = nn.Linear(8*8*8, 100)
        self.relu5 = nn.ReLU()
        self.linear2 = nn.Linear(100, 10)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))
        out = self.relu4(self.conv4(out))
        out = out.view(out.size(0), -1)
        out = self.relu5(self.linear1(out))
        out = self.linear2(out)
        return out

# 16*16*8 (2048) -->  16*16*8 (2048) --> 16*16*8 (2048) --> 16*16*8 (2048) --> 16*16*8 (2048) --> 512 --> 100 --> 100
# 10952 ReLUs
#deeper model
class cifar_model_deeper(nn.Module):

    def __init__(self):
        super(cifar_model_deeper, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(8, 8, 4, stride=2, padding=1)
        self.relu6 = nn.ReLU()
        self.linear1 = nn.Linear(8*8*8, 100)
        self.relu7 = nn.ReLU()
        self.linear2 = nn.Linear(100, 100)
        self.relu8 = nn.ReLU()
        self.linear3 = nn.Linear(100, 10)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))
        out = self.relu4(self.conv4(out))
        out = self.relu5(self.conv5(out))
        out = self.relu6(self.conv6(out))
        out = out.view(out.size(0), -1)
        out = self.relu7(self.linear1(out))
        out = self.relu8(self.linear2(out))
        out = self.linear3(out)
        return out

# 32*32*32 (32768) --> 32*16*16 (8192) --> 64*16*16 (16384) --> 64*8*8 (4096) --> 512 --> 512 
# 54272 ReLUs
class cifar_model_large(nn.Module):

    def __init__(self):
        super(cifar_model_large, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        self.linear1 = nn.Linear(64*8*8,512)
        self.relu5 = nn.ReLU()
        self.linear2 = nn.Linear(512,512)
        self.relu6 = nn.ReLU()
        self.linear3 = nn.Linear(512,10)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))
        out = self.relu4(self.conv4(out))
        out = out.view(out.size(0), -1)
        out = self.relu5(self.linear1(out))
        out = self.relu6(self.linear2(out))
        out = self.linear3(out)
        return out

# 16*16*16 (4096) --> 32*8*8 (2048) --> 100 
# 6244 ReLUs
# wide model
def cifar_model_wide_orig():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model
#################################################################################
#################################################################################


class MLP(nn.Module):
    # MLP-300-100

    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 300)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(300, 100)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(100, 10)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.linear1(out)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out)
        return out


# ###########################################################
# model and data generation functions for testing


def generate_random_toy_data(clip_min=0., clip_max=1.):
    data = torch.Tensor(BATCH_SIZE, DIM_INPUT).uniform_(clip_min, clip_max)
    label = torch.LongTensor(BATCH_SIZE).random_(NUM_CLASS)
    return data, label


def generate_random_image_toy_data(clip_min=0., clip_max=1.):
    data = torch.Tensor(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE).uniform_(
        clip_min, clip_max)
    label = torch.LongTensor(BATCH_SIZE).random_(NUM_CLASS)
    return data, label


def generate_data_model_on_vec():
    data, label = generate_random_toy_data()
    model = SimpleModel()
    model.eval()
    return data, label, model


def generate_data_model_on_img():
    data, label = generate_random_image_toy_data()
    model = SimpleImageModel()
    model.eval()
    return data, label, model


# ###########################################################
# construct data needed for testing
vecdata, veclabel, vecmodel = generate_data_model_on_vec()
imgdata, imglabel, imgmodel = generate_data_model_on_img()


# ###########################################################
# construct groups and configs needed for testing defenses


defense_kwargs = {
    BinaryFilter: {},
    BitSqueezing: {"bit_depth": 4},
    MedianSmoothing2D: {},
    GaussianSmoothing2D: {"sigma": 3, "channels": COLOR_CHANNEL},
    AverageSmoothing2D: {"kernel_size": 5, "channels": COLOR_CHANNEL},
    JPEGFilter: {},
}

defenses = defense_kwargs.keys()

# store one suitable data for test
defense_data = {
    BinaryFilter: vecdata,
    BitSqueezing: vecdata,
    MedianSmoothing2D: imgdata,
    GaussianSmoothing2D: imgdata,
    AverageSmoothing2D: imgdata,
    JPEGFilter: imgdata,
}

nograd_defenses = [
    BinaryFilter,
    BitSqueezing,
    JPEGFilter,
]

withgrad_defenses = [
    MedianSmoothing2D,
    GaussianSmoothing2D,
    AverageSmoothing2D,
]

image_only_defenses = [
    MedianSmoothing2D,
    GaussianSmoothing2D,
    AverageSmoothing2D,
    JPEGFilter,
]

# as opposed to image-only
general_input_defenses = [
    BitSqueezing,
    BinaryFilter,
]


# ###########################################################
# construct groups and configs needed for testing attacks


# as opposed to image-only
general_input_attacks = [
    GradientSignAttack,
    LinfBasicIterativeAttack,
    GradientAttack,
    L2BasicIterativeAttack,
    LinfPGDAttack,
    MomentumIterativeAttack,
    FastFeatureAttack,
    CarliniWagnerL2Attack,
    ElasticNetL1Attack,
    LBFGSAttack,
    JacobianSaliencyMapAttack,
    SinglePixelAttack,
    DDNL2Attack,
    SparseL1DescentAttack,
    L1PGDAttack,
    LinfSPSAAttack,
    LinfFABAttack,
    L2FABAttack,
    L1FABAttack,
]

image_only_attacks = [
    SpatialTransformAttack,
    LocalSearchAttack,
]

label_attacks = [
    GradientSignAttack,
    LinfBasicIterativeAttack,
    GradientAttack,
    L2BasicIterativeAttack,
    LinfPGDAttack,
    MomentumIterativeAttack,
    CarliniWagnerL2Attack,
    ElasticNetL1Attack,
    LBFGSAttack,
    JacobianSaliencyMapAttack,
    SpatialTransformAttack,
    DDNL2Attack,
    SparseL1DescentAttack,
    L1PGDAttack,
    LinfSPSAAttack,
    LinfFABAttack,
    L2FABAttack,
    L1FABAttack,
]

feature_attacks = [
    FastFeatureAttack,
]

batch_consistent_attacks = [
    GradientSignAttack,
    LinfBasicIterativeAttack,
    GradientAttack,
    L2BasicIterativeAttack,
    LinfPGDAttack,
    MomentumIterativeAttack,
    FastFeatureAttack,
    JacobianSaliencyMapAttack,
    DDNL2Attack,
    SparseL1DescentAttack,
    L1PGDAttack,
    LinfSPSAAttack,
    # FABAttack,
    # CarliniWagnerL2Attack,  # XXX: not exactly sure: test says no
    # LBFGSAttack,  # XXX: not exactly sure: test says no
    # SpatialTransformAttack,  # XXX: not exactly sure: test says no
]


targeted_only_attacks = [
    JacobianSaliencyMapAttack,
]

# attacks that can take vector form of eps and eps_iter
vec_eps_attacks = [
    LinfBasicIterativeAttack,
    L2BasicIterativeAttack,
    LinfPGDAttack,
    FastFeatureAttack,
    SparseL1DescentAttack,
    L1PGDAttack,
    GradientSignAttack,
    GradientAttack,
    MomentumIterativeAttack,
    LinfSPSAAttack,
]

# ###########################################################
# helper functions


def merge2dicts(x, y):
    z = x.copy()
    z.update(y)
    return z
