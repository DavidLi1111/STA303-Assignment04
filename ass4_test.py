# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
import os

import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as trn
from tqdm import tqdm

from torchcp.classification.predictors import ClusterPredictor, ClassWisePredictor, SplitPredictor
from torchcp.classification.scores import THR, APS, SAPS, RAPS, Margin
from torchcp.classification import Metrics
from torchcp.utils import fix_randomness
# from examples.common.dataset import build_dataset
def build_dataset(dataset_name, transform=None, mode='train'):
    # User's data directory
    usr_dir = os.path.expanduser('~')
    data_dir = os.path.join(usr_dir, "data")

    # Default transform if none provided
    if transform is None:
        if dataset_name in ['cifar10', 'cifar100', 'svhn', 'stl10']:
            # Default transform for color image datasets
            transform = trn.Compose([
                trn.Resize(256),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif dataset_name == 'mnist':
            # Default transform for MNIST
            transform = trn.Compose([
                trn.ToTensor(),
                trn.Normalize((0.1307,), (0.3081,))
            ])

    # Load specific dataset
    if dataset_name == 'imagenet':
        dataset = dset.ImageFolder(data_dir + "/imagenet/val", transform)
    elif dataset_name == 'mnist':
        dataset = dset.MNIST(data_dir, train=(mode == "train"), download=True, transform=transform)
    elif dataset_name == 'cifar10':
        dataset = dset.CIFAR10(data_dir, train=(mode == "train"), download=True, transform=transform)
    elif dataset_name == 'cifar100':
        dataset = dset.CIFAR100(data_dir, train=(mode == "train"), download=True, transform=transform)
    elif dataset_name == 'svhn':
        split = 'train' if mode == 'train' else 'test'
        dataset = dset.SVHN(data_dir, split=split, download=True, transform=transform)
    elif dataset_name == 'stl10':
        split = 'train' if mode == 'train' else 'test'
        dataset = dset.STL10(data_dir, split=split, download=True, transform=transform)
    else:
        raise NotImplementedError("The specified dataset is not supported")

    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--alpha', default=0.1, type=float)
    args = parser.parse_args()

    fix_randomness(seed=args.seed)

    #######################################
    # Loading ImageNet dataset and a pytorch model
    #######################################
    model_name = 'ResNet101'
    model = torchvision.models.resnet101(weights="IMAGENET1K_V1", progress=True)
    # model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # 修改为单通道输入
    model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(model_device)


    # dataset = build_dataset('cifar10',mode='test')

    # cal_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.5, 0.5])
    cal_dataset=build_dataset('svhn',mode='train')
    test_dataset=build_dataset('svhn',mode='test')
    cal_data_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=1024, shuffle=False, pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, pin_memory=True)

    
    #######################################
    # A standard process of conformal prediction
    #######################################    
    alpha = args.alpha
    predictors = [SplitPredictor, ClassWisePredictor, ClusterPredictor]
    score_functions = [THR(),  APS(), RAPS(1, 0)]
    for score in score_functions: 
        for class_predictor in predictors:
            predictor = class_predictor(score, model)
            predictor.calibrate(cal_data_loader, alpha)
            print(f"Experiment--Data : ImageNet, Model : {model_name}, Score : {score.__class__.__name__}, Predictor : {predictor.__class__.__name__}, Alpha : {alpha}")
            print(predictor.evaluate(test_data_loader))
