# coding:utf-8
import sys
sys.path.append(".")
import os
import random
import logging
import torch
import numpy as np

def init_logger():
    """ Output log information to the console
    Params:
        asctime: print time
        levelname: print level
        name: print name
        message: print message
    """
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )

def check_path(args):
    """ Check whether the target directory exists. If no, create a directory """
    print("Start checking path...")
    if not os.path.exists(args.data_path):
        print("Creating data path...")
        os.makedirs(args.data_path)

    if not os.path.exists(args.model_path):
        print("Creating model path...")
        os.makedirs(args.model_path)
    print("Check path done.")

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    ttorch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def show_model(args):
    model_path = os.path.join(args.model_path, args.model_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    net = torch.load(model_path, map_location=torch.device(device))
    print(type(net))
    print(len(net))
    for k in net.keys():
        print(k)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True) # pred: location index of the largest number in each batch
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res = correct_k.mul_(100.0 / batch_size)
        return res