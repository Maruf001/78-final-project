#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models
from utils import progress_bar

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
parser.add_argument(
    "--model", default="ResNet18", type=str, help="model type (default: ResNet18)"
)
parser.add_argument("--name", default="0", type=str, help="name of run")
parser.add_argument("--seed", default=0, type=int, help="random seed")
parser.add_argument("--batch-size", default=128, type=int, help="batch size")
parser.add_argument("--epoch", default=200, type=int, help="total epochs to run")
parser.add_argument(
    "--no-augment",
    dest="augment",
    action="store_false",
    help="use standard augmentation (default: True)",
)
parser.add_argument("--decay", default=1e-4, type=float, help="weight decay")
parser.add_argument(
    "--alpha",
    default=1.0,
    type=float,
    help="mixup interpolation coefficient (default: 1)",
)
parser.add_argument("--variant", default="mixup", type=str, help="variant of mixup")
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    torch.manual_seed(args.seed)

# Confusion matrix (used when variant=="weighted-mixup")
# initialized to EV of a random classifer
confusion_matrix = 1 / 10 * torch.ones(10, 10)

# Data
print("==> Preparing data..")
if args.augment:
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
else:
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )


transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

trainset = datasets.CIFAR10(
    root="~/data", train=True, download=False, transform=transform_train
)
# trainloader = torch.utils.data.DataLoader(trainset,
#                                           batch_size=args.batch_size,
#                                           shuffle=True, num_workers=8)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True
)

testset = datasets.CIFAR10(
    root="~/data", train=False, download=False, transform=transform_test
)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100,
#                                          shuffle=False, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)


def make_doubly_stochastic(matrix, n=100):
    for _ in range(n):
        # Row normalization
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        # Column normalization
        matrix = matrix / matrix.sum(axis=0, keepdims=True)


def weighted_mixup(x1, y1, normalized_confusion_matrix, alpha=1.0, beta=1.0):
    # sort x1 and y1 by class label
    x1 = x1[torch.argsort(y1)]
    y1 = y1[torch.argsort(y1)]

    # Raise the confusion matrix to the power of beta (element-wise) and normalize it
    matrix = normalized_confusion_matrix**beta
    matrix = make_doubly_stochastic(matrix)

    x2 = []
    y2 = []
    x_pool = x1.copy()
    y_pool = y1.copy()

    for class_label in range(10):
        n_examples = (y1 == class_label).sum()

        # Create weights for each example
        class_weights = matrix[class_label]
        example_weights = [class_weights[y_pool[j]] for j in range(x_pool.size()[0])]
        weights = torch.tensor(example_weights)
        weights = weights / weights.sum()

        # Sample examples and remove them from the pool
        sampled_indices = np.random.choice(
            range(x_pool.size()[0]), n_examples, p=weights, replace=False
        )
        unsampled_indices = [
            i for i in range(x_pool.size()[0]) if i not in sampled_indices
        ]

        x2.extend(x_pool[sampled_indices])
        y2.extend(y_pool[sampled_indices])
        x_pool = x_pool[unsampled_indices]
        y_pool = y_pool[unsampled_indices]

    x2 = torch.tensor(x2)
    y2 = torch.tensor(y2)

    # one hot encode y1 and y2
    y1 = torch.nn.functional.one_hot(y1, 10)
    y2 = torch.nn.functional.one_hot(y2, 10)

    # Mix x1 and x2, y1 and y2
    x = torch.zeros_like(x1)
    y = torch.zeros_like(y1)
    for i in range(x1.size()[0]):
        lam = np.random.beta(alpha, alpha)
        x[i] = lam * x1[i] + (1 - lam) * x2[i]
        y[i] = lam * y1[i] + (1 - lam) * y2[i]

    return x, y


# Model
if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load("./checkpoint/ckpt.t7" + args.name + "_" + str(args.seed))
    net = checkpoint["net"]
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"] + 1
    rng_state = checkpoint["rng_state"]
    torch.set_rng_state(rng_state)
else:
    print("==> Building model..")
    net = models.__dict__[args.model]()

if not os.path.isdir("results"):
    os.mkdir("results")
logname = (
    "results/log_"
    + net.__class__.__name__
    + "_"
    + args.name
    + "_"
    + str(args.seed)
    + ".csv"
)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    print("Using CUDA..")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay
)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(epoch):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets_a, targets_b, lam = mixup_data(
            inputs, targets, args.alpha, use_cuda
        )
        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
        outputs = net(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (
            lam * predicted.eq(targets_a.data).cpu().sum().float()
            + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float()
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(
            batch_idx,
            len(trainloader),
            "Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)"
            % (
                train_loss / (batch_idx + 1),
                reg_loss / (batch_idx + 1),
                100.0 * correct / total,
                correct,
                total,
            ),
        )
    return (train_loss / batch_idx, reg_loss / batch_idx, 100.0 * correct / total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(
            batch_idx,
            len(testloader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (test_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )
    acc = 100.0 * correct / total
    if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
        checkpoint(acc, epoch)
    if acc > best_acc:
        best_acc = acc
    return (test_loss / batch_idx, 100.0 * correct / total)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print("Saving..")
    state = {"net": net, "acc": acc, "epoch": epoch, "rng_state": torch.get_rng_state()}
    if not os.path.isdir("checkpoint"):
        os.mkdir("checkpoint")
    torch.save(state, "./checkpoint/ckpt.t7" + args.name + "_" + str(args.seed))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if not os.path.exists(logname):
    with open(logname, "w") as logfile:
        logwriter = csv.writer(logfile, delimiter=",")
        logwriter.writerow(
            ["epoch", "train loss", "reg loss", "train acc", "test loss", "test acc"]
        )

for epoch in range(start_epoch, args.epoch):
    train_loss, reg_loss, train_acc = train(epoch)
    print("done training epoch")
    test_loss, test_acc = test(epoch)
    adjust_learning_rate(optimizer, epoch)
    with open(logname, "a") as logfile:
        logwriter = csv.writer(logfile, delimiter=",")
        logwriter.writerow(
            [epoch, train_loss, reg_loss, train_acc, test_loss, test_acc]
        )
