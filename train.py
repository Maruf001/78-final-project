#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.


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
from StratifiedSampler import StratifiedSampler
from torch.utils.data import DataLoader


def train_cifar10(
    lr=0.1,  # learning rate
    resume=False,  # resume from checkpoint
    model="ResNet18",  # model type
    name="0",  # name of run
    seed=0,  # random seed
    batch_size=250,  # batch size
    n_epochs=100,  # total epochs to run
    augment=True,  # use standard augmentation
    decay=1e-4,  # weight decay
    mixup="standard",  # mixup type, either "standard", "weighted", or "erm"
    alpha=1.0,  # mixup interpolation coefficient
    gamma=0.125,  # weighted mixup regularizing coefficient
    mu=0.9,  # confusion matrix exponential moving average momentum
    live=False,  # print live progress bar
    resume_epoch=0,  # resume from epoch
    gamma_growth_factor=1.5,  # gamma multiplied by this factor every step_size epochs
    step_size=20,
):
    if live:
        from utils import progress_bar

    torch.manual_seed(seed)
    use_cuda = torch.cuda.is_available()

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print("==> Preparing data..")
    if augment:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
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

    # Use a custom stratified data loaded to ensure that the class distribution is the same in each batch
    sampler = StratifiedSampler(trainset, batch_size=batch_size)
    trainloader = DataLoader(trainset, batch_sampler=sampler)

    testset = datasets.CIFAR10(
        root="~/data", train=False, download=False, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    # Model
    if resume:
        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
        checkpoint = torch.load(
            f"./checkpoint/ckpt.t7_{name}_{model}_{mixup}_{gamma}_{resume_epoch}"
        )
        net = checkpoint["net"]
        start_epoch = checkpoint["epoch"] + 1
        rng_state = checkpoint["rng_state"]
        torch.set_rng_state(rng_state)
        gamma **= start_epoch // step_size + 1
    else:
        print("==> Building model..")
        net = models.__dict__[model]()

    if not os.path.isdir("results"):
        os.mkdir("results")
    logname = f"results/log_{name}_{model}_{mixup}_{gamma}.csv"

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net)
        print(torch.cuda.device_count())
        cudnn.benchmark = True
        print("Using CUDA..")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=decay)

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def standard_mixup(x, y, alpha, use_cuda=True):
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

    def make_doubly_stochastic(matrix, n=100):
        """Convert a matrix to a doubly stochastic matrix using Sinkhorn-Knopp algorithm."""
        for _ in range(n):
            # Row normalization
            matrix = matrix / matrix.sum(axis=1, keepdims=True)
            # Column normalization
            matrix = matrix / matrix.sum(axis=0, keepdims=True)
        return matrix

    def weighted_mixup(x1, y1, cm, alpha, gamma, epoch, use_cuda=True):
        batch_size = x1.size()[0]

        # calculate current gamma
        if epoch > 0 and epoch % step_size == 0:
            gamma *= gamma_growth_factor

        # Raise the confusion matrix to the power of gamma (element-wise) and normalize it
        matrix = cm**gamma
        matrix = make_doubly_stochastic(matrix)

        x2_idx = []
        remaining_indices = list(range(batch_size))
        for class_label in range(10):
            # Create weights for each remaining example
            example_weights = [matrix[class_label][y1[j]] for j in remaining_indices]
            weights = np.asarray(example_weights).astype(
                np.float64
            )  # float64 precision
            weights = weights / weights.sum()

            # Sample examples without replacement
            sampled_indices = np.random.choice(
                remaining_indices, batch_size // 10, p=weights, replace=False
            )
            x2_idx.extend(sampled_indices)
            remaining_indices = [
                j for j in remaining_indices if j not in sampled_indices
            ]

        if use_cuda:
            x2_idx = torch.LongTensor(x2_idx).cuda()

        x2 = x1[x2_idx]
        y2 = y1[x2_idx]

        # Mix x1 and x2
        lam = np.random.beta(alpha, alpha)
        mixed_x = lam * x1 + (1 - lam) * x2

        return mixed_x, y1, y2, lam

    # Confusion matrix initialized to EV of a random classifer
    cm = 1 / 10 * torch.ones(10, 10)
    cms_tensor = torch.zeros(10, 10, n_epochs)

    def train(epoch):
        nonlocal best_acc, cm, cms_tensor
        epoch_cm = torch.zeros_like(cm)

        # store the confusion matrix used for each epoch
        cms_tensor[:, :, epoch] = cm

        print("\nEpoch: %d" % epoch)
        net.train()
        train_loss = 0
        reg_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # ---------- ASSERT THAT THE BATCH IS STRATIFIED ----------
            unique_targets = set(targets.tolist())
            assert (
                len(unique_targets) == 10
            ), "Batch should contain samples from all classes"
            assert all(
                [
                    targets.tolist().count(label) == batch_size // 10
                    for label in unique_targets
                ]
            ), "Each class should have the same number of samples in the batch"
            ## ---------- END OF ASSERTION ----------

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            if mixup == "standard":
                inputs, targets_a, targets_b, lam = standard_mixup(
                    inputs, targets, alpha, use_cuda
                )
            elif mixup == "weighted":
                inputs, targets_a, targets_b, lam = weighted_mixup(
                    inputs, targets, cm, alpha, gamma, epoch, use_cuda
                )
            elif mixup == "erm":
                inputs, targets_a, targets_b, lam = inputs, targets, targets, 1.0

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

            # ----------- Confusion matrix (start) --------------
            # convert NN output to a probability distribution
            predicted_softmax = torch.nn.functional.softmax(outputs.data, dim=1)

            # put expected labels for the batch into a matrix
            # use += in case targets_a[i] == targets_b[i]
            expected = torch.zeros_like(outputs)
            for i in range(targets_a.size(0)):
                expected[i][targets_a[i]] += lam
                expected[i][targets_b[i]] += 1 - lam

            # set the predicted value for any class that is being mixed up to 0
            for i in range(targets_a.size(0)):
                predicted_softmax[i][targets_a[i]] = 0
                predicted_softmax[i][targets_b[i]] = 0

            # calculate epoch confusion matrix
            epoch_cm += torch.matmul(expected.cpu().t(), predicted_softmax.cpu())
            # ----------- Confusion matrix (end) --------------

            if live:
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

        # ----------- Confusion matrix (start) --------------
        # turn each row into a probability distribution
        epoch_cm = 0.9 * epoch_cm / epoch_cm.sum(axis=1, keepdims=True)
        # make the diagonal have probability 0.1
        epoch_cm = epoch_cm + 0.1 * torch.eye(10)
        # compute the exponential moving average
        cm = mu * cm + (1 - mu) * epoch_cm

        print("cm after epoch", epoch, cm)
        # ----------- Confusion matrix (end) --------------

        train_acc = (100.0 * correct / total).item()
        return (train_loss / batch_idx, reg_loss / batch_idx, train_acc)

    def test(epoch):
        nonlocal best_acc

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
            if live:
                progress_bar(
                    batch_idx,
                    len(testloader),
                    "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (
                        test_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        correct,
                        total,
                    ),
                )

        acc = 100.0 * correct / total
        checkpoint(epoch)
        if acc > best_acc:
            best_acc = acc

        test_loss = test_loss / batch_idx
        test_acc = (100.0 * correct / total).item()
        return test_loss, test_acc

    def checkpoint(epoch):
        print("Saving..")
        state = {
            "net": net,
            "epoch": epoch,
            "rng_state": torch.get_rng_state(),
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        torch.save(
            state, f"./checkpoint/ckpt.t7_{name}_{model}_{mixup}_{gamma}_{epoch}"
        )

    def adjust_learning_rate(optimizer, epoch):
        nonlocal lr
        if epoch == 50 or epoch == 75:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    if not os.path.exists(logname):
        with open(logname, "w") as logfile:
            logwriter = csv.writer(logfile, delimiter=",")
            logwriter.writerow(
                [
                    "epoch",
                    "train loss",
                    "reg loss",
                    "train acc",
                    "test loss",
                    "test acc",
                ]
            )

    for epoch in range(start_epoch, n_epochs):
        train_loss, reg_loss, train_acc = train(epoch)
        print("done training epoch")
        test_loss, test_acc = test(epoch)
        adjust_learning_rate(optimizer, epoch)
        with open(logname, "a") as logfile:
            logwriter = csv.writer(logfile, delimiter=",")
            logwriter.writerow(
                [epoch, train_loss, reg_loss, train_acc, test_loss, test_acc]
            )

    # save the confusion matrices
    torch.save(
        cms_tensor,
        f"results/cm_{name}_{model}_{mixup}_{gamma}_gammagrowth={gamma_growth_factor}_step={step_size}.pt",
    )


if __name__ == "__main__":

    # train_cifar10(mixup="standard")

    # exponential growth of gamma --> going from low to roughly 1 over training epochs
    train_cifar10(
        name="gamma-growth",
        mixup="weighted",
        gamma=0.25,
        step_size=20,
        gamma_growth_factor=1.5,
        mu=0.0,
    )
    train_cifar10(
        name="gamma-growth",
        mixup="weighted",
        gamma=0.25,
        step_size=20,
        gamma_growth_factor=2,
        mu=0.0,
    )

    train_cifar10(
        name="gamma-growth",
        mixup="weighted",
        gamma=0.2,
        step_size=20,
        gamma_growth_factor=1.5,
        mu=0.0,
    )
    train_cifar10(
        name="gamma-growth",
        mixup="weighted",
        gamma=0.2,
        step_size=20,
        gamma_growth_factor=2,
        mu=0.0,
    )

    # train_cifar10(mixup="weighted", gamma=0.25)
    # train_cifar10(mixup="weighted", gamma=0.125)
    # train_cifar10(mixup="weighted", gamma=1)
    # train_cifar10(mixup="weighted", gamma=2)
    # train_cifar10(mixup="weighted", gamma=4)
    # train_cifar10(mixup="weighted", gamma=8)
    # train_cifar10(mixup="erm", name="decay1e-2", decay=1e-2)
    # train_cifar10(mixup="erm", name="decay1e-3", decay=1e-3)
    # train_cifar10(mixup="erm", name="decay1e-4", decay=1e-4)
    # train_cifar10(mixup="weighted", name="mu0", gamma=1, mu=0.0)
    # train_cifar10(mixup="weighted", name="mu05", gamma=1, mu=0.5)
    # train_cifar10(mixup="weighted", name="mu0", gamma=0.125, mu=0.0)
    # train_cifar10(mixup="weighted", name="mu05", gamma=0.125, mu=0.5)
    # train_cifar10(mixup="weighted", name="mu0", gamma=0.25, mu=0.0)
    # train_cifar10(mixup="weighted", name="mu05", gamma=0.25, mu=0.5)
    # train_cifar10(mixup="weighted", name="mu0", gamma=0.5, mu=0.0)
    # train_cifar10(mixup="weighted", name="mu05", gamma=0.5, mu=0.5)
    # train_cifar10(mixup="weighted", name="mu05_alpha02", gamma=1, mu=0.5, alpha=0.2)
    # train_cifar10(mixup="standard", name="alpha02", alpha=0.2)
    # train_cifar10(mixup="weighted", name="mu05_alpha01", gamma=1, mu=0.5, alpha=0.1)
    # train_cifar10(mixup="standard", name="alpha01", alpha=0.1)
    pass
