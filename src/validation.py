import torch
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.optim as optim
import numpy as np


def train_epoch(model, dataloader, loss_fn, optimizer):
    train_correct = 0
    model.train()

    for i, (A, H, labels) in enumerate(dataloader):
        # Sets the gradients to zero
        optimizer.zero_grad()
        # do forward pass
        output = model((A, H))
        # calculate loss
        loss = loss_fn(output, labels)
        # backpropagate loss
        loss.backward()
        # do parameter update
        optimizer.step()
        # calculate labels from predictions
        scores, predictions = torch.max(output.data, 1)
        train_correct += (predictions == labels).sum().item()

    return train_correct


def valid_epoch(model, dataloader, loss_fn):
    val_correct = 0
    model.eval()

    for i, (A, H, labels) in enumerate(dataloader):
        # do forward pass
        output = model((A, H))
        # calculate loss
        loss = loss_fn(output, labels)
        # calculate labels from predictions
        _, predictions = torch.max(output.data, 1)
        val_correct += (predictions == labels).sum().item()

    return val_correct


def cross_val(model_func, dataset, num_epochs=15):
    criterion = nn.CrossEntropyLoss()

    splits = KFold(n_splits=10, shuffle=True)
    for fold, (train_ids, test_ids) in enumerate(
            splits.split(np.arange(len(dataset)))):
        print('Fold {}'.format(fold + 1))
        model = model_func()
        optimizer = optim.Adam(model.parameters(), lr=0.005)

        # use 10 K-fold crossvalidation indexes to create subsets of class
        # Dataloader for test and train
        train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_ids)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=15,
                                                   sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=15,
                                                  sampler=test_sampler)

        for epoch in range(num_epochs):
            train_correct = train_epoch(model, train_loader, criterion,
                                        optimizer)
            test_correct = valid_epoch(model, test_loader, criterion)

            # calculate scores
            train_acc = train_correct / len(train_loader.sampler) * 100
            test_acc = test_correct / len(test_loader.sampler) * 100
            print(
                "Epoch:{}/{}  Training Acc {:.2f} % VG Test Acc {:.2f} %".format(
                    epoch + 1, num_epochs, train_acc, test_acc))
