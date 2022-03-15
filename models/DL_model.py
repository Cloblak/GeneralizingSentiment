import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchsummary import summary
from transformers import AutoModel, BertTokenizerFast, BertModel
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary









# Train the model
def __train(
        model,
        criterion,
        optimizer,
        dataloaders,
        scheduler,
        device,
        dataset_sizes,
        num_epochs=10,
    ):
    """
    Model training
    :param model:
    :param criterion:
    :param optimizer:
    :param dataloaders:
    :param scheduler:
    :param device:
    :param num_epochs:
    :return:
    """
    model = model.to(device)  # Send model to GPU if available
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for (labels, text, offsets) in dataloaders[phase]:
                text = text.to(device)
                labels = labels.to(device)
                offsets = offsets.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass to get outputs and calculate loss
                # Track gradient only for training data
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model.forward(text, offsets)
                    loss = criterion(outputs, labels)

                    # Backpropagation to get the gradients with respect to each weight
                    # Only if in train
                    if phase == 'train':
                        loss.backward()
                        # Update the weights
                        optimizer.step()

                # statistics
                # Convert loss into a scalar and add it to running_loss
                running_loss += loss.item() * labels.size(0)
                # Track number of correct predictions
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

                # Step along learning rate scheduler when in train
            if phase == 'train':
                scheduler.step()

            # Calculate and display average loss and accuracy for the epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} loss: {:.4f} accuracy: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # If model performs better on val set, save weights as the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best validation set accuracy: {:3f}'.format(best_acc))

    # Load the weights from best model
    model.load_state_dict(best_model_wts)

    return model



def evaluate(dataloader, model, device):
    # Generate predictions and calculate accuracy
    model.to(device)
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predited_label = model.forward(text, offsets)
            #loss = criterion(predited_label, label)
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count


def train_model(
    text, dataloaders, batch_size, class_names, dataset_sizes, num_epochs=10
):
    # We will used a pre-trained bert-base-uncased model
    # displays a summary of the model layers and the output shape of the input after passing through each layer.
    # Instantiate pre-trained resnet
    net = BertModel.from_pretrained('bert-base-uncased')


    # Display a summary of the layers of the model and output shape after each layer
    summary(net, (text.shape[1:]), batch_size=batch_size, device="cpu")

    # Get the number of inputs to final Linear layer
    num_ftrs = net.pooler.dense.in_features

    # Replace final Linear layer with a new Linear with the same number of inputs
    # since we have 3 classes
    net.fc = nn.Linear(in_features=num_ftrs, out_features=3)

    # We will use Cross Entropy as the cost/loss function and Adam for the optimizer.

    # Cross entropy loss combines softmax and nn.NLLLoss() in one single class.
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)

    # Decay LR by a factor of 0.1 every 7 epochs (not used since Adam converges better)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        net.cuda()

    # Train the model
    net, cost_path = __train(
        net,
        criterion,
        optimizer,
        dataloaders,
        device,
        dataset_sizes,
        num_epochs,
    )

    # Test the pre-trained model
    acc, recall_vals = evaluate(net, dataloaders["val"], device)
    print("Test set accuracy is {:.3f}".format(acc))
    for i in range(3):
        print("For class {}, recall is {}".format(class_names[i], recall_vals[i]))
    plt.plot(cost_path)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    print("All done!")
    return net