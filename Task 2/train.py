import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from skimage import io, transform
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    """Function for saving checkpoints."""
    print('/Saving checkpoint..')
    torch.save(state, filename)


def load_checkpoint(checkpoint):
    """Function for loading checkpoints."""
    print('/Loading checkpoint..')
    torch.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Hyperparams
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
# Load model from checkpoint
load_model = False

# Dataset parameters: percentage of train data to total data, number of classes
TRAIN_SIZE_RATIO = 0.8 
NUM_CLASSES = 2

# Path to data folders
image_path = "internship_data/"
FILE = "my_model.pth"
CHECKPOINT = "my_checkpoint.pth.tar"

# Set up transformations: resize all images to 224x224
transformations = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Creating dataset with Imagefolder
dataset = ImageFolder(image_path, transform=transformations)
# Tran/Valid shapes
TRAIN_SIZE = int(len(dataset) * TRAIN_SIZE_RATIO // 1)
VALID_SIZE = int(len(dataset) - TRAIN_SIZE)
# Split the data to train and validation sets
train_set, val_set = torch.utils.data.random_split(dataset, [TRAIN_SIZE, VALID_SIZE])
# Create DataLoader iterators
train_loader = DataLoader(dataset=train_set,
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_set,
                        batch_size=BATCH_SIZE, shuffle=False)


# Create Convolution Neural Network class
class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
            )
        self.pooling = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1))
        self.fc_layer1 = nn.Linear(16*28*28, NUM_CLASSES)
        self.init_weights()

    def forward(self, x):
        # Conv1 layer with Activation function RELU afterwords
        x = F.relu(self.conv1(x))
        # Maxpooling with 2x2 size and stride 2
        x = self.pooling(x)
        # Conv3 layer with Activation function RELU afterwords
        x = F.relu(self.conv2(x))
        # Conv4 layer with Activation function RELU afterwords
        x = self.pooling(x)
        x = x.reshape(x.shape[0], -1)
        # Fully connected layer
        x = self.fc_layer1(x)
        return x

    # Weights initialization 'Kaiming He' for RELU function
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# Send to CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize CNN
model = CNN().to(device)
# in case Loading model checkpoint
if load_model:
    load_model(torch.load(CHECKPOINT))
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 factor=0.1,
                                                 patience=5,
                                                 verbose=True)


# print model parameters
def count_parameters(model):
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(model)
print(f'The model has {count_parameters(model):,} trainable parameters')

for epoch in range(NUM_EPOCHS):
    # Model checkpoints
    if epoch == 5:
        checkpoint = {'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)
    # Loop for Progression bar information
    loop = tqdm(
        enumerate(train_loader),
        total=len(train_loader), leave=False)
    # Training step
    running_loss = 0
    num_correct = 0
    lr_losses = []
    model.train()
    for batch_idx, (batch, labels) in loop:
        # Transfer to CUDA
        batch = batch.to(device=device)
        labels = labels.to(device=device)
        # Forward Pass
        scores = model(batch)
        loss = criterion(scores, labels)
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        lr_losses.append(loss)
        # Get running accuracy
        _, predictions = torch.max(scores.data, 1)
        num_correct = (predictions == labels).sum().item()
        # Progress bar
        loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        loop.set_postfix(running_loss=loss.item())

    # update scheduler
    mean_loss = sum(lr_losses) / len(lr_losses)
    scheduler.step(mean_loss)

    # Validation step
    model.eval()
    with torch.no_grad():
        # Loop for Progression bar information
        num_correct = 0
        num_samples = 0
        loop_val = tqdm(enumerate(val_loader),
                        total=len(val_loader), leave=False)
        for batch_idx, (batch, labels) in loop_val:
            # Transfer to CUDA
            batch = batch.to(device=device)
            labels = labels.to(device=device)
            # Forward Pass
            scores = model(batch)
            _, predictions = torch.max(scores.data, 1)
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)
    print(f'Valid set {num_correct} / {num_samples} \
          with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')


# Save model to file
torch.save(model, FILE)
