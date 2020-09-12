import os
from tqdm import tqdm
import torch
import json
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
# Parse path as argument
parser = argparse.ArgumentParser(description='Images for classification')
parser.add_argument('indir', type=str, help='Input dir for images')
args = parser.parse_args()

# CUDA activations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CNN model class
class CNN(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1)
            )
        self.pooling = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)
            )
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1)
            )
        self.fc_layer1 = nn.Linear(16*8*8, NUM_CLASSES)
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

    # Weights initialization 'Kaiming Xe' for RELU function
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# Transformations to same size as in train
transformations = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                          std = [0.229, 0.224, 0.225])
    ])

# Creating test dataset from folder provided by path
dataset = ImageFolder(args.indir, transform=transformations)
test_loader = DataLoader(dataset=dataset,shuffle=False)

# Load pretrained model
model = torch.load("my_model.pth")

model.eval()
#store predictions
preds = []
for idx, (data, labels) in enumerate(test_loader):
    output = model(data)
    pred = torch.argmax(output, 1)
    pred = pred.detach().cpu()
    # convert int answers to string
    if pred.item() == 0:
        preds.append('female')
    else:
        preds.append('male')

# Get sample names in order how they are in dataset and calculation
names = [x[0] for x in dataset.samples]
# Store them in list
file_names = []
for name in names:
    item_name = os.path.basename(name)
    file_names.append(item_name)

# Create dictionary with answers and file names
process_results = {}
for i in range(len(file_names)):
    process_results[file_names[i]] = preds[i]
#export to json
with open("process_results.json", "wb") as f:
    f.write(json.dumps(process_results).encode("utf-8"))
