import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class Conv2D_Input(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv2D_Input, self).__init__()
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.conv_input= nn.Conv2d(in_channels, out_channels, kernel_size=7, padding= 'same')
        self.maxpool_input = nn.MaxPool2d(3, padding = 1)

    def forward(self, x):
        out = self.batch_norm(x)
        out = self.conv_input(out)
        out = self.maxpool_input(out)
        return out



# MixedLink block
class MixedLinkBlock(nn.Module):
    def __init__(self, in_channels, k,downsample = None):
        super(MixedLinkBlock, self).__init__()
        self.in_channels = in_channels
        self.k = k
        self.conv1_add = nn.Conv2d(in_channels, k, kernel_size=1, padding= 'same')
        self.bn1_add = nn.BatchNorm2d(k)
        self.relu_add = nn.ReLU(inplace=True)
        self.conv2_add = nn.Conv2d(k, k, kernel_size=3, padding= 'same')
        self.bn2_add = nn.BatchNorm2d(k)


        self.conv1_concat = nn.Conv2d(in_channels, k, kernel_size=1, padding= 'same')
        self.bn1_concat = nn.BatchNorm2d(k)
        self.relu_concat = nn.ReLU(inplace=True)
        self.conv2_concat = nn.Conv2d(k, k, kernel_size=3, padding= 'same')
        self.bn2_concat = nn.BatchNorm2d(k)

        self.downsample = downsample




    def forward(self, x):
        out_add = self.conv1_add(x)
        out_add = self.bn1_add(out_add)
        out_add = self.relu_add(out_add)
        out_add = self.conv2_add(out_add)
        out_add = self.bn2_add(out_add)

        out_concat = self.conv1_concat(x)
        out_concat = self.bn1_concat(out_concat)
        out_concat = self.relu_concat(out_concat)
        out_concat = self.conv2_concat(out_concat)
        out_concat = self.bn2_concat(out_concat)

        out_add = x[:,-self.k:,:,:] + out_add
        out_add = torch.cat((x[:,:-self.k,:,:], out_add), 1)
        out = torch.cat((out_add, out_concat),1)

        if self.downsample:
            out = self.downsample(out)

        print(out.shape)
        return out

# Mixnet
class MixNet(nn.Module):
    def __init__(self, block, k, layers, num_classes=10):
        super(MixNet, self).__init__()
        self.k = k
        self.conv_input = Conv2D_Input(3, 2*k)
        self.layer1 = self.make_layer(block, 2*k, layers[0])
        self.layer2 = self.make_layer(block, 2*k + layers[0] * k, layers[1])
        self.layer3 = self.make_layer(block, 2*k + sum(layers[:-1]) * k, layers[2])
        self.avg_pool = nn.AvgPool2d(2*k + sum(layers) * k)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, in_channels, blocks):

        layers = []

        for i in range(0, blocks-1):

            layers.append(block(in_channels, self.k))
            in_channels += self.k

        downsample = nn.Sequential(
            nn.Conv2d(in_channels+self.k, in_channels+self.k, kernel_size=1, padding= 'same'),
            nn.AvgPool2d(3, stride=2))
        layers.append(block(in_channels, self.k ,downsample))

        return nn.Sequential(*layers)

    def forward(self, x):
        print(x.shape)
        out = self.conv_input(x)
        print(out.shape)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        print(' ')
        print(' ')
        return out


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MixNet(MixedLinkBlock, 12, [6, 12, 12]).to(device)


# Hyper-parameters
num_epochs = 80
learning_rate = 0.001

# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True,
                                             transform=transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                            train=False,
                                            transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'mixnet.ckpt')
