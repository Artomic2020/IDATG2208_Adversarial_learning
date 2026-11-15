import torch  #pytorch for deep learning and datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt  # for plotting graphs
import numpy as np   #for numerical operations 
 
 
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(), 
    transforms.RandomCrop(32, padding=4),  #(**For later FGSM/PGD evaluation, train=False and omit the random flip and crop)
    transforms.ToTensor(),     # converts CIFAR-10’s pixel values from integers in [0, 255] to floats in [0, 1]
    transforms.Normalize((0.4914, 0.4822, 0.4465),  # mean and a standard deviation for each color channel, in red, green, blue
                         (0.2023, 0.1994, 0.2010))  # Normalize each RGB channel: (pixel - channel_mean) / channel_std  Also can be written: transforms.Normalize(mean, std)
])

mean = (0.5071, 0.4867, 0.4408) # mean and a standard deviation for each color channel, in red, green, blue
std = (0.2675, 0.2565, 0.2761)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)  # Normalize each RGB channel: (pixel - channel_mean) / channel_std # 
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
seed = 42
torch.manual_seed(seed)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

classes = trainset.classes

print(classes)


# Define the residual block

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels !=out_channels:
            self.shortcut = nn.Sequential( 
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out + self.shortcut(x)
        out = self.relu(out)
        return(out)
    #between loading a batch and computing the training loss
#Define ResNet18 Architecture

class ResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers=[]
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18().to(device)

    
model = ResNet18().to(device)
print(model)   

criterion = nn.CrossEntropyLoss()  # Cross entropy loss is used for multi-class classification
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4) #Stochastic gradient descent(SGD) with momentum helps faster convergence
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) #StepLR scheduler reduces learning rate 

#Train the model

num_epochs = 10
train_losses, train_acc_list, test_acc_list = [], [], []

for epoch in range (num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0,0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
   
        # Start (PGD) here with a copy of the clean batch
        
        adv_inputs = inputs.detach().clone()   #detach() starts the attack from a tensor with no gradient history. clone()
            
            #clone() gives adv_inputs its own storage so we can perturb it and still use inputs as a reference
        
        
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()\  
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_loss = running_loss / len(trainloader.dataset)
    train_acc = 100. * correct/total
    train_losses.append(train_loss)
    train_acc_list.append(train_acc)
    
    
   
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_acc= 100. * correct / total
    test_acc_list.append(test_acc)
    
    
    print("Training on:", device, flush=True)
    scheduler.step()
    print(f'Epoch [{epoch+1}/{num_epochs}]  Train Loss: {train_loss: .4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc: .2f}%')
    print("Test images counted:", total)

  


        