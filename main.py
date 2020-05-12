import torch
from torch import nn
from torch import optim
from torchvision import datasets,transforms,models
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 10

data_dir = '.\hymenoptera_data'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(data_dir+'\\train',transform=train_transforms)
test_data = datasets.ImageFolder(data_dir+'\\test',transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=64)

model = models.densenet121(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 500)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(500, 2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.classifier.parameters(),lr=0.03)

model.to(device)

for epoch in range(epochs):
    running_loss = 0
    running_corrects = 0
    for inputs,labels in train_loader:
        inputs,labels = inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        _,preds = torch.max(logps.data,1)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data).to(torch.float32)

    epoch_loss = running_loss/len(train_data)
    epoch_acc = running_corrects/len(train_data)
    print("epoch_loss: "+str(epoch_loss),"  epoch_acc: "+str(epoch_acc.cpu().numpy()))

model.eval()
running_loss = 0
running_corrects = 0
for inputs,labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    logps = model.forward(inputs)
    _, preds = torch.max(logps.data, 1)
    loss = criterion(logps, labels)
    running_loss += loss.item()
    running_corrects += torch.sum(preds == labels.data).to(torch.float32)
epoch_loss = running_loss/len(train_data)
epoch_acc = running_corrects/len(train_data)
print("epoch_loss: "+str(epoch_loss),"  epoch_acc: "+str(epoch_acc.cpu().numpy()))