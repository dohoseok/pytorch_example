import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim, cuda
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 128

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                           shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = 'cuda' if cuda.is_available() else 'cpu'


#######################################################
print("Train Dataset Size", len(train_loader.dataset))
print("Test Dataset Size", len(test_loader.dataset))

for batch_idx, (data, target) in enumerate(train_loader):
  print(data.shape)
  print(target.shape)
  break

######################################################

for batch_idx, (data, target) in enumerate(train_loader):
  plt.figure()
  print("*"*10)
  f, axarr = plt.subplots(1,5) 
  for i in range(5):
    image = torch.transpose(data[i], 0, 2)
    image = torch.transpose(image, 0, 1)
    image = image.cpu().detach().numpy()
    label = target[i]
    axarr[i].imshow(image)
    print(classes[label])
  break
