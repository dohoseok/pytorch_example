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

model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(512,10)
print(model)

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % 100 == 0 :
      print("Train Epoch : {} | Batch Status : {}/{} | Loss : {:.5f}".format(
          epoch, batch_idx*len(data), len(train_loader.dataset), loss.item()
      ))

def test():
  model.eval()
  test_loss = 0
  correct = 0
  for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    output = model(data)

    test_loss += criterion(output, target).item()

    pred = output.data.max(1, keepdim=True)[1]
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()

  test_loss /= len(test_loader.dataset)

  print("="*30)
  print(f"Average loss : {test_loss:.4f}, Accuracy:{correct}/{len(test_loader.dataset)}"
        f"({100. * correct / len(test_loader.dataset):.1f} %)")
  print("="*30)


start = time.time()
for epoch in range(10):
  epoch_start = time.time()
  train(epoch)
  m, s =divmod(time.time() - epoch_start, 60)
  print(f'Training time : {m:.0f}m {s:.0f}s')

  test()
m, s =divmod(time.time() - start, 60)
print(f'Total time : {m:.0f}m {s:.0f}s')

save_path = "./cifar_resnet.pth"
torch.save(model.state_dict(), save_path)


# Inference Model
model.load_state_dict(torch.load(save_path))

dataiter = iter(test_loader)
images, labels = dataiter.next()

outputs = model(images.to(device))
_, predicted = torch.max(outputs, 1)

# print images
imshow(torchvision.utils.make_grid(images[:10], nrow=5, ncol=2))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(10)))
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(10)))
