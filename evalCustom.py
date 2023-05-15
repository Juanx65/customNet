import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

from models.customNet import ConvNet

# Inicialización del modelo
model = ConvNet()

# Definición del optimizador y la función de pérdida
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

## Carga de los datos
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

""" ## Entrenamiento del modelo
num_epochs = 10
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = loss_function(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

## Guardar el modelo
torch.save(model, 'weights/best.pth') """

# Prueba del modelo
model = torch.load('weights/best.pth')
model.eval()
tiempos = 0
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        start = time.time()
        outputs = model(images)
        torch.cuda.synchronize() 
        end = time.time()
        tiempos += (end-start)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Tiempo avg : ', tiempos / total, ' segundos')
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
