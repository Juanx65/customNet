import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch.profiler import profile, record_function, ProfilerActivity

from models.customNet import ConvNet

BATCH_SIZE = 2048

## Inicialización del modelo
model = ConvNet()

## Definición del optimizador y la función de pérdida
#optimizer = optim.Adam(model.parameters(), lr=0.001)
#loss_function = nn.CrossEntropyLoss()

## Carga de los datos
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

#train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

#train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
device = torch.device('cuda:0')
model = torch.load('weights/best.pth')
model.to(device)
model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True,
                 record_shapes=True,
                 with_stack=True,
                 on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/log_vanilla')) as prof:
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            prof.step()  # Need to call this at the end of each step to notify profiler of steps' 
    print(prof.key_averages().table(sort_by="cuda_time_total"))
    #prof.export_chrome_trace("trace_vanilla.json")
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
