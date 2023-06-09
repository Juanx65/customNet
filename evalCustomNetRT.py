import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch.profiler import profile, record_function, ProfilerActivity

from models import engine
import os

BATCH_SIZE = 2048

## Carga de los datos
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

#train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

#train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

## Prueba del modelo
device = torch.device('cuda:0')
current_directory = os.path.dirname(os.path.abspath(__file__))
engine_path = os.path.join(current_directory,'weights/best.engine')
Engine = engine.TRTModule(engine_path, device)
Engine.set_desired(['outputs'])
model = Engine

#with torch.no_grad():
with torch.set_grad_enabled(False):
    correct = 0
    total = 0
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True,
                 record_shapes=True,
                 with_stack=True,
                 on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/log_trt_fp32')) as prof:
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            if( predicted.size() == labels.size() ):
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            prof.step()
    print(prof.key_averages().table(sort_by="cuda_time_total"))
    #prof.export_chrome_trace("trace_RT.json")
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
