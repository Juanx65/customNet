############################################
# falta aplicar algun algoritmo que recorra la imagen para poder hacer detecciones en partes de esta y asegurarse
# de asi poder entrenar de manera correcta la red
# en este momento tenemos error de que las labels de entrada estan en distinta cantidad que las detecciones
# hechas sobre la imagen, lo que impide el entrenamiento
############################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset
from PIL import Image
import os
from torch.utils.data import DataLoader
from torch import optim
import torchvision.transforms as transforms

device = "cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# definir la arquitectura de la red neuronal
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 157 * 157, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)  # Salida de 5: 1 para la clase y 4 para las coordenadas del bounding box

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 157 * 157)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum") # Para la regresión del bounding box
        self.bce_loss = nn.BCEWithLogitsLoss() # Para la clasificación

    def forward(self, pred, target):
        # separar las predicciones en la clasificación y las coordenadas del bounding box
        pred_class = pred[:, 0]
        pred_box = pred[:, 1:]

        # hacer lo mismo para los targets
        target_class = target[:, 0]
        target_box = target[:, 1:]

        # calcular la pérdida
        classification_loss = self.bce_loss(pred_class, target_class)
        regression_loss = self.mse_loss(pred_box, target_box)

        # la pérdida total es la suma de la pérdida de clasificación y la pérdida de regresión
        loss = classification_loss + regression_loss
        return loss

def load_data_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            data_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return data_config

class YOLODataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.img_files = os.listdir(folder_path)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.folder_path, self.img_files[index])
        # Carga la imagen
        img = Image.open(img_path).convert('RGB')
        
        # Normaliza la imagen
        img = transforms.ToTensor()(img)
        
        # Carga las etiquetas correspondientes
        label_path = img_path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt')
        labels = []
        if os.path.getsize(label_path) > 0:
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    labels.append(torch.tensor([float(i) for i in line.split()]))
        else:  # Si el archivo de etiquetas existe pero está vacío
            #labels.append(torch.zeros(5))  # Etiqueta ficticia
            print("vacio")

        return img, labels  # Devuelve una lista de tensores

def collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return images, labels
    
def train_model(yaml_file_path):
    # Carga la configuración de datos
    data_config = load_data_config(yaml_file_path)
    prepath = yaml_file_path.split("data.yaml")[0]
    
    # Crea los datasets
    train_dataset = YOLODataset(prepath+data_config['train'])
    val_dataset = YOLODataset(prepath+data_config['val'])
    test_dataset = YOLODataset(prepath+data_config['test'])
    # En la creación de tu DataLoader:
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn)

    # Crea el modelo
    model = BaseModel()
    
    # Crea la función de pérdida
    criterion = BaseLoss()

    # Crea el optimizador
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Entrena el modelo
    for epoch in range(10):  # Itera sobre los datos varias veces
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # Obtiene las entradas
            inputs_list, labels_list = data

            # Convierte las imágenes en un tensor
            inputs = torch.stack(inputs_list).to(device)
            labels_list = [label.to(device) for sublist in labels_list for label in sublist]
            labels = torch.stack(labels_list)

            # Pone a cero los gradientes del parámetro
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            print("output: ", outputs)
            print("labels: ", labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Imprime las estadísticas
            running_loss += loss.item()
            if i % 2000 == 1999:    # Imprime cada 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            print('Finished Training')



    # Ahora, vamos a probar el modelo en el conjunto de validación
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the validation images: %d %%' % (100 * correct / total))


current_directory = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_directory,"datasets/data.yaml")
train_model(data_path)