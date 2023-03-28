import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset, DataLoader

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from PIL import Image

# %% MODEL PATH
model_path = 'E:\\duraton\\huesos\\vision_artificial\\craneos_ia\\model0_resnet18.pt'

# %% Get MEAN and STD
# Define las transformaciones para el preprocesamiento de las imágenes

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
])

# Carga los datos desde la estructura de directorios
train_path = 'E:\\duraton\\huesos\\vision_artificial\\craneos_ia\\train_0'
norm_data = datasets.ImageFolder(root=train_path, transform=transform)
norm_loader = DataLoader(norm_data, batch_size=len(norm_data), shuffle=True)

data = next(iter(norm_loader))
mean_list = [data[0].mean().item() for i in range(3)] # Mean de los 3 canales
std_list = [data[0].std().item() for i in range(3)] # STD de los 3 canales
# %% APPLY TRANSFORMATIONS
no_flipped_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.RandomHorizontalFlip(p=0),
        transforms.ToTensor(),
        transforms.Normalize(mean_list, std_list)
])

flipped_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean_list, std_list)
])

# Carga los datos desde la estructura de directorios
train_path = 'E:\\duraton\\huesos\\vision_artificial\\craneos_ia\\train_0'
train_original = datasets.ImageFolder(root=train_path, transform=no_flipped_transforms)
train_flipped = datasets.ImageFolder(root=train_path, transform=flipped_transforms)
train_data = ConcatDataset([train_original, train_flipped])

val_path = 'E:\\duraton\\huesos\\vision_artificial\\craneos_ia\\validation_0'
val_original = datasets.ImageFolder(root=val_path, transform=no_flipped_transforms)
val_flipped = datasets.ImageFolder(root=val_path, transform=flipped_transforms)
val_data = ConcatDataset([val_original, val_flipped])

# Crea los iteradores de datos para el entrenamiento, validación y prueba
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

# %% DEVICE CUDA
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# %% DEFINE MODEL

# Cargar la ResNet50 pre-entrenada
resnet = models.resnet18(weights='DEFAULT')

# Congelar los parámetros de la ResNet
for param in resnet.parameters():
    param.requires_grad = False

# Reemplazar la última capa de clasificación para adaptarse a nuestro problema de clasificación binaria
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 2)  # 2 clases en este caso
model = resnet

# %% TRAIN
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#, momentum=0.9
num_epochs = 50
total_step = len(train_loader)
# Configurar el registro de TensorBoard
writer = SummaryWriter()
# Entrenar el modelo
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        
        # # Usamos cuda
        # images = images.to(device)
        # labels = labels.to(device)
        # model = model.to(device)
        
        # Forward pass
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        
        # Backward y optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Imprimir estadísticas
        if (i+1) % 2 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
    # Validación
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            # # Cuda
            # inputs = inputs.to(device)
            # targets = targets.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100.0 * correct / total
    
    # Registrar los resultados en TensorBoard
    writer.add_scalar('Loss/train', loss.item(), epoch)
    writer.add_scalar('Accuracy/validation', accuracy, epoch)
    model.train()

    print(f'Epoch {epoch+1}, loss: {loss.item():.3f}, accuracy: {accuracy:.3f}')

# Cerrar el registro de TensorBoard
writer.close()
            
# %% SAVE MODEL
# Guardar el modelo entrenado
torch.save(model.state_dict(), model_path)

# %% LOAD MODEL
# # Cargar el modeloEFAULT')
resnet = models.resnet18(weights='DEFAULT')

# Congelar los parámetros de la ResNet
for param in resnet.parameters():
    param.requires_grad = False

# Reemplazar la última capa de clasificación para adaptarse a nuestro problema de clasificación binaria
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 2)  # 2 clases en este caso
model = resnet

model.load_state_dict(torch.load(model_path))

# %% ENTENDER ENTRENAMIENTO

layer_list = [(name, layer) for name, layer in model.named_children()]
for name, layer in model.named_children():
    print(name)

# def get_intermediate_layer(model, layer_name):
#     intermediate_output = None
#     def hook(model, input, output):
#         nonlocal intermediate_output
#         intermediate_output = output
#     layer = model._modules.get(layer_name)
#     layer.register_forward_hook(hook)
#     return intermediate_output

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook









# Obtener una imagen de ejemplo y transformarla para que sea compatible con el modelo
img_path = 'E:/duraton/huesos/vision_artificial/craneos_ia/train_0/aguila_imperial/aguila_imperial_train (75).jpg'
example_image = Image.open(img_path)
original_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.RandomHorizontalFlip(p=0),
        transforms.ToTensor(),
        transforms.Normalize(mean_list, std_list)
        
])
input_tensor = original_transforms(example_image)
example_loader = DataLoader(input_tensor, batch_size=32, shuffle=True)

model.conv1.register_forward_hook(get_activation('conv1'))
output = model(example_loader.dataset.unsqueeze(0))
print(output.shape)
# print(activation['fc2'])  


image = input_tensor.squeeze()
np_image = image.numpy()
to_pil = transforms.ToPILImage()
pil_img = to_pil(input_tensor) 

plt.imshow(pil_img)

# %%
# 


# Definir una función para imprimir los tensores de las capas
def print_tensor(module, input, output):
    print(f"{module.__class__.__name__} output shape: {output.shape}")

# Registrar la función de impresión para cada capa del modelo
for name, layer in model.named_modules():
    layer.register_forward_hook(print_tensor)

# Pasar una imagen a través del modelo
input_image = torch.rand(1, 3, 224, 224)  # una imagen de tamaño 224x224 con 3 canales
output = model(input_tensor.unsqueeze(0))

# Eliminar los ganchos para liberar memoria
for name, layer in model.named_modules():
    layer.register_forward_hook(None)
    
# %%
def get_feature_maps(model, x):
    """
    Obtiene el tensor de salida de cada capa convolucional en el modelo dado.
    """
    # Lista para almacenar los tensores de salida
    features = []
    
    # Iterar sobre todas las capas del modelo
    for name, module in model._modules.items():
        try:
            # Ejecutar el módulo sobre el tensor de entrada
            x = module(x)
            # Si la salida es un tensor, agregarlo a la lista
            if isinstance(x, torch.Tensor):
                features.append(x)
                print('ok')
        except:
            # Si no se puede ejecutar el módulo, mostrar un mensaje de error
            print(f"Error al obtener características de la capa {name}")
    
    return features


# Establecer el modelo en modo de evaluación
model.eval()

# Crear un tensor de entrada de ejemplo
x = original_transforms(example_image).unsqueeze(0)

# Obtener los tensores de salida de cada capa convolucional
feature_maps = get_feature_maps(model, x)

# Imprimir la forma de cada tensor de salida
for i, f in enumerate(feature_maps):
    print(f"Capa {i+1}: {f.shape}")

#%%
layer_outputs = []

def layer_hook(module, input, output):
    layer_outputs.append(output)

# Registra los hooks en cada capa de la red
for layer in model.modules():
    layer.register_forward_hook(layer_hook)


img_tensor = input_tensor.unsqueeze(0)

# # Ejecuta la imagen a través de la red
with torch.no_grad():
    model(img_tensor)

# Guarda los tensores de salida de cada capa en variables
# layer1_tensor = layer_outputs[0]
# layer2_tensor = layer_outputs[1]
# layer3_tensor = layer_outputs[2]

# # Visualiza los tensores
# plt.imshow(layer1_tensor[0, 0, :, :])
# plt.show()

# plt.imshow(layer2_tensor[0, 0, :, :])
# plt.show()

# plt.imshow(layer3_tensor[0, 0, :, :])
# plt.show()




