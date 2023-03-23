import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import ConcatDataset, DataLoader

# Define las transformaciones para el preprocesamiento de las imágenes
original_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

flipped_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Carga los datos desde la estructura de directorios
train_path = 'E:\\duraton\\huesos\\vision_artificial\\craneos_ia\\train'
train_original = datasets.ImageFolder(root=train_path, transform=original_transforms)
train_flipped = datasets.ImageFolder(root=train_path, transform=flipped_transforms)
train_data = ConcatDataset([train_original, train_flipped])

val_path = 'E:\\duraton\\huesos\\vision_artificial\\craneos_ia\\validation'
val_original = datasets.ImageFolder(root=val_path, transform=original_transforms)
val_flipped = datasets.ImageFolder(root=val_path, transform=flipped_transforms)
val_data = ConcatDataset([val_original, val_flipped])

# Crea los iteradores de datos para el entrenamiento, validación y prueba
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)


# %%
import torch.nn as nn
import torchvision.models as models

# Cargar la ResNet50 pre-entrenada
resnet = models.resnet50(weights='DEFAULT')

# Congelar los parámetros de la ResNet
for param in resnet.parameters():
    param.requires_grad = False

# Reemplazar la última capa de clasificación para adaptarse a nuestro problema de clasificación binaria
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 2)  # 2 clases en este caso

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet.fc.parameters(), lr=0.001, momentum=0.9)
num_epochs = 100
total_step = len(train_loader)
# %%
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Definir el modelo
model = resnet
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Configurar el registro de TensorBoard
writer = SummaryWriter()
# Entrenar el modelo
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
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
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100.0 * correct / total
    
    # Registrar los resultados en TensorBoard
    writer.add_scalar('Loss/train', loss.item(), epoch)
    writer.add_scalar('Accuracy/validation', accuracy, epoch)

    print(f'Epoch {epoch+1}, loss: {loss.item():.3f}, accuracy: {accuracy:.3f}')

# Cerrar el registro de TensorBoard
writer.close()
            
# %%

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.utils.data as data
# from torch.utils.tensorboard import SummaryWriter

# # Definir el modelo
# model = resnet
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001)

# # Configurar el registro de TensorBoard
# writer = SummaryWriter()


# # Entrenar el modelo y registrar los resultados en TensorBoard
# for epoch in range(10):
#     # Entrenamiento
#     model.train()
#     for i, (inputs, targets) in enumerate(train_loader):
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#     # Validación
#     model.eval()
#     total = 0
#     correct = 0
#     with torch.no_grad():
#         for inputs, targets in val_loader:
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs.data, 1)
#             total += targets.size(0)
#             correct += (predicted == targets).sum().item()
#     accuracy = 100.0 * correct / total
    
#     # Registrar los resultados en TensorBoard
#     writer.add_scalar('Loss/train', loss.item(), epoch)
#     writer.add_scalar('Accuracy/validation', accuracy, epoch)

#     print(f'Epoch {epoch+1}, loss: {loss.item():.3f}, accuracy: {accuracy:.3f}')

# # Cerrar el registro de TensorBoard
# writer.close()

