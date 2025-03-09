# AlexNet Implementation in PyTorch & Keras

## Overview
This repository contains an implementation of the AlexNet deep convolutional neural network in both PyTorch and Keras. The model is based on the paper *ImageNet Classification with Deep Convolutional Neural Networks* by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton.

## PyTorch Implementation

### Dependencies
Ensure you have the following dependencies installed:
```bash
pip install torch torchvision
```

### Data Preprocessing
The CIFAR-10 dataset is used for training and evaluation. The dataset is preprocessed using transformations including resizing, normalization, and conversion to tensors.
```python
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

transform = transforms.Compose([
    transforms.Resize(227),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
indices = list(range(1000))
train_set = Subset(train_set, indices)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_set = Subset(test_set, indices)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)
```

### Model Architecture
The AlexNet model is implemented using PyTorch.
```python
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(4096, 1000)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu7(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
```

### Training
The model is compiled with a cross-entropy loss function and optimized using Stochastic Gradient Descent (SGD).
```python
model = AlexNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
```

## Keras Implementation

### Dependencies
Ensure you have TensorFlow installed:
```bash
pip install tensorflow
```

### Data Preprocessing
The CIFAR-10 dataset is loaded and resized before feeding into the model.
```python
import tensorflow as tf
from tensorflow import keras

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train, y_train, X_test, y_test = X_train[:10000], y_train[:10000], X_test[:1000], y_test[:1000]

def resize_images(images):
    return tf.image.resize(images, [128, 128])

X_train = resize_images(X_train)
X_test = resize_images(X_test)
```

### Model Architecture
The AlexNet model is implemented in Keras using `Sequential` API.
```python
model = keras.models.Sequential([
    keras.layers.Conv2D(96, (11,11), strides=(4,4), activation='relu', input_shape=(128,128,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D((3,3), strides=(2,2)),
    keras.layers.Conv2D(256, (5,5), strides=(1,1), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D((3,3), strides=(2,2)),
    keras.layers.Conv2D(384, (3,3), strides=(1,1), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(384, (1,1), strides=(1,1), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(256, (1,1), strides=(1,1), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D((3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])
```

### Training
```python
alexnet_optimizer = tf.optimizers.SGD(learning_rate=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=alexnet_optimizer, metrics=['accuracy'])
model.summary()
```

## License
This implementation is provided for educational purposes. Feel free to use and modify as needed.


