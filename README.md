# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

The objective of this project is to create a CNN that can categorize images of fashion items from the Fashion MNIST dataset. This dataset includes grayscale images of clothing and accessories such as T-shirts, trousers, dresses, and footwear. The task is to accurately predict the correct category for each image while ensuring the model is efficient and robust.

1.Training data: 60,000 images

2.Test data: 10,000 images

3.Classes: 10 fashion categories

The CNN consists of multiple convolutional layers with activation functions, followed by pooling layers, and ends with fully connected layers to output predictions for all 10 categories.

## Neural Network Model

<img width="1183" height="467" alt="425547172-cb131631-9bba-4dc8-a3c8-dd7a9b3c98ba" src="https://github.com/user-attachments/assets/4bb8cd96-bf53-4665-9f77-25e96fec1121" />

## DESIGN STEPS

### STEP 1:
Import the necessary libraries such as NumPy, Matplotlib, and PyTorch.

### STEP 2:
Load and preprocess the dataset:

Resize images to a fixed size (128×128).
Normalize pixel values to a range between 0 and 1.
Convert labels into numerical format if necessary.
### STEP 3:
Define the CNN Architecture, which includes:

Input Layer: Shape (8,128,128)
Convolutional Layer 1: 8 filters, kernel size (16×16), ReLU activation
Max-Pooling Layer 1: Pool size (2×2)
Convolutional Layer 2: 24 filters, kernel size (8×8), ReLU activation
Max-Pooling Layer 2: Pool size (2×2)
Fully Connected (Dense) Layer:
First Dense Layer with 256 neurons
Second Dense Layer with 128 neurons
Output Layer for classification
### STEP 4:
Define the loss function (e.g., Cross-Entropy Loss for classification) and optimizer (e.g., Adam or SGD).

### STEP 5:
Train the model by passing training data through the network, calculating the loss, and updating the weights using backpropagation.

### STEP 6:
Evaluate the trained model on the test dataset using accuracy, confusion matrix, and other performance metrics.

### STEP 7:
Make predictions on new images and analyze the results.
## PROGRAM

### Name: ARUN KUMAR B
### Register Number: 212223230021
```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from torchsummary import summary

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=False, transform=transform, download=True
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f'Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}')


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier().to(device)

print('Name: ARUN KUMAR.B')
print('Register Number: 212223230021')
summary(model, input_size=(1, 28, 28))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Name: ARUN KUMAR.B')
        print('Register Number: 212223230021')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

train_model(model, train_loader)

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print('Name: ARUN KUMAR.B')
    print('Register Number: 212223230021')
    print(f'Test Accuracy: {accuracy:.4f}')

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    print('Name: ARUN KUMAR.B')
    print('Register Number: 212223230021')
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

test_model(model, test_loader)


def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)

    class_names = dataset.classes
    print('Name: ARUN KUMAR.B')
    print('Register Number: 212223230021')
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted.item()]}')
    plt.axis("off")
    plt.show()
    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted.item()]}')

# Example: Predict image at index 80
predict_image(model, image_index=80, dataset=test_dataset)
```

## OUTPUT
<img width="903" height="446" alt="111" src="https://github.com/user-attachments/assets/4ebf5b2a-ef98-4eb0-aea2-67e242c599c0" />

### Training Loss per Epoch
<img width="827" height="266" alt="112" src="https://github.com/user-attachments/assets/612eb3d7-a271-4ef5-864f-974b4cf63738" />

### Confusion Matrix

<img width="1029" height="681" alt="113" src="https://github.com/user-attachments/assets/a7f4cc8b-f3c1-43b2-b930-2857f2ed4685" />



### Classification Report

<img width="661" height="410" alt="114" src="https://github.com/user-attachments/assets/5ec317fb-13e1-44b9-9ece-f4f5fa991a5f" />




### New Sample Data Prediction
<img width="574" height="560" alt="115" src="https://github.com/user-attachments/assets/d6a21106-ed7d-47cc-b443-d13f404df144" />



## RESULT

Thus, a convolutional deep neural network for image classification and to verify the response for new images is to developed successfully.
