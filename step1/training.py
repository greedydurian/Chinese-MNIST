import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from cnn import Net
from dataProcessor.dsLoader import DatasetLoader
import torch.nn as nn
import pandas as pd

datasetLoader = DatasetLoader(csv_file='/Users/oha/Desktop/assessment/chinese_mnist.csv',
                              img_dir='/Users/oha/Desktop/assessment/data/data',
                              batch_size=32)

# calling loaders from datasetLoader
train_loader = datasetLoader.train_loader
val_loader = datasetLoader.val_loader
test_loader = datasetLoader.test_loader

# Initialize the model
model = Net()

#  loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# create lr scheduler
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

epochs = 10
best_acc = 0.0 

# loop through each epoch
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images, labels  

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch+1}/{epochs}.. '
          f'Train loss: {running_loss/len(train_loader):.3f}')

    # validation after each epoch
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print('Validation accuracy: {:.2f} %'.format(val_accuracy))

   
    torch.save(model.state_dict(), '/Users/oha/Desktop/assessment/best_model.pth')

    # decrease lr at the end of epoch
    scheduler.step()

# testing after training is complete
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
   
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Test accuracy: {:.2f} %'.format(100 * correct / total))
