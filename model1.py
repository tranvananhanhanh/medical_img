import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import train_dataloader 



model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50',weights='ResNet50_Weights.DEFAULT' )
model.eval()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 1
print_every = 1  # In ra mỗi 100 mini-batches

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.long()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
         # Print loss every 'print_every' mini-batches
        if (i + 1) % print_every == 0:
            print('[Epoch %d, Batch %5d] Loss: %.3f' %
                  (epoch + 1, i + 1, loss.item()))
    # Calculate average loss for the epoch
    average_loss = running_loss / len(train_dataloader)
    print('[Epoch %d] Loss: %.3f' % (epoch + 1, average_loss))


print('Finished Training')

# Save the trained model
torch.save(model.state_dict(), 'resnet50_binary_classification.pth')