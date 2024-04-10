import torch
from model1 import model
from dataset import test_dataloader
import torch.nn as nn

def test_model(model, test_dataloader, criterion):
    model.eval()  # Chuyển sang chế độ đánh giá
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  
        for data in test_dataloader:
            inputs, labels = data
            labels = labels.long()

            outputs = model(inputs)
            
            # Tính loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # Dự đoán nhãn
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Tính toán các chỉ số đánh giá
    avg_loss = test_loss / len(test_dataloader)
    accuracy = correct / total
    
    print('Test Loss: {:.4f}'.format(avg_loss))
    print('Test Accuracy: {:.2f}%'.format(100 * accuracy))

# Load saved model
model.load_state_dict(torch.load('resnet50_binary_classification.pth'))
model.eval()  # Đảm bảo rằng model ở chế độ đánh giá

# Khởi tạo DataLoader cho tập dữ liệu kiểm tra (test_dataset)


# Định nghĩa hàm loss
criterion = nn.CrossEntropyLoss()

# Test model
test_model(model, test_dataloader, criterion)
