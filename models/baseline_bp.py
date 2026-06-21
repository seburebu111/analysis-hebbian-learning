import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # CrossEntropyLoss applies Softmax internally
        return x

def train_bp_model(model, train_dataloader, test_dataloader, criterion, optimizer, epochs=50):
    train_loss_history, train_acc_history = [], []
    test_loss_history, test_acc_history = [], []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_dataloader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss_history.append(epoch_loss / len(train_dataloader))
        train_acc_history.append(100 * correct / total)

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_loss_history.append(test_loss / len(test_dataloader))
        test_acc_history.append(100 * correct / total)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss_history[-1]:.4f} | Train Acc: {train_acc_history[-1]:.2f}% | "
                  f"Test Loss: {test_loss_history[-1]:.4f} | Test Acc: {test_acc_history[-1]:.2f}%")

    return train_loss_history, train_acc_history, test_loss_history, test_acc_history
