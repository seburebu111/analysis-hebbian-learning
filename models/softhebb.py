import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# 1. MODEL DEFINITION
# ---------------------------------------------------------
class SoftHebbMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2):
        super().__init__()
        # For Hebbian learning, bias is often omitted or handled separately. 
        # We omit it here for simplicity and to match the pure weight update rule.
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Softmax is applied to the final output for the supervised error calculation
        x = F.softmax(self.fc3(x), dim=1) 
        
        return x

# ---------------------------------------------------------
# 2. TRAINING FUNCTION
# ---------------------------------------------------------
def train_softhebb_model(model, train_dataloader, test_dataloader, criterion, optimizer_name='SGD', epochs=50, eta=0.01, t_invert=12.0):
    """
    Trains a model using the SoftHebb unsupervised learning algorithm for hidden layers,
    and a standard supervised delta rule for the final classification layer.
    """
    train_loss_history, train_acc_history = [], []
    test_loss_history, test_acc_history = [], []
    
    # define function to register the pre-activations
    pre_activation = {}
    def get_pre_activation(name):
        def hook(model, input, output):
            pre_activation[name] = output.detach()
        return hook
        
    linear_layers = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            layer.register_forward_hook(get_pre_activation(name))
            linear_layers.append((name, layer))

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_dataloader, 0):
            inputs, target = data
            inputs = torch.flatten(inputs, 1)
            target_onehot = F.one_hot(target, num_classes=model.fc3.out_features).float()
            
            # Forward pass
            outputs = model(inputs)
            
            # Previous layer's post-activation (starting with inputs)
            current_input = inputs
            batch_size = inputs.shape[0]
            
            for l_idx, (name, layer) in enumerate(linear_layers):
                pre_x = pre_activation.pop(name)
                
                if l_idx < len(linear_layers) - 1:
                    # --- SoftHebb Unsupervised Update for Hidden Layers ---
                    # wta = softmax(t_invert * pre_x)
                    wta = F.softmax(t_invert * pre_x, dim=1)
                    
                    # yx = wta^T @ current_input
                    yx = torch.matmul(wta.t(), current_input)
                    
                    # yu = sum(wta * pre_x, dim=0)
                    yu = torch.sum(wta * pre_x, dim=0).unsqueeze(1)
                    
                    # delta_weight = yx - yu * W
                    delta_w = yx - yu * layer.weight.detach()

                    # The next layer's input will be the post-activation
                    current_input = F.relu(pre_x)
                    
                    del wta, yx, yu
                else:
                    # --- Supervised Delta Rule for Final Layer ---
                    # error = outputs - target_onehot (outputs already has softmax applied in forward pass)
                    error = outputs - target_onehot
                    
                    # delta_w = - error^T @ current_input
                    delta_w = -torch.matmul(error.t(), current_input)

                # apply the weight change immediately
                if optimizer_name == 'SGD':
                    with torch.no_grad():
                        layer.weight += eta * delta_w / batch_size
                        
                del delta_w, pre_x
                        
            # keep track of the loss
            loss = criterion(outputs, target)
            epoch_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
        train_loss_history.append(epoch_loss / len(train_dataloader))
        train_acc_history.append(100 * correct / total)
        
        # Test the model
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data in test_dataloader:
                inputs, target = data
                inputs = torch.flatten(inputs, 1)
                target_onehot = F.one_hot(target, num_classes=model.fc3.out_features).float()

                outputs = model(inputs)
                loss = criterion(outputs, target)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()

        test_loss_history.append(test_loss / len(test_dataloader))
        test_acc_history.append(100 * test_correct / test_total)
            
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss_history[-1]:.4f} | Train Acc: {train_acc_history[-1]:.2f}% | "
                  f"Test Loss: {test_loss_history[-1]:.4f} | Test Acc: {test_acc_history[-1]:.2f}%")
            
    return train_loss_history, train_acc_history, test_loss_history, test_acc_history
