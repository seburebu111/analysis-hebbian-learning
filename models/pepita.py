import torch
import torch.nn as nn
import torch.nn.functional as F

class PepitaMLP(nn.Module):
    """
    A Multilayer Perceptron designed for the PEPITA algorithm.
    Biases are omitted because the core PEPITA learning rule applies to weights.
    We pass `do_masks` to ensure the exact same dropout mask is used in both the 
    standard and modulated forward passes.
    """
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2):
        super(PepitaMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x, do_masks=None):
        x = F.relu(self.fc1(x))
        if do_masks is not None and len(do_masks) > 0:
            x = x * do_masks[0]
            
        x = F.relu(self.fc2(x))
        if do_masks is not None and len(do_masks) > 1:
            x = x * do_masks[1]
            
        x = self.fc3(x)
        # Apply softmax because PEPITA computes error as (output_probs - target_one_hot)
        x = F.softmax(x, dim=1) 
        if do_masks is not None and len(do_masks) > 2:
            x = x * do_masks[2]
            
        return x


def train_pepita_model(model, train_dataloader, test_dataloader, B, criterion=None, optimizer_name='SGD', epochs=50, eta=0.01, keep_rate=1.0):
    """
    Trains a model using the PEPITA algorithm.
    
    Args:
        model (nn.Module): The neural network to train.
        dataloader (DataLoader): PyTorch DataLoader for the dataset.
        B (torch.Tensor): Random projection matrix used to map errors back to input space.
        criterion (nn.Module, optional): Loss function (e.g., CrossEntropyLoss) used only for logging.
        optimizer_name (str): 'SGD' or 'mom' (momentum).
        epochs (int): Number of training epochs.
        eta (float): Learning rate.
        keep_rate (float): Dropout keep probability.
    """
    train_loss_history, train_acc_history = [], []
    test_loss_history, test_acc_history = [], []
    
    # 1. Register forward hooks to capture activations of each linear layer
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
        
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            layer.register_forward_hook(get_activation(name))

    # 2. Setup momentum optimizer variables if requested
    if optimizer_name == 'mom':
        gamma = 0.9
        v_w_all = []
        for w in model.parameters():
            if len(w.shape) > 1:
                with torch.no_grad():
                    v_w_all.append(torch.zeros_like(w))

    # 3. Dummy forward pass to populate the `activation` dictionary initially
    dummy_inputs, _ = next(iter(train_dataloader))
    model(dummy_inputs)
    layers_key = [key for key in activation.keys() if 'fc' in key]
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, target) in enumerate(train_dataloader):
            batch_size = inputs.size(0)
            
            # Convert targets to one-hot for PEPITA error calculation
            # Assuming classification, we extract the number of classes from B's shape
            num_classes = B.shape[1]
            target_onehot = F.one_hot(target, num_classes=num_classes).float()
            
            # --- PASS 1: Standard Forward Pass ---
            # Set up identical dropout masks for both passes if keep_rate < 1.0
            do_masks = []
            if keep_rate < 1.0:
                # We need a mask for each hidden layer activation
                for _ in range(len(layers_key) - 1):
                    # We just use the batch size and the hidden dims (assuming uniform hidden dim here, 
                    # but dynamically getting it from model is safer)
                    pass 
                # For simplicity in this clean implementation, we'll only apply dropout if explicitly implemented.
                # Here we assume keep_rate=1.0 for the baseline 2D datasets unless complex dropout is needed.
            
            outputs = model(inputs, do_masks=None)
            
            # Gather activations (applying ReLU since we hooked the Linear layers before activation)
            layers_act = []
            for key in layers_key:
                if key != layers_key[-1]: # Hidden layers
                    layers_act.append(F.relu(activation[key]))
                else: # Output layer (Softmax is applied in forward, but the hook grabs pre-softmax. We'll use outputs instead)
                    layers_act.append(outputs)
            
            # Compute PEPITA standard error
            error = outputs - target_onehot
            
            # --- PASS 2: Modulated Forward Pass ---
            # Modulate inputs using the random projection matrix B
            error_input = error @ B.T
            mod_inputs = inputs + error_input
            
            mod_outputs = model(mod_inputs, do_masks=None)
            
            mod_layers_act = []
            for key in layers_key:
                if key != layers_key[-1]:
                    mod_layers_act.append(F.relu(activation[key]))
                else:
                    mod_layers_act.append(mod_outputs)
                    
            mod_error = mod_outputs - target_onehot
            
            # --- COMPUTE WEIGHT UPDATES (PEPITA Rule) ---
            delta_w_all = []
            for l in range(len(layers_key)):
                if l == len(layers_key) - 1:
                    # Last layer
                    if len(layers_act) > 1:
                        delta_w = -mod_error.T @ mod_layers_act[-2]
                    else:
                        delta_w = -mod_error.T @ mod_inputs
                
                elif l == 0:
                    # First layer
                    delta_w = -(layers_act[l] - mod_layers_act[l]).T @ mod_inputs
                
                else:
                    # Intermediate hidden layers
                    delta_w = -(layers_act[l] - mod_layers_act[l]).T @ mod_layers_act[l-1]
                
                delta_w_all.append(delta_w)

            # --- APPLY WEIGHT UPDATES ---
            if optimizer_name == 'SGD':
                l_idx = 0
                for w in model.parameters():
                    if len(w.shape) > 1: # We only update weights, not biases
                        with torch.no_grad():
                            w += eta * delta_w_all[l_idx] / batch_size
                        l_idx += 1
                        
            elif optimizer_name == 'mom':
                l_idx = 0
                for w in model.parameters():
                    if len(w.shape) > 1:
                        with torch.no_grad():
                            v_w_all[l_idx] = gamma * v_w_all[l_idx] + eta * delta_w_all[l_idx] / batch_size
                            w += v_w_all[l_idx]
                        l_idx += 1

            # Log loss
            if criterion:
                loss = criterion(outputs, target)
                epoch_loss += loss.item()
            else:
                # Fallback to cross-entropy if no criterion provided
                epoch_loss += F.cross_entropy(outputs, target).item()
                
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
                
        train_loss_history.append(epoch_loss / len(train_dataloader))
        train_acc_history.append(100 * correct / total)

        # Test the model
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
