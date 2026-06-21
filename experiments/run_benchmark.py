import argparse
import time
import os
import sys
import math
import resource
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath('..'))
from models.baseline_bp import BaselineMLP, train_bp_model
from models.pepita import PepitaMLP, train_pepita_model
from models.softhebb import SoftHebbMLP, train_softhebb_model

def main():
    parser = argparse.ArgumentParser(description="Benchmark Model Performance")
    parser.add_argument('--model', type=str, required=True, choices=['BP', 'PEPITA', 'SoftHebb'])
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    print(f"Loading data and initializing {args.model} for {args.epochs} epochs...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    data_dir = '../data'
    
    # Suppress dataset download printings
    train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    
    batch_size = 128
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = 784
    hidden_dim = 512
    output_dim = 10
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    start_time = time.time()

    if args.model == 'BP':
        model = BaselineMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        train_bp_model(model, train_dataloader, test_dataloader, criterion, optimizer, epochs=args.epochs)
    
    elif args.model == 'PEPITA':
        model = PepitaMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        sd = math.sqrt(1 / input_dim)
        B = (torch.rand(input_dim, output_dim) * 2 * sd - sd) * 0.05
        train_pepita_model(model, train_dataloader, test_dataloader, B, criterion, optimizer_name='SGD', epochs=args.epochs, eta=0.01)
    
    elif args.model == 'SoftHebb':
        model = SoftHebbMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        train_softhebb_model(model, train_dataloader, test_dataloader, criterion, optimizer_name='SGD', epochs=args.epochs, eta=0.03)

    end_time = time.time()

    avg_epoch_time = (end_time - start_time) / args.epochs
    
    if torch.cuda.is_available():
        memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        # ru_maxrss is in bytes on mac and kilobytes on linux
        ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == 'darwin':
            memory_mb = ru_maxrss / (1024 * 1024)
        else:
            memory_mb = ru_maxrss / 1024

    print(f"\n=== Benchmark Results for {args.model} ===")
    print(f"Total Time ({args.epochs} epochs): {end_time - start_time:.2f} s")
    print(f"Average Time per epoch: {avg_epoch_time:.2f} s")
    print(f"Peak Process Memory: {memory_mb:.2f} MB")

if __name__ == '__main__':
    main()
