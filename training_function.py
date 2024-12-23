import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    return correct / targets.size(0)


def training_loop(
        network: torch.nn.Module,
        train_data: torch.utils.data.Dataset,
        eval_data: torch.utils.data.Dataset,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        show_progress: bool = False) -> tuple[list, list, list, list, int]:
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=batch_size)

    # Set the target device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)

    print(f"Training on device: {device}")

    # Create optimizer
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate)

    # Loss function for classification
    Loss = nn.CrossEntropyLoss()

    # Lists to store losses and accuracies
    train_losses = []
    eval_losses = []
    train_accuracies = []
    eval_accuracies = []
    epoch_trained = 0

    for epoch in range(num_epochs):
        network.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        if show_progress:
            loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
        else:
            loop = train_loader

        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = Loss(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct_train += (outputs.argmax(1) == targets).sum().item()
            total_train += targets.size(0)

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        network.eval()
        eval_loss = 0.0
        correct_eval = 0
        total_eval = 0

        with torch.no_grad():
            for inputs, targets in eval_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = network(inputs)
                loss = Loss(outputs, targets)
                eval_loss += loss.item()
                correct_eval += (outputs.argmax(1) == targets).sum().item()
                total_eval += targets.size(0)

        avg_eval_loss = eval_loss / len(eval_loader)
        eval_losses.append(avg_eval_loss)
        eval_accuracy = correct_eval / total_eval
        eval_accuracies.append(eval_accuracy)

        epoch_trained = epoch + 1

        if show_progress:
            print(
                f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Eval Loss: {avg_eval_loss:.4f}, Eval Acc: {eval_accuracy:.4f}')

    return train_losses, eval_losses, train_accuracies, eval_accuracies, epoch_trained
