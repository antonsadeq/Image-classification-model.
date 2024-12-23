import torch
import torch.nn as nn
from Data_preprocessing import train_dataset, test_dataset
from training_function import training_loop
from architecture import MyCNN
from torchvision import transforms

torch.random.manual_seed(1234)
num_classes = 20

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30)
])
train_dataset.transform = transform
model = MyCNN(
    input_channels=1,
    hidden_channels=[128, 512, 512, 1024],
    use_batch_normalization=True,
    num_classes=num_classes,
    kernel_size=[3, 3, 3, 3],
    activation_function=nn.ReLU(),
    use_dropout=True
)

# Parameters for training
num_epochs = 30
batch_size = 32
learning_rate = 0.001
show_progress = True

# Train the model
print(f"Training Model 1")
train_losses, eval_losses, train_accuracies, eval_accuracies, epoch_trained = training_loop(
    network=model,
    train_data=train_dataset,
    eval_data=test_dataset,
    num_epochs=num_epochs,
    batch_size=batch_size,
    learning_rate=learning_rate,
    show_progress=show_progress
)
print(f"Model 1 trained for {epoch_trained} epochs")
print(f"Final Train Loss: {train_losses[-1]:.4f}, Final Train Accuracy: {train_accuracies[-1]:.4f}")
print(f"Final Eval Loss: {eval_losses[-1]:.4f}, Final Eval Accuracy: {eval_accuracies[-1]:.4f}")

# Save the model with the lowest evaluation loss
best_epoch = eval_losses.index(min(eval_losses))
torch.save(model.state_dict(), "model.pth")
print(f"Model saved at epoch {best_epoch + 1} with Eval Loss: {eval_losses[best_epoch]:.4f}, Eval Accuracy: {eval_accuracies[best_epoch]:.4f}")
