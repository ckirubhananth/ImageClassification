import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim

# Import available models
from src.models.cnn_model import CNN
from src.models.resnet_model import ResNet50Classifier
from src.models.efficientNet_model import EfficientNetB2Classifier

def get_model(selected_model, num_classes):
    """ Initialize and return the chosen model with correct parameters. """
    if selected_model == "CNN":
        return CNN(num_classes=num_classes), (64, 64)  # Return expected input size
    elif selected_model == "ResNet-50":
        return ResNet50Classifier(num_classes=num_classes), (224, 224)
    elif selected_model == "EfficientNet-B2":
        return EfficientNetB2Classifier(num_classes=num_classes), (224, 224)
    else:
        raise ValueError(f"Unknown model selection: {selected_model}")

def train_model(selected_model, training_data_folder, model_save_path, progress_bar, ui, epochs=20, learning_rate=0.001):
    """ Trains a selected model on user-selected images. """

    # Get model & dynamically determine expected input size
    model, input_size = get_model(selected_model, num_classes=len(datasets.ImageFolder(training_data_folder).classes))

    # Define preprocessing dynamically based on selected model
    transform = transforms.Compose([
        transforms.Resize(input_size),  # Dynamically adjust input size
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard normalization
    ])
    
    # Load dataset
    dataset = datasets.ImageFolder(root=training_data_folder, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    # **Handle model-specific freezing logic**
    if selected_model == "ResNet-50":
        for param in model.parameters():
            param.requires_grad = False  # Freeze all layers

        for param in model.model.fc.parameters():
            param.requires_grad = True  # Fine-tune only final layer

    elif selected_model == "EfficientNet-B2":
        for param in model.parameters():
            param.requires_grad = False  # Freeze all layers

        for param in model.model.classifier.parameters():  # EfficientNet uses `classifier`
            param.requires_grad = True  # Fine-tune classifier

    # Define optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=0.0001)

    # Training loop
    progress_bar["maximum"] = epochs

    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for images, labels in dataloader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        scheduler.step(running_loss)
        accuracy = (correct_predictions / total_samples) * 100

        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss:.4f} - Accuracy: {accuracy:.2f}% - LR: {scheduler.get_last_lr()[0]:.6f}")

        progress_bar["value"] = epoch + 1
        ui.status_label.config(text=f"Epoch {epoch+1}: Loss {running_loss:.4f}, Acc {accuracy:.2f}%", foreground="green")
        ui.update()

    # Save trained model
    os.makedirs(model_save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_save_path, f"{selected_model}_trained_model.pth"))
    print(f"Model saved successfully! ({selected_model})")
