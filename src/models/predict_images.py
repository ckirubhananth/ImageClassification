import os
import torch
import torchvision.transforms as transforms
from PIL import Image

# Import available models dynamically
from src.models.cnn_model import CNN
from src.models.resnet_model import ResNet50Classifier
from src.models.efficientNet_model import EfficientNetB2Classifier

def get_model(selected_model, model_path, num_classes):
    """ Load the selected model dynamically based on user choice. """
    if selected_model == "CNN":
        model = CNN(num_classes=num_classes)
    elif selected_model == "ResNet-50":
        model = ResNet50Classifier(num_classes=num_classes)
    elif selected_model == "EfficientNet-B2":
        model = EfficientNetB2Classifier(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model selection: {selected_model}")

    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

def process_image(image_path, selected_model):
    """ Apply preprocessing dynamically based on selected model. """
    image_sizes = {
        "CNN": (64, 64),
        "ResNet-50": (224, 224),
        "EfficientNet-B2": (224, 224)
    }
    
    transform = transforms.Compose([
        transforms.Resize(image_sizes.get(selected_model, (224, 224))),  # Adjust size dynamically
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard normalization
    ])

    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def classify_images(selected_model, model_path, image_folder, output_folder, progress_bar, ui):
    """ Predict and organize images using the selected model. """
    training_folder = os.path.dirname(model_path).replace("_model", "_training")
    class_labels = os.listdir(training_folder) if os.path.exists(training_folder) else []

    # Load the correct model dynamically
    model = get_model(selected_model, model_path, num_classes=len(class_labels))

    images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    progress_bar["maximum"] = len(images)

    for i, image_name in enumerate(images):
        image_path = os.path.join(image_folder, image_name)
        image_tensor = process_image(image_path, selected_model)
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        predicted_label = class_labels[predicted_class.item()] if predicted_class.item() < len(class_labels) else "Unknown"

        # Create subfolder for predicted class
        person_folder = os.path.join(output_folder, predicted_label)
        os.makedirs(person_folder, exist_ok=True)

        # Move image to correct folder
        new_image_path = os.path.join(person_folder, image_name)
        os.rename(image_path, new_image_path)
        print(f"Classified {image_name} as {predicted_label} → Saved to {person_folder}")

        # Update progress bar in UI
        progress_bar["value"] = i + 1
        ui.update()
