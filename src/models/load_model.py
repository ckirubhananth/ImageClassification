import torch
from src.models.cnn_model import CNN  # Import CNN model architecture

def load_model(model_path):
    """ Load a trained PyTorch model from the given file path. """
    try:
        model = CNN()  # Initialize CNN architecture
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set model to evaluation mode
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Test execution (Optional)
if __name__ == "__main__":
    test_model_path = "saved_models/trained_model.pth"  # Update with actual path
    loaded_model = load_model(test_model_path)
