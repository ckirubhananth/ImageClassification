import tkinter as tk
from tkinter import filedialog

def select_model_file():
    """ Open file selection dialog for choosing a single model file (.pth). """
    file_path = filedialog.askopenfilename(
        title="Select Model File",
        filetypes=[("PyTorch Model Files", "*.pth")]
    )
    return file_path

def select_folder(title="Select Folder"):
    """ Opens a dialog for selecting a folder and returns the selected path. """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_selected = filedialog.askdirectory(title=title)
    return folder_selected

def select_training_data():
    """ Opens dialog for selecting the training data folder. """
    return select_folder("Select Training Data Folder")

def select_model_folder():
    """ Opens dialog for selecting an existing trained model folder. """
    return select_folder("Select Trained Model Folder")

def select_prediction_images():
    """ Opens dialog for selecting a folder containing images for classification. """
    return select_folder("Select Folder with Images to Classify")

def select_output_folder():
    """ Opens dialog for selecting the output folder where classified images will be saved. """
    return select_folder("Select Output Folder for Processed Images")

# Example usage
if __name__ == "__main__":
    print("Training Data:", select_training_data())
    print("Model Folder:", select_model_folder())
    print("Images to Predict:", select_prediction_images())
    print("Output Folder:", select_output_folder())
