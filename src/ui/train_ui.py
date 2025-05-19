import tkinter as tk
from tkinter import ttk
from src.ui.file_selection import select_training_data
from src.models.train_model import train_model  # Import business logic

class TrainModelUI(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        ttk.Label(self, text="Train a Model", font=("Arial", 16)).pack(pady=10)

        # Model selection dropdown
        ttk.Label(self, text="Select Model:", font=("Arial", 12)).pack(pady=5)
        self.model_options = ["CNN", "ResNet-50", "EfficientNet-B2"]
        self.model_selection = ttk.Combobox(self, values=self.model_options)
        self.model_selection.pack()
        self.model_selection.bind("<<ComboboxSelected>>", self.update_description)  # Bind selection event

        # Model description label
        self.model_description = ttk.Label(self, text="", wraplength=300, foreground="gray")
        self.model_description.pack(pady=5)

        # Image size information label
        self.image_size_label = ttk.Label(self, text="Image Size: Not Selected", foreground="gray")
        self.image_size_label.pack(pady=5)

        # Training data selection
        self.data_button = ttk.Button(self, text="Select Training Data", command=self.select_data)
        self.data_button.pack()
        self.data_label = ttk.Label(self, text="No folder selected", foreground="gray")
        self.data_label.pack()

        # Train button
        self.train_button = ttk.Button(self, text="Start Training", command=self.start_training, state="disabled")
        self.train_button.pack(pady=10)

        # Progress bar
        self.progress_bar = ttk.Progressbar(self, length=400, mode="determinate")
        self.progress_bar.pack(pady=10)

        # Status label
        self.status_label = ttk.Label(self, text="Training Status: Not Started", foreground="blue")
        self.status_label.pack()

        self.data_folder = None

    def update_description(self, event=None):
        """ Updates model description and expected image size based on selection. """
        descriptions = {
            "CNN": "Basic Convolutional Neural Network. Works well for simple image classifications.",
            "ResNet-50": "Deep Residual Network. Optimized for feature extraction and accurate classification.",
            "EfficientNet-B2": "Highly efficient model with improved accuracy and lower computational cost."
        }
        image_sizes = {
            "CNN": "64x64",
            "ResNet-50": "224x224",
            "EfficientNet-B2": "224x224"
        }

        selected_model = self.model_selection.get()
        self.model_description.config(text=descriptions.get(selected_model, ""))
        self.image_size_label.config(text=f"Image Size: {image_sizes.get(selected_model, 'Not Selected')}")
        self.check_ready()

    def select_data(self):
        """ Select training data folder and update label. """
        self.data_folder = select_training_data()
        self.data_label.config(text=self.data_folder if self.data_folder else "No folder selected")
        self.check_ready()

    def check_ready(self):
        """ Enable training button only if model and folder are selected. """
        if self.data_folder and self.model_selection.get():
            self.train_button.config(state="normal")
        else:
            self.train_button.config(state="disabled")

    def start_training(self):
        """ Start training process with the selected model. """
        selected_model = self.model_selection.get()
        if not self.data_folder or not selected_model:
            print("Error: Please select a model and training data!")
            return

        train_model(selected_model, self.data_folder, "saved_models", self.progress_bar, self)

        print(f"Training {selected_model} Completed!")

# To integrate this, call `TrainModelUI(parent_frame)` inside `MainWindow.py`
