import tkinter as tk
from tkinter import ttk
from src.ui.file_selection import select_model_file, select_prediction_images, select_output_folder
from src.models.predict_images import classify_images  # Import the business logic

class PredictImagesUI(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        ttk.Label(self, text="Image Classification Progress", font=("Arial", 16)).pack(pady=10)

        # Model selection dropdown
        ttk.Label(self, text="Select Model:", font=("Arial", 12)).pack(pady=5)
        self.model_options = ["CNN", "ResNet-50", "EfficientNet-B2"]
        self.model_selection = ttk.Combobox(self, values=self.model_options)
        self.model_selection.pack()
        self.model_selection.bind("<<ComboboxSelected>>", self.update_description)

        # Model description label
        self.model_description = ttk.Label(self, text="", wraplength=300, foreground="gray")
        self.model_description.pack(pady=5)

        # Model file selection
        self.model_button = ttk.Button(self, text="Select Model File", command=self.select_model)
        self.model_button.pack()
        self.model_label = ttk.Label(self, text="No file selected", foreground="gray")
        self.model_label.pack()

        # Image folder selection
        self.image_button = ttk.Button(self, text="Select Image Folder", command=self.select_images)
        self.image_button.pack()
        self.image_label = ttk.Label(self, text="No folder selected", foreground="gray")
        self.image_label.pack()

        # Output folder selection
        self.output_button = ttk.Button(self, text="Select Output Folder", command=self.select_output)
        self.output_button.pack()
        self.output_label = ttk.Label(self, text="No folder selected", foreground="gray")
        self.output_label.pack()

        # Start Classification Button
        self.start_button = ttk.Button(self, text="Start Classification", command=self.start_prediction, state="disabled")
        self.start_button.pack(pady=10)

        # Progress bar
        self.progress_bar = ttk.Progressbar(self, length=400, mode="determinate")
        self.progress_bar.pack(pady=10)

        self.model_file = None
        self.image_folder = None
        self.output_folder = None
        self.selected_model = None  # Track user-selected model

    def update_description(self, event=None):
        """ Updates model description based on selection. """
        descriptions = {
            "CNN": "Basic Convolutional Neural Network for image classification.",
            "ResNet-50": "Deep Residual Network, optimized for feature extraction.",
            "EfficientNet-B2": "Highly efficient model with strong accuracy and low computational cost."
        }
        self.selected_model = self.model_selection.get()
        self.model_description.config(text=descriptions.get(self.selected_model, ""))
        self.check_ready()

    def select_model(self):
        """ Select model file and update label. """
        self.model_file = select_model_file()
        self.model_label.config(text=self.model_file if self.model_file else "No file selected")
        self.check_ready()

    def select_images(self):
        """ Select image folder and update label. """
        self.image_folder = select_prediction_images()
        self.image_label.config(text=self.image_folder if self.image_folder else "No folder selected")
        self.check_ready()

    def select_output(self):
        """ Select output folder and update label. """
        self.output_folder = select_output_folder()
        self.output_label.config(text=self.output_folder if self.output_folder else "No folder selected")
        self.check_ready()

    def check_ready(self):
        """ Enable start button only if model, data folder, and selected model type are chosen. """
        if self.model_file and self.image_folder and self.output_folder and self.selected_model:
            self.start_button.config(state="normal")
        else:
            self.start_button.config(state="disabled")

    def start_prediction(self):
        """ Call prediction logic using the selected model type. """
        if not all([self.model_file, self.image_folder, self.output_folder, self.selected_model]):
            print("Error: Select all required paths and a model!")
            return

        classify_images(self.selected_model, self.model_file, self.image_folder, self.output_folder, self.progress_bar, self)

        print(f"Classification Completed with {self.selected_model}!")

# To integrate this, call `PredictImagesUI(parent_frame)` inside `MainWindow.py`
