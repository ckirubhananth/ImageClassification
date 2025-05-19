import os
import torch
import tkinter as tk
from tkinter import ttk
from src.ui.file_selection import select_model_folder
from src.models.load_model import load_model

class LoadModelUI(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)  # Correctly attach to parent window

        ttk.Label(self, text="Load Trained Model", font=("Arial", 16)).pack(pady=10)

        # Model folder selection
        self.model_button = ttk.Button(self, text="Select Model Folder", command=self.select_model)
        self.model_button.pack()
        self.model_label = ttk.Label(self, text="No folder selected", foreground="gray")
        self.model_label.pack()

        # Load Model Button
        self.load_button = ttk.Button(self, text="Load Model", command=self.load_selected_model, state="disabled")
        self.load_button.pack(pady=10)

        # Model Info Display
        self.status_label = ttk.Label(self, text="Model Status: Not Loaded", foreground="red")
        self.status_label.pack()

        self.model_folder = None
        self.model = None

    def select_model(self):
        """ Select model folder and update label. """
        self.model_folder = select_model_folder()
        self.model_label.config(text=self.model_folder if self.model_folder else "No folder selected")
        self.check_ready()

    def check_ready(self):
        """ Enable load button only if folder is selected. """
        if self.model_folder:
            self.load_button.config(state="normal")
        else:
            self.load_button.config(state="disabled")

    def load_selected_model(self):
        """ Load the trained model and update UI. """
        if not self.model_folder:
            self.status_label.config(text="Error: No model selected!", foreground="red")
            return

        # Find model file
        model_files = [f for f in os.listdir(self.model_folder) if f.endswith(".pth")]
        if not model_files:
            self.status_label.config(text="No trained model found!", foreground="red")
            return

        model_path = os.path.join(self.model_folder, model_files[0])
        self.model = load_model(model_path)

        if self.model is None:
            self.status_label.config(text="Error loading model!", foreground="red")
        else:
            self.status_label.config(text=f"Model Loaded: {model_files[0]}", foreground="green")

# To integrate this, call `LoadModelUI(parent_frame)` inside `MainWindow.py`
