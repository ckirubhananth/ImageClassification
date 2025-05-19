import tkinter as tk
from tkinter import ttk
from src.ui.train_ui import TrainModelUI
from src.ui.load_model_ui import LoadModelUI
from src.ui.predict_ui import PredictImagesUI

class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Photo Classifier")
        self.root.geometry("800x600")

        # Create Navigation Buttons
        self.nav_frame = ttk.Frame(root, padding=10)
        self.nav_frame.pack(fill="x")

        self.train_button = ttk.Button(self.nav_frame, text="Train Model", command=self.show_train)
        self.train_button.pack(side="left", padx=5)

        self.predict_button = ttk.Button(self.nav_frame, text="Predict Images", command=self.show_predict)
        self.predict_button.pack(side="left", padx=5)

        # Create Main Content Frame (where UI sections will be displayed)
        self.content_frame = ttk.Frame(root, padding=20)
        self.content_frame.pack(fill="both", expand=True)

        # Initialize UI Sections (hidden by default)
        self.train_ui = TrainModelUI(self.content_frame)
        self.load_ui = LoadModelUI(self.content_frame)
        self.predict_ui = PredictImagesUI(self.content_frame)

        # Show Default Section
        self.show_train()

    def show_train(self):
        """ Show Train Model UI """
        self.hide_all()
        self.train_ui.pack(fill="both", expand=True)

    def show_load(self):
        """ Show Load Model UI """
        self.hide_all()
        self.load_ui.pack(fill="both", expand=True)

    def show_predict(self):
        """ Show Predict Images UI """
        self.hide_all()
        self.predict_ui.pack(fill="both", expand=True)

    def hide_all(self):
        """ Hide all frames before showing new content """
        self.train_ui.pack_forget()
        self.load_ui.pack_forget()
        self.predict_ui.pack_forget()

# Run Main Window independently
if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()
