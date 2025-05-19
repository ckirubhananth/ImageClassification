import tkinter as tk

class MenuBar:
    def __init__(self, root, main_app):
        self.root = root
        self.main_app = main_app

        # Create menu bar
        self.menu_bar = tk.Menu(root)

        # Add menu options
        self.menu_bar.add_command(label="Train a Model", command=self.main_app.train_model_ui)
        self.menu_bar.add_command(label="Open an Existing Model", command=self.main_app.open_existing_model_ui)
        self.menu_bar.add_command(label="Predict Images", command=self.main_app.predict_images_ui)

        # Attach menu to root window
        self.root.config(menu=self.menu_bar)
