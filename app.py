import tkinter as tk
from src.ui.main_window import MainWindow  # Import main UI class
import sys
def main():
    sys.path.append("src/")
    root = tk.Tk()
    app = MainWindow(root)  # Initialize the main window
    root.mainloop()


if __name__ == "__main__":
    main()
