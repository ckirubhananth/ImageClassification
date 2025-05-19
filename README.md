**PhotoClassifierApp 📸**
A dynamic image classification application supporting CNN, ResNet-50, and EfficientNet-B2, with custom preprocessing pipelines, model selection, and UI integration.

*🚀 Features*
✅ Supports CNN (64x64), ResNet-50 & EfficientNet-B2 (224x224) dynamically. ✅ Flexible UI for model training and prediction—no hardcoded dependencies. ✅ Efficient preprocessing pipeline with automated resizing & normalization. ✅ Modular codebase for easy customization and expansion.

*🛠 Installation*
Clone the Repository
```bash
git clone https://github.com/YourUsername/PhotoClassifierApp.git
cd PhotoClassifierApp
```

*Install Dependencies*
```bash
pip install -r requirements.txt
```

*Run the Application*
For Training Models
```bash
python app.py
```

*For Images Training (Training)*
*Select CNN, ResNet-50, or EfficientNet-B2 in the UI.
*Choose the training dataset folder.
*Start training and monitor progress.

*For Image Classification (Prediction)*
*Select the trained model file.
*Choose an image folder for classification.
*Save classified images to the output folder.

*📜 Folder Structure*

![image](https://github.com/user-attachments/assets/631ceb93-1d94-4154-8a49-4db7584fe125)


*🧩 How It Works*
✔ Model Selection in UI → Supports CNN (lightweight), ResNet-50 (feature-rich), and EfficientNet-B2 (balanced) ✔ Dynamic Resizing → 64x64 for CNN, 224x224 for deeper architectures ✔ Training Pipeline → Fine-tuning models with preprocessing, normalization, and batch optimization ✔ Prediction Workflow → Classifies images and organizes them into folders based on trained labels

Trained all the three models with dogs and cats pictures and the trained models can be located in saved_models folder.
