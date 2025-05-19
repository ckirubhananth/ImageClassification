**PhotoClassifierApp 📸**
A dynamic image classification application supporting CNN, ResNet-50, and EfficientNet-B2, with custom preprocessing pipelines, model selection, and UI integration.

*🚀 Features*
✅ Supports CNN (64x64), ResNet-50 & EfficientNet-B2 (224x224) dynamically. ✅ Flexible UI for model training and prediction—no hardcoded dependencies. ✅ Efficient preprocessing pipeline with automated resizing & normalization. ✅ Modular codebase for easy customization and expansion.

*🛠 Installation*
1️⃣ Clone the Repository
```bash
git clone https://github.com/YourUsername/PhotoClassifierApp.git
cd PhotoClassifierApp

2️⃣ Install Dependencies
```bash
pip install -r requirements.txt

3️⃣ Run the Application
For Training Models
```bash
python app.py

*Select CNN, ResNet-50, or EfficientNet-B2 in the UI.
*Choose the training dataset folder.
*Start training and monitor progress.

For Image Classification (Prediction)
```bash
python predict.py

*Select the trained model file.
*Choose an image folder for classification.
*Save classified images to the output folder.

*📜 Folder Structure*
PhotoClassifierApp/
│── src/
│   ├── ui/                  # User interface components
│   ├── models/              # Model implementations (CNN, ResNet-50, EfficientNet-B2)
│   ├── utils/               # Helper functions
│── data/                    # Training datasets
│── saved_models/            # Trained model storage
│── app.py                   # Main application entry
│── predict.py               # Image classification entry
│── requirements.txt         # Dependencies list
│── README.md                # Documentation

*🧩 How It Works*
✔ Model Selection in UI → Supports CNN (lightweight), ResNet-50 (feature-rich), and EfficientNet-B2 (balanced) ✔ Dynamic Resizing → 64x64 for CNN, 224x224 for deeper architectures ✔ Training Pipeline → Fine-tuning models with preprocessing, normalization, and batch optimization ✔ Prediction Workflow → Classifies images and organizes them into folders based on trained labels

Trained all the three models with dogs and cats pictures and the trained models can be located in saved_models folder.