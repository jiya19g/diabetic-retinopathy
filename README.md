# DIABETIC RETINOPATHY DETECTION USING DEEP LEARNING
## Digital Image & Video Processing Project

---

# PROJECT OVERVIEW
This project focuses on automated detection and severity classification of Diabetic Retinopathy (DR) using retinal fundus images.

Digital Image Processing + Deep Learning are combined to enhance retinal imagery (CLAHE enhancement) and classify DR into 5 severity levels using CNN-based transfer learning models.

Models used:
- Multi-Branch CNN (DenseNet201 + ResNet50)
- Baseline 2D CNN
- Ensemble Fusion Model

Explainability added using Saliency Maps to visualize key regions influencing predictions.

---

# DATASET USED
Kaggle: APTOS 2019 Blindness Detection  
https://www.kaggle.com/c/aptos2019-blindness-detection  
Labels represent DR severity classes:  
0 - No DR  
1 - Mild  
2 - Moderate  
3 - Severe  
4 - Proliferative DR

---

# SETUP / INSTALLATION REQUIREMENTS

Install dependencies:
pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn tqdm kaggle

---

# DATASET DOWNLOAD (COLAB STEPS)

1) Upload kaggle.json to Colab:
from google.colab import files  
files.upload()

2) Setup kaggle auth:
mkdir ~/.kaggle  
mv kaggle.json ~/.kaggle/  
chmod 600 ~/.kaggle/kaggle.json

3) Download dataset:
kaggle competitions download -c aptos2019-blindness-detection  
unzip aptos2019-blindness-detection.zip -d aptos_data

Update paths inside code:
csv_path = "aptos_data/train.csv"  
img_dir = "aptos_data/train_images"

---

# IMPORTANT PIPELINE STEPS

1) CLAHE enhancement on images (DIP stage)  
2) Normalization + resizing (224 x 224)  
3) Multi-Branch CNN (DenseNet201 + ResNet50 fusion)  
4) Baseline 2D CNN for comparison  
5) Ensemble fusion by weighted averaging  
6) Saliency Maps for explainability

---

# FINAL RESULTS SUMMARY

| Model              | Test Accuracy |
|--------------------|---------------|
| Multi-Branch CNN   | 90.32%        |
| 2D CNN Baseline    | 74.73%        |
| Ensemble Fusion    | 89.96%        |

---

# OUTPUT FILES GENERATED

- multibranch_model_1.h5
- cnn_model_1.h5
- saliency visualizations for interpretability

---

# CONCLUSION
CLAHE preprocessing significantly enhanced retinal vessel visibility, enabling superior feature extraction. Transfer learning improved model convergence while multi-branch CNN delivered the strongest performance. Ensemble improved stability and predictive reliability.

This pipeline demonstrates a clinically relevant, interpretable, and scalable approach for automated Diabetic Retinopathy severity classification using Medical Image Processing + Deep Learning.
