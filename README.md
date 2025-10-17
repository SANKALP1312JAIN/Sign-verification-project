# 🔐 SigniSure: Deep Learning-Based Signature Authentication System

SigniSure is a Flask-powered web application for real-time signature verification using deep learning. It employs a custom Siamese Neural Network trained on the [CEDAR Signature Dataset](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset) to authenticate signatures with high precision.

## ✨ Key Features
- **Signature Verification:** Compare two signatures to determine authenticity
- **Signature Detection:** Automatically locate signatures in documents/PDFs
- **Multi-Format Support:** Process both image files and PDF documents
- **Intuitive Interface:** Clean, responsive web UI
- **High Accuracy:** 87.9% test accuracy on CEDAR dataset

## 🧠 Model Performance

| Metric        | Value   |
|---------------|---------|
| Accuracy      | 87.9%   |
| Precision     | 82.1%   |
| Recall        | 96.1%   |
| F1-Score      | 88.5%   |
| ROC AUC       | 95.7%   |

- **Architecture:** Siamese Neural Network with custom CNN encoder
- **Training Data:** [CEDAR Signature Dataset](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset)
- **Optimal Threshold:** 0.2602 (determined via ROC analysis)

## 📁 Project Structure

```plaintext
SigniSure/
├── app.py                    # Flask application entry point
├── signature_utils.py        # Signature processing functions
├── model_loader.py           # Model architecture & loading
├── detect_signature.py       # Signature region detection
├── requirements.txt          # Python dependencies
├── templates/                # HTML templates
├── static/                   # CSS, images, JavaScript
├── uploads/                  # User-uploaded files (.gitignore)
├── best_signature_model.pth  # Trained model weights
└── README.md                 # Project documentation
```

## 🚀 Quick Start

1. **Clone repository:**
    ```bash
    git clone https://github.com/GajananTongale/SigniSure.git
    cd SigniSure
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download dataset (optional for retraining):**
    - Download [CEDAR Dataset](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset)
    - Place in the `data/` directory

4. **Launch application:**
    ```bash
    python app.py
    ```

5. **Access in browser:**  
   Go to `http://localhost:5000`

## 💻 Usage

- **Signature Verification:** Upload reference and test signatures
- **Signature Detection:** Process images/PDFs to locate signatures
- **PDF Processing:** Use dropdown to analyze previously uploaded PDFs

## 🌐 Deployment

| Platform        | Type        | Notes                          |
|-----------------|-------------|--------------------------------|
| Render          | Full stack  | Supports Flask backend         |
| Railway         | Full stack  | Easy Python deployment         |
| GitHub Pages    | Frontend    | Static files only (no backend) |

> **Note:** Flask backend requires WSGI-compatible hosting.

## 👥 Development Team

| Member              | Institution                          | GitHub Profile                          |
|---------------------|--------------------------------------|-----------------------------------------|
| Sankalp Jain        | Delhi Technological University (DTU) | [SANKALP1312JAIN](https://github.com/SANKALP1312JAIN) 

**Internship:**  
Developed at Indian Oil Corporation Limited – Panipat Refinery (IS Dept.)

## 📜 License

© 2024 Sankalp Jain 
Developed during internship at Indian Oil Corporation Limited – Panipat Refinery

> **This project is for educational/demonstration purposes only.  
> All rights reserved. Reproduction, redistribution, or commercial use without explicit permission is strictly prohibited.**
