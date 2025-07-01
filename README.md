```markdown
# SigniSure: Deep Learning-Based Signature Authentication

SigniSure is a Flask-based web application for real-time signature verification using deep learning. It leverages a custom Siamese Neural Network trained on the [CEDAR Signature Dataset](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset) to distinguish between genuine and forged signatures with high accuracy.

---

## 🚀 Features
- **Signature Verification:** Upload two signatures and verify if they are genuine or forged.
- **Signature Detection:** Automatically detect signature regions in scanned documents or PDFs.
- **PDF & Image Support:** Works with both image files and PDF documents.
- **Modern Web UI:** Clean, responsive interface for easy use.
- **Model Accuracy:** Achieves **87% accuracy** on the CEDAR test set.

---

## 🧠 Model Details
- **Architecture:** Siamese Neural Network with a custom CNN encoder.
- **Training Dataset:** [CEDAR Signature Dataset](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset)
- **Optimal Threshold:** 0.2602 (determined via ROC analysis)  
✅ **Model Performance:**  
   • Accuracy: 87.9%  
   • Precision: 82.1%  
   • Recall: 96.1%  
   • F1-Score: 88.5%  
   • ROC AUC: 95.7%

---

## 📦 Project Structure
```

├── app.py                  # Main Flask server
├── signature\_utils.py      # Signature processing utilities
├── model\_loader.py         # Model architecture & loading
├── detect\_signature.py     # Signature region detection
├── requirements.txt        # Python dependencies
├── templates/              # HTML templates (Jinja2)
├── static/                 # Static files (CSS, images)
├── uploads/                # Uploaded files (not tracked in git)
├── best\_signature\_model.pth# Trained model weights (not in repo)
└── README.md               # This file

````

---

## ⚡ Quickstart
1. **Clone the repository:**
   ```sh
   git clone https://github.com/GajananTongale/SigniSure.git
   cd SigniSure
````

2. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```
3. **Download the CEDAR dataset:**

   * [CEDAR Signature Dataset on Kaggle](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset)
   * Place sample images in the appropriate folder if you want to retrain.
4. **Run the Flask server:**

   ```sh
   python app.py
   ```
5. **Open your browser:**

   * Go to `http://localhost:5000`

---

## 📝 Usage

* **Verify Signatures:** Upload a genuine and a test signature to check authenticity.
* **Detect Signature:** Upload a document or PDF to auto-detect signature regions.
* **Auto-Detect in Uploaded PDFs:** Use the dropdown to process any previously uploaded PDF.

---

## 🛠️ Deployment

* **Backend:** Deploy Flask app on [Render](https://render.com/), [Railway](https://railway.app/), or similar.
* **Frontend:** Static files can be served via GitHub Pages (for React/static UI only).
* **Note:** GitHub Pages does **not** support Python/Flask backend hosting.

---

## 📚 References

* **Dataset:** [CEDAR Signature Dataset (Kaggle)](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset)
* **Model:** Siamese Neural Network (see `model_loader.py` and notebook for details)

---

## 👥 Collaborators

This project was collaboratively developed by:

* **Sankalp Jain** – Delhi Technological University (DTU)
  [GitHub: SANKALP1312JAIN](https://github.com/SANKALP1312JAIN)

* **Gajanan Tongale** – Vishwakarma Institute of Technology, Pune (VIT-Pune)
  [GitHub: GajananTongale](https://github.com/GajananTongale)

> Developed during their internship under the Information Systems (IS) Department at
> **Indian Oil Corporation Limited – Panipat Refinery**

---

## 📄 License

© 2024 Sankalp Jain (DTU) and Gajanan Tongale (VIT-Pune)
Developed during internship at Indian Oil Corporation Limited – Panipat Refinery

This project is **subject to copyright** and is intended for educational and demonstrative purposes only.
**Reproduction, redistribution, or commercial use without explicit permission is strictly prohibited.**
Not to be copied, republished, or reused in any form without prior consent from the authors.

All rights reserved.

