# ECG Classification using ECGMaxViT

> A custom hybrid CNN-Transformer architecture combining **EfficientNetV2-S** and **MaxViT** for automated 12-lead ECG image classification into 4 cardiac conditions.

---

## 📌 Project Description

This project classifies 12-lead ECG images into **4 cardiac conditions** using a custom hybrid architecture called **ECGMaxViT**. It combines the local feature extraction of a CNN backbone (EfficientNetV2-S) with the global contextual understanding of a Multi-Axis Vision Transformer (MaxViT).

**Problem it solves:**
Manual ECG interpretation is time-consuming and requires expert knowledge. This model automates the classification process, helping in early and accurate detection of cardiac conditions — Myocardial Infarction (MI), Abnormal Heartbeat (AHB), Previous Myocardial Infarction (PMI), and Normal ECG.

---

## ✨ Features

- Custom hybrid CNN + Vision Transformer architecture (ECGMaxViT)
- Automatic dataset discovery and class mapping
- Clinically-aware data augmentation — rotations and flips were avoided because they can distort ECG waveform morphology and lead polarity
- Gradient accumulation for effective larger batch training
- Early stopping with model checkpointing
- Training visualization (loss curves, accuracy curves, confusion matrix)
- Multi-class accuracy, F1-score, precision, and recall evaluation

---

## ⚙️ Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/chetnchaudhari/ECGMaxVit.git
cd ECGMaxViT

# 2. Install dependencies
pip install -r requirements.txt
```

> **GPU recommended.** Tested on NVIDIA Tesla T4 (15.6 GB VRAM) via Kaggle.

---

## ▶️ Usage

### Run on Kaggle (Recommended)
1. Upload `ECGMaxViT.ipynb` to [Kaggle](https://kaggle.com)
2. Add the ECG Image Dataset as a Kaggle input
3. Enable GPU accelerator (T4 or P100)
4. Click **Run All**

### Run Locally
```bash
# Launch the notebook
jupyter notebook ECGMaxViT.ipynb
```

> Update the `DATA_ROOT` variable in **Section 3** of the notebook to point to your local dataset folder.

### Inference Example
```python
import torch
from PIL import Image
from torchvision import transforms

# Load model
model = ECGMaxViT(num_classes=4, pretrained=False)
model.load_state_dict(torch.load('ecg_maxvit_best.pth'))
model.eval()

# Predict
CLASS_NAMES = ['AHB', 'MI', 'Normal', 'PMI']

def predict_ecg(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = transform(Image.open(image_path).convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(img), dim=1)
    conf, idx = probs.max(1)
    print(f"Predicted: {CLASS_NAMES[idx]} ({conf.item()*100:.1f}%)")

predict_ecg("sample_ecg.png")
# Output → Predicted: MI (97.3%)
```

---

## 📦 Requirements / Dependencies

```
Python        3.8+
PyTorch       2.0+
torchvision   0.15+
timm          0.9+
torchmetrics  1.0+
opencv-python 4.7+
scikit-learn  1.2+
pandas        1.5+
matplotlib    3.7+
seaborn       0.12+
Pillow        9.5+
numpy         1.23+
jupyterlab    4.0+
ipywidgets    8.0+
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🗂️ Dataset Information

**Source:** ECG Image Dataset — Mendeley Data
- Version 1 → DOI: [10.17632/gwbz3fsgp8.1](https://doi.org/10.17632/gwbz3fsgp8.1)
- Version 2 → DOI: [10.17632/gwbz3fsgp8.2](https://doi.org/10.17632/gwbz3fsgp8.2)

**Format:** JPEG / PNG images of 12-lead ECG printouts

**Classes (COVID-19 excluded):**

| Class | Full Name | Original | After Augmentation |
|-------|-----------|----------|--------------------|
| MI    | Myocardial Infarction | 313 | 1,252 |
| AHB   | Abnormal Heartbeat | 779 | 1,558 |
| PMI   | Previous Myocardial Infarction | 375 | 1,500 |
| Normal | Normal ECG | 1,143 | 2,286 |
| **Total** | | **2,610** | **6,596** |

**Preprocessing:**
- Images resized to **224 × 224**
- Normalized with ImageNet mean and std
- Color space augmentation only (BGRA, XYZ, HLS, RGBA) — rotations and flips were excluded to preserve ECG waveform morphology and lead polarity
- Stratified split: **60% Train / 20% Validation / 20% Test**

---

## 🏗️ Project Structure

```
ECGMaxViT/
│
│── ECGMaxViT.ipynb        ← Main notebook (full pipeline)
│── requirements.txt       ← Python dependencies
│── README.md              ← Project documentation
│── LICENSE                ← MIT License
│── .gitignore             ← Ignored files (weights, data, checkpoints)
│
│── data/                  ← Dataset folder (download separately)
│   │── gwbz3fsgp8-1/
│   │   │── MI/
│   │   │── AHB/
│   │   │── PMI/
│   │   └── Normal/
│   └── gwbz3fsgp8-2/
│       │── MI/
│       │── AHB/
│       │── PMI/
│       └── Normal/
│
│── outputs/               ← Saved model weights, plots (generated at runtime)
│   │── ecg_maxvit_best.pth
│   │── confusion_matrix.png
│   └── training_curves.png
```

**Notebook Sections:**

| Section | Description |
|---------|-------------|
| 1 | Install Dependencies |
| 2 | Import Libraries & Device Setup |
| 3 | Dataset Setup & Exploration |
| 4 | Data Augmentation |
| 5 | ECGDataset & DataLoader |
| 6 | ECGMaxViT Model Architecture |
| 7 | Training Configuration |
| 8 | Training Loop |
| 9 | Evaluation & Metrics |
| 10 | Visualization |

---

## 🧠 Model Architecture

```
Input Image (224 × 224 × 3)
        │
        ├──► ModCNN (EfficientNetV2-S) ──► 1280-dim local features
        │         └─ Depthwise-separable convolutions
        │             Global Average Pooling
        │
        ├──► MaxViT Module ──────────────► 512-dim global features
        │         └─ Local window attention
        │             Global grid attention
        │
        └──► MLP Fusion Head
                  Concatenate [1280 + 512] = 1792 dims
                  LayerNorm → FC(512) → GELU → Dropout(0.3)
                  FC(512 → 4)
                  └──► Output: MI | AHB | PMI | Normal
```

| Module | Backbone | Output | Role |
|--------|----------|--------|------|
| ModCNN | EfficientNetV2-S | 1280-dim | Local spatial features |
| MaxViTModule | maxvit_tiny_tf_224 | 512-dim | Global contextual features |
| MLPFusion | Custom MLP | 4-class logits | Fusion + Classification |

**Total Parameters:** ~51.5M

---

## 📊 Results / Output

### Overall Performance

| Metric | Validation Set | Test Set |
|--------|---------------|----------|
| **Accuracy** | **96.74%** | *(run notebook to generate)* |
| **F1 Score** | **96.29%** | *(run notebook to generate)* |
| Precision | — | *(run notebook to generate)* |
| Recall | — | *(run notebook to generate)* |

> Test set metrics and per-class breakdown (including confusion matrix) are generated automatically at the end of the notebook.

### Training Progression

| Epoch | Train Loss | Train Acc | Val Acc | Val F1 |
|-------|-----------|-----------|---------|--------|
| 1     | 0.9910    | 63.79%    | 80.29%  | 79.17% |
| 10    | 0.4374    | 96.11%    | 94.09%  | 94.23% |
| 20    | 0.3857    | 98.48%    | 95.15%  | 95.15% |
| 30    | 0.3661    | 99.44%    | 95.22%  | 95.20% |
| 40    | 0.3593    | 99.72%    | 96.06%  | 96.29% |
| 51*   | —         | —         | **96.74%** | — |

*Early stopped at epoch 51 (patience = 20 epochs)

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning Rate | 2e-4 |
| Weight Decay | 1e-4 |
| LR Scheduler | 5-epoch warmup → Cosine Decay |
| Loss Function | CrossEntropyLoss (label_smoothing=0.1) |
| Effective Batch Size | 32 (8 physical × 4 accumulation steps) |
| Gradient Clipping | max_norm = 1.0 |
| Max Epochs | 100 |
| Early Stopping Patience | 20 epochs |

---

## ⚠️ Limitations

- Trained on ECG **image** data rather than raw signal data — performance may differ on raw signal-based approaches
- Dataset size remains relatively limited (~2,600 original images across 4 classes)
- External clinical validation was not performed
- Model performance may vary across different hospitals, devices, or ECG recording standards
- The training–validation accuracy gap suggests mild overfitting; further regularization or more data may improve generalization

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork this repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please make sure your code is clean and well-commented.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Developed by Chetan Chaudhari**

- Feel free to open an [issue](../../issues) for questions or suggestions
- ⭐ If you found this project helpful, please give it a star!

---

##  Acknowledgements

- [timm](https://github.com/huggingface/pytorch-image-models) — EfficientNetV2-S and MaxViT pretrained model weights
- [Mendeley Data](https://data.mendeley.com/datasets/gwbz3fsgp8) — ECG Image Dataset
- [Kaggle](https://kaggle.com) — GPU compute platform

---

## 📚 References

- Dosovitskiy et al. — *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (ViT)
- Tu et al. — *MaxViT: Multi-Axis Vision Transformer* (ECCV 2022)
- Tan & Le — *EfficientNetV2: Smaller Models and Faster Training* (ICML 2021)
- ECG Image Dataset — Mendeley Data, DOI: [10.17632/gwbz3fsgp8](https://doi.org/10.17632/gwbz3fsgp8)