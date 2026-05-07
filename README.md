# ECG Classification using ECGMaxViT

> A hybrid deep learning model combining **EfficientNetV2-S (CNN)** and **MaxViT (Vision Transformer)** for automated 12-lead ECG image classification.

---

## 📌 Project Description

This project classifies 12-lead ECG images into **4 cardiac conditions** using a novel hybrid architecture called **ECGMaxViT**. It combines the local feature extraction power of a CNN backbone with the global contextual understanding of a Multi-Axis Vision Transformer (MaxViT).

**Problem it solves:**
Manual ECG interpretation is time-consuming and requires expert knowledge. This model automates the classification process, helping in early and accurate detection of cardiac conditions such as Myocardial Infarction (MI), Abnormal Heartbeat (AHB), Previous Myocardial Infarction (PMI), and Normal ECG.

---

## ✨ Features

- Hybrid CNN + Vision Transformer architecture (ECGMaxViT)
- Automatic dataset discovery and class mapping
- Clinically-aware data augmentation (color space transforms only — no rotation/flip)
- Gradient accumulation for effective larger batch training
- Early stopping with model checkpointing
- Training visualization (loss curves, confusion matrix)
- Multi-class accuracy, F1-score, precision, and recall evaluation

---

## ⚙️ Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ECGMaxViT.git
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

### Loading a Saved Model
```python
import torch

model = ECGMaxViT(num_classes=4, pretrained=False)
model.load_state_dict(torch.load('ecg_maxvit_best.pth'))
model.eval()
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
- Color space augmentation (BGRA, XYZ, HLS, RGBA) — no rotation or flipping to preserve ECG waveform polarity
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
        │
        ├──► MaxViT Module ──────────────► 512-dim global features
        │
        └──► MLP Fusion
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

**Total Parameters:** 51,504,668 (~206 MB)

---

## 📊 Results / Output

### Best Performance
| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **96.74%** |
| Best Validation F1 Score | 96.29% |
| Training stopped at | Epoch 51 (early stopping) |
| Final Training Accuracy | ~99.9% |

### Training Progression

| Epoch | Train Loss | Train Acc | Val Acc | Val F1 |
|-------|-----------|-----------|---------|--------|
| 1     | 0.9910    | 63.79%    | 80.29%  | 79.17% |
| 10    | 0.4374    | 96.11%    | 94.09%  | 94.23% |
| 20    | 0.3857    | 98.48%    | 95.15%  | 95.15% |
| 30    | 0.3661    | 99.44%    | 95.22%  | 95.20% |
| 40    | 0.3593    | 99.72%    | 96.06%  | 96.29% |
| 51*   | —         | —         | **96.74%** | — |

*Early stopped (patience = 20 epochs)

### Training Configuration Used

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

```
MIT License — free to use, modify, and distribute with attribution.
```

---

## 👤 Author Information

**Developed by Chetan Chaudhari**

- 📧 Feel free to open an issue for questions or suggestions
- ⭐ If you found this project helpful, please give it a star!

---

## 🙏 Acknowledgements

- [timm](https://github.com/huggingface/pytorch-image-models) — EfficientNetV2-S and MaxViT pretrained models
- [Mendeley Data](https://data.mendeley.com/datasets/gwbz3fsgp8) — ECG Image Dataset
- [Kaggle](https://kaggle.com) — GPU compute platform

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{ecgmaxvit2026,
  title        = {ECGMaxViT: A Hybrid CNN and MaxViT Model for 12-Lead ECG Image Classification},
  author       = {Chetan Chaudhari},
  year         = {2026},
  howpublished = {\url{https://github.com/YOUR_USERNAME/ECGMaxViT}}
}
```

**Dataset:**
```
ECG Image Dataset, Mendeley Data
DOI: 10.17632/gwbz3fsgp8
```
