# 🩺 Knee MRI Soft-Tissue Tear Classification & Grad-CAM Visual Explainability
### *A Multi-Planar Deep Learning Pipeline for ACL & Meniscal Tear Detection using CNN + BiLSTM Sequences*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Gradio UI](https://img.shields.io/badge/Gradio-Demo-orange.svg)](https://gradio.app/)

---

## 📌 1. Project Overview & Clinical Rationale

Musculoskeletal knee injuries—specifically **Anterior Cruciate Ligament (ACL) tears** and **meniscal tears**—are among the most prevalent joint pathologies requiring diagnostic imaging.

* **Why MRI over Plain Radiographs (X-rays)?** Plain X-rays capture dense cortical bone but are transparent to soft-tissue structures like ligaments and menisci. Magnetic Resonance Imaging (MRI) is the gold standard modality for non-invasive soft-tissue joint evaluation.
* **MRNet Pulse Sequences & Modality Specifications:**
  * **Sagittal Series:** T2-weighted sequence *(Primary view for ACL tear visualization)*.
  * **Coronal Series:** T1-weighted sequence.
  * **Axial Series:** Proton Density (PD)-weighted sequence.
* **Core Technical Pipeline:**
  1. 2D Convolutional Feature Extraction via fine-tuned ResNet18 backbone.
  2. 1D Spatial Sequence Reasoning via Bidirectional LSTM (BiLSTM).
  3. Temporal Attention Aggregation for slice relevance weighting.
  4. Multi-Planar Fusion via Logistic Regression ensembling across Sagittal (T2) and Axial (PD) planes.
  5. Grad-CAM visual heatmaps projecting visual attention onto original MRI slices.

---

## 📊 2. Empirical Validation Results (Stanford MRNet Dataset)

Evaluated on the official **Stanford MRNet Validation Set** (120 3D MRI exams, ACL Tear Task) with **1,000-resample Bootstrap 95% Confidence Intervals**:

| Model Strategy | Series View | ROC-AUC (95% CI) | Decision Threshold ($\tau^*$) | Accuracy | Precision | Sensitivity | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNN Max-Pooling (Baseline)** | Axial (PD) | 0.782 [0.697 – 0.858] | 0.50 | 70.8% | 75.7% | 51.9% | 0.615 |
| **CNN + BiLSTM (Single Plane)** | Axial (PD) | 0.850 [0.778 – 0.917] | 0.74 | **80.8%** | **87.8%** | 66.7% | 0.758 |
| **CNN + BiLSTM (Primary View)** | Sagittal (T2) | 0.813 [0.725 – 0.882] | 0.49 | 76.7% | 75.0% | 72.2% | 0.736 |
| **Multi-Plane Fused Ensemble** | Sagittal + Axial | **0.849** [0.775 – 0.915] | **0.40** | **80.8%** | **81.6%** | **74.1%** | **0.777** |

---

## 🧪 3. Systematic Sequence Pooling Ablation Suite

Evaluating sequence aggregation mechanisms:

| Pooling Mechanism | ROC-AUC | Accuracy | Structural Cause & Finding |
| :--- | :---: | :---: | :--- |
| **Slice Mean-Pooling** | **0.540** | **45.0%** | **Fails almost completely** because focal tears occupy only 1–2 slices out of 16; averaging across all slices dilutes the focal lesion signal. |
| **Attention-Only (No LSTM)** | **0.723** | **58.3%** | Learns slice importance weights, improving over naive mean pooling. |
| **Slice Max-Pooling (Baseline)** | **0.782** | **70.8%** | Isolates maximum slice feature map activation (standard MRNet baseline). |
| **BiLSTM + Attention (Proposed)** | **0.849** | **80.8%** | Captures 3D spatial slice sequence continuity across anatomical planes. |

---

## 🚀 4. How to Run the Codebase

### 1. Requirements Setup
```bash
pip install torch torchvision gradio scikit-learn matplotlib opencv-python-headless tqdm
```

### 2. Multi-Plane Model Training
```bash
# Train Sagittal series model (Primary ACL view)
python train.py --task acl --plane sagittal --epochs 3 --batch_size 4 --num_slices 12

# Train Axial series model
python train.py --task acl --plane axial --epochs 3 --batch_size 4 --num_slices 12
```

### 3. Multi-Plane Fusion & Bootstrap CI Evaluation
```bash
python fusion.py --task acl --num_slices 12
```

### 4. Sequence Aggregation Ablation Experiments
```bash
python ablation.py --plane sagittal --task acl --num_slices 12
```

### 5. Interactive Gradio Web Demo App
```bash
python app.py
```
Open **`http://127.0.0.1:7861`** in your browser.

---

## 📄 5. Documentation & Reports

* Master Project Documentation: [`aboutmodel.md`](file:///c:/Users/USER/OneDrive/Desktop/DEEP%20LEARNING/project/aboutmodel.md)
* Summary Report: [`PROJECT_REPORT.md`](file:///c:/Users/USER/OneDrive/Desktop/DEEP%20LEARNING/project/PROJECT_REPORT.md)
* Course Memory & Roadmap: [`dl-course-project.md`](file:///c:/Users/USER/OneDrive/Desktop/DEEP%20LEARNING/project/dl-course-project.md)

---

## ⚠️ 6. Educational & Research Prototype Disclaimer

> [!IMPORTANT]
> **Research Prototype Notice:** This framework is developed exclusively for academic research evaluation and educational demonstration. It is **not** a certified medical device and must **not** be used for clinical diagnostic decision-making. All model outputs require verification by a board-certified radiologist.
