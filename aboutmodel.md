# 🩺 AboutModel: Knee MRI Soft-Tissue Tear Classification & Explainability Framework
### *Deep Learning Architecture for Knee MRI ACL & Meniscal Tear Detection using CNN + BiLSTM Sequences and Grad-CAM Visual Interpretability*

---

## 📋 Table of Contents
1. [Executive Summary & Project Overview](#1-executive-summary--project-overview)
2. [Clinical Background & Medical Imaging Physics](#2-clinical-background--medical-imaging-physics)
3. [Stanford MRNet Dataset Architecture](#3-stanford-mrnet-dataset-architecture)
4. [Neural Network Architectures & Model Design](#4-neural-network-architectures--model-design)
5. [Visual Explainability via Grad-CAM Heatmaps](#5-visual-explainability-via-grad-cam-heatmaps)
6. [Empirical Performance Results & Multi-Plane Fusion](#6-empirical-performance-results--multi-plane-fusion)
7. [Systematic Sequence Pooling Ablation Suite](#7-systematic-sequence-pooling-ablation-suite)
8. [Interactive Gradio Web Application](#8-interactive-gradio-web-application)
9. [Codebase File Map & Execution Guide](#9-codebase-file-map--execution-guide)
10. [Academic Limitations & Research Framing](#10-academic-limitations--research-framing)

---

## 1. Executive Summary & Project Overview

Musculoskeletal knee injuries—specifically **Anterior Cruciate Ligament (ACL) tears** and **meniscal tears**—are among the most prevalent joint pathologies requiring advanced diagnostic imaging.

This project implements an automated multi-plane deep learning framework for **knee MRI tear classification**. Utilizing the **Stanford MRNet Dataset**, the pipeline extracts slice-level representations via a fine-tuned ResNet18 backbone, compares standard **Max-Pooling (MRNet baseline)** against a **Bidirectional LSTM (BiLSTM)** sequence architecture, ensembles multi-plane views via **Logistic Regression**, and projects **Grad-CAM visual heatmaps** to highlight visual saliency.

---

## 2. Clinical Background & Medical Imaging Physics

### A. Soft Tissue vs. Cortical Bone
* Plain radiographs (X-rays) capture dense cortical bone but are transparent to soft-tissue structures like ligaments (ACL/PCL), meniscus cartilages, and articular surfaces. X-rays cannot visualize soft-tissue ligament tears.
* Magnetic Resonance Imaging (MRI) is the gold standard modality for non-invasive joint assessment.

### B. MRNet Pulse Sequence Physics & Modality Specifications
The dataset utilizes three distinct anatomical plane orientations, each acquired with specific pulse sequence physics:

| Plane View | MRI Pulse Sequence Type | Tissue Physics & Diagnostic Purpose |
| :--- | :--- | :--- |
| **Sagittal Series** | **T2-Weighted Sequence** | **Primary anatomical view for ACL tear visualization.** Fluid/edema appears hyperintense (bright). |
| **Coronal Series** | **T1-Weighted Sequence** | High anatomical detail for subchondral bone and joint space margins. |
| **Axial Series** | **Proton Density (PD)-Weighted** | High SNR for cartilage surfaces and ligament cross-sections. |

*(Note: None of the MRNet sequences is STIR, and only the Sagittal series is T2-weighted).*

---

## 3. Stanford MRNet Dataset Architecture

### A. Dataset Overview (`MRNet-v1.0`)
* **Total Exams:** 1,370 knee MRI series (1,130 training exams, 120 official validation exams).
* **Voxel Array Structure:** Each exam consists of 3D series saved as `.npy` arrays with dimensions $(S, H, W)$, where $S$ is the number of 2D slices ($S \in [12, 44]$) and $H = W = 256$ pixels.
* **Target Diagnostic Tasks:**
  1. `ACL Tear` (Anterior Cruciate Ligament Tear)
  2. `Meniscal Tear`
  3. `General Abnormality`

### B. Prevention of Validation Leakage
To prevent optimistic bias and validation leakage:
* **Internal 15% Validation Split (`split='val'`):** Derived 15% from training exams ($N = 200$) for model checkpoint saving (`best_lstm_sagittal.pth`, `best_lstm_axial.pth`) and decision threshold selection ($\tau^*$).
* **Official Test Set (`split='valid'`):** Evaluated strictly **once** to report unbiased test metrics.

### C. Class Imbalance Penalty & Data Augmentation
* ACL tear positive prevalence in training data is ~18.4%.
* Weighted Binary Cross-Entropy (`BCEWithLogitsLoss`) is applied with positive weight $w_{\text{pos}} = 4.43$:
  $$\mathcal{L} = - w_{\text{pos}} \cdot y \log(\sigma(z)) - (1 - y) \log(1 - \sigma(z))$$
* **Augmentations:** Random Horizontal Flip ($p=0.5$), Random Affine Translation ($\pm 5\%$, $\pm 10^\circ$), and Color/Intensity Jitter ($10\%$).

---

## 4. Neural Network Architectures & Model Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Input Knee MRI Scan (S Slices)                    │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     │ ResNet18 Feature Extractor     │
                     │ (Frozen Early Layers)         │
                     └───────────────┬───────────────┘
                                     │ (S, 512)
                 ┌───────────────────┴───────────────────┐
                 │                                       │
  [Baseline: Max-Pooling]                    [Proposed: BiLSTM]
  ┌──────────────┴──────────────┐           ┌────────────┴────────────┐
  │ Max-Pool over Sequence S    │           │ Bidirectional LSTM      │
  │ (1, 512)                    │           │ (S, 256)                │
  └──────────────┬──────────────┘           └────────────┬────────────┘
                 │                                       │ Attention Weighting
  ┌──────────────┴──────────────┐           ┌────────────┴────────────┐
  │ Linear Classifier           │           │ Attention-Pooled FC     │
  └──────────────┬──────────────┘           └────────────┬────────────┘
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │  Output Probability Score &     │
                    │  Grad-CAM Visual Heatmap       │
                    └─────────────────────────────────┘
```

### A. Slice Feature Extractor (`SliceFeatureExtractor`)
* Backbone: Pretrained ResNet18. Early layers (`conv1`, `bn1`, `layer1`) are frozen to prevent overfitting.
* Output per slice: 512-dimensional feature vector $\mathbf{f}_i \in \mathbb{R}^{512}$.

### B. Baseline Model (`MRNetBaseline`)
* Slice features $\mathbf{f}_i$ are max-pooled across the sequence dimension $S$:
  $$\mathbf{g} = \max_{i=1}^S \mathbf{f}_i$$
* Linear classification head maps $\mathbf{g} \to \text{logit}$.

### C. Proposed Model (`MRNetLSTM`)
* Sequence $\{\mathbf{f}_1, \dots, \mathbf{f}_S\}$ is processed by a 1-layer Bidirectional LSTM (hidden dimension = 128, bidirectional = 256 outputs).
* An attention mechanism computes slice importance weights $\alpha_i$:
  $$\alpha_i = \frac{\exp(\mathbf{w}^T \mathbf{h}_i)}{\sum_{j=1}^S \exp(\mathbf{w}^T \mathbf{h}_j)}$$
  $$\mathbf{c} = \sum_{i=1}^S \alpha_i \mathbf{h}_i$$
* Fully connected classifier with Dropout (0.3) maps $\mathbf{c} \to \text{logit}$.

---

## 5. Visual Explainability via Grad-CAM Heatmaps

Grad-CAM (Gradient-weighted Class Activation Mapping) extracts spatial gradients from `layer4` of the ResNet backbone:

$$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial Y^c}{\partial A_{i,j}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

The resulting activation map is overlaid onto the original MRI slice using a JET color map. Red/yellow hyperintense regions correlate with deep feature activation saliency.

---

## 6. Empirical Performance Results & Multi-Plane Fusion

Evaluated on the official **MRNet 120-exam validation set** (ACL Tear Task) with **1,000-resample Bootstrap 95% Confidence Intervals**:

| Model Architecture / Strategy | Series View | ROC-AUC (95% CI) | Decision Threshold ($\tau^*$) | Accuracy | Precision | Sensitivity (Recall) | F1-Score | Sens @ 90% Spec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNN Max-Pooling (Baseline)** | Axial (PD) | 0.782 [0.697 – 0.858] | 0.50 | 70.8% | 75.7% | 51.9% | 0.615 | 46.3% |
| **CNN + BiLSTM (Single Plane)** | Axial (PD) | 0.850 [0.778 – 0.917] | 0.74 | **80.8%** | **87.8%** | 66.7% | 0.758 | 66.7% |
| **CNN + BiLSTM (Primary View)** | Sagittal (T2) | 0.813 [0.725 – 0.882] | 0.49 | 76.7% | 75.0% | 72.2% | 0.736 | 57.4% |
| **Multi-Plane Fused Ensemble** | Sagittal + Axial | **0.849** [0.775 – 0.915] | **0.40** | **80.8%** | **81.6%** | **74.1%** | **0.777** | **66.7%** |

---

## 7. Systematic Sequence Pooling Ablation Suite

Evaluated on the official MRNet validation set using `ablation.py`:

| Pooling Mechanism | ROC-AUC | Accuracy | Empirical Finding & Structural Cause |
| :--- | :---: | :---: | :--- |
| **Slice Mean-Pooling** | **0.540** | **45.0%** | **Fails almost completely** because focal tears occupy only 1–2 slices out of 16; averaging across all slices dilutes the focal lesion signal. |
| **Attention-Only (No LSTM)** | **0.723** | **58.3%** | Learns slice importance weights, improving over naive mean pooling. |
| **Slice Max-Pooling (Baseline)** | **0.782** | **70.8%** | Isolates maximum slice feature map activation (standard MRNet baseline). |
| **BiLSTM + Attention (Proposed)** | **0.849** | **80.8%** | Captures 3D spatial slice sequence continuity across anatomical planes. |

---

## 8. Interactive Gradio Web Application

* **Local Web Server:** Live on **[http://127.0.0.1:7861](http://127.0.0.1:7861)**
* **Features:**
  * Exam selector for validation case IDs (`1130`, `1131`, `1132`, ...).
  * Task selector (`ACL Tear`, `Meniscal Tear`, `General Abnormality`).
  * Model architecture comparison (`CNN + BiLSTM` vs `CNN + MaxPool`).
  * Real-time probability score gauge, binary diagnostic badge, original MRI slice viewer, and Grad-CAM visual heatmap overlay.

---

## 9. Codebase File Map & Execution Guide

### Workspace Directory: `C:\Users\USER\OneDrive\Desktop\DEEP LEARNING\project`

1. **[`dataset.py`](file:///C:/Users/USER/OneDrive/Desktop/DEEP%20LEARNING/project/dataset.py):** PyTorch Dataset handling volume loading, 15% internal validation split (`split='val'`), 3-channel RGB conversion, and augmentations.
2. **[`models.py`](file:///C:/Users/USER/OneDrive/Desktop/DEEP%20LEARNING/project/models.py):** `MRNetBaseline`, `MRNetLSTM`, and `GradCAM` hook class.
3. **[`train.py`](file:///C:/Users/USER/OneDrive/Desktop/DEEP%20LEARNING/project/train.py):** Model training loop with weighted loss and plateau LR scheduler.
4. **[`evaluate_and_gradcam.py`](file:///C:/Users/USER/OneDrive/Desktop/DEEP%20LEARNING/project/evaluate_and_gradcam.py):** Generates ROC curves, 95% Bootstrap CIs, and Grad-CAM overlays.
5. **[`fusion.py`](file:///C:/Users/USER/OneDrive/Desktop/DEEP%20LEARNING/project/fusion.py):** Multi-plane Logistic Regression ensembling across Sagittal and Axial views.
6. **[`ablation.py`](file:///C:/Users/USER/OneDrive/Desktop/DEEP%20LEARNING/project/ablation.py):** Sequence aggregation ablation suite.
7. **[`app.py`](file:///C:/Users/USER/OneDrive/Desktop/DEEP%20LEARNING/project/app.py):** Gradio web application.

---

## 10. Academic Limitations & Research Framing

1. **Single-Center Data:** MRNet dataset is sourced exclusively from Stanford University Medical Center.
2. **NLP-Extracted Labels:** Ground truth labels were mined from clinical radiology reports via regex.
3. **Visual Saliency Boundaries:** Grad-CAM heatmaps reflect deep neural network feature activation, not histopathologically verified tissue pathology.
4. **Non-Diagnostic Research Prototype:** Developed strictly for academic demonstration and research benchmarking.
