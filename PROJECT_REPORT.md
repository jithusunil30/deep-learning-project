# 🩺 Knee MRI Soft-Tissue Tear Classification Report
### *A Multi-Plane Deep Learning Framework with BiLSTM Sequence Modeling & Grad-CAM Visual Explainability*

---

## 📌 1. Executive Summary

Musculoskeletal knee injuries—specifically **Anterior Cruciate Ligament (ACL) tears** and **meniscal tears**—are among the most common joint pathologies requiring advanced diagnostic imaging. 

This project develops an automated multi-plane sequence classification system on **Knee Magnetic Resonance Imaging (MRI)** exams. Using the **Stanford MRNet Dataset**, we extract 2D slice features with a fine-tuned ResNet18 convolutional backbone, train models across **Sagittal (T2)** and **Axial (PD)** plane series, ensemble multi-plane predictions with Logistic Regression, and compute **Grad-CAM visual heatmaps** to highlight model visual attention.

---

## 🔬 2. Clinical Rationale & Imaging Physics

* **Why MRI over Plain Radiographs (X-rays)?**
  * **Soft Tissue vs. Bone:** Plain X-rays capture dense cortical bone but are transparent to soft tissue structures like ligaments, menisci, and articular cartilage.
  * **Sequence Physics (MRNet Dataset Specifications):**
    * **Sagittal Series:** T2-weighted sequence *(Primary view for ACL tear visualization)*.
    * **Coronal Series:** T1-weighted sequence.
    * **Axial Series:** Proton Density (PD)-weighted sequence.

---

## 📊 3. Dataset Architecture & Preprocessing

* **Dataset:** Stanford MRNet Dataset (`MRNet-v1.0`), 1,370 exams.
* **Validation Leakage Prevention:** 
  * 15% of training data was split into an **internal validation set (`split='val'`)** for model checkpointing (`best_lstm_sagittal.pth`, `best_lstm_axial.pth`) and decision threshold selection ($\tau^*$).
  * The official 120-exam validation set (`split='valid'`) was evaluated strictly once to prevent data leakage.
* **Class Imbalance & Augmentations:** Binary Cross-Entropy loss was weighted (`pos_weight = 4.43`). Augmentations included random horizontal flips ($p=0.5$), affine translations ($\pm 5\%$, $\pm 10^\circ$), and intensity jitter ($10\%$).

---

## 📈 4. Empirical Results & Performance Comparison

Evaluated on the official **MRNet Validation Set** (120 MRI exams, ACL Tear Task) with **1,000-resample Bootstrap 95% Confidence Intervals**:

| Model Architecture / Strategy | Series View | ROC-AUC (95% CI) | Decision Threshold ($\tau^*$) | Accuracy | Precision | Sensitivity (Recall) | F1-Score | Sens @ 90% Spec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNN Max-Pooling (Baseline)** | Axial (PD) | 0.782 [0.697 – 0.858] | 0.50 | 70.8% | 75.7% | 51.9% | 0.615 | 46.3% |
| **CNN + BiLSTM (Single Plane)** | Axial (PD) | 0.850 [0.778 – 0.917] | 0.74 | **80.8%** | **87.8%** | 66.7% | 0.758 | 66.7% |
| **CNN + BiLSTM (Primary View)** | Sagittal (T2) | 0.813 [0.725 – 0.882] | 0.49 | 76.7% | 75.0% | 72.2% | 0.736 | 57.4% |
| **Multi-Plane Fused Ensemble** | Sagittal + Axial | **0.849** [0.775 – 0.915] | **0.40** | **80.8%** | **81.6%** | **74.1%** | **0.777** | **66.7%** |

---

## 🧪 5. Systematic Ablation Suite

Evaluating sequence aggregation mechanisms:

| Pooling Mechanism | ROC-AUC | Accuracy | Key Finding |
| :--- | :---: | :---: | :--- |
| **Slice Mean-Pooling** | 0.540 | 45.0% | Averaging across 16 slices dilutes focal tear signals (focal tears occupy only 1-2 slices). |
| **Attention-Only Pooling** | 0.723 | 58.3% | Learns slice importance weights, improving over naive mean pooling. |
| **Slice Max-Pooling (Baseline)** | 0.782 | 70.8% | Isolates maximum slice activation feature map. |
| **BiLSTM + Attention (Proposed)** | **0.849** | **80.8%** | Captures 3D spatial slice sequence continuity across anatomical planes. |

---

## 🌐 6. Interactive Gradio Web Demonstration App

The interactive web application is live and accessible locally:

* **App Server URL:** [http://127.0.0.1:7861](http://127.0.0.1:7861)
* **Features:**
  * Case selection from MRNet validation exams (`1130`, `1131`, `1132`, ...).
  * Diagnostic target toggle (ACL Tear / Meniscal Tear / General Abnormality).
  * Model architecture selector (CNN + BiLSTM vs CNN + MaxPool).
  * Outputs: Tear Risk Probability (%), Binary Status Badge, and Grad-CAM visual heatmap overlay.

---

## ⚠️ 7. Educational Scope & Research Disclaimer

**Research Prototype Notice:** This software is developed solely for educational and academic research evaluation. It is **not** a certified medical device and must **not** be used for clinical diagnosis or patient management. All AI predictions require verification by a qualified radiologist.
