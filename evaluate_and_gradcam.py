import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
import cv2

from dataset import MRNetDataset
from models import MRNetBaseline, MRNetLSTM, GradCAM

def calc_sensitivity_at_specificity(fps, tps, spec_target=0.80):
    # Specificity = 1 - FPR, so FPR = 1 - Specificity
    fpr_target = 1.0 - spec_target
    valid_idxs = np.where(fps <= fpr_target)[0]
    if len(valid_idxs) > 0:
        idx = valid_idxs[-1]
        return tps[idx]
    return 0.0

def bootstrap_auc_ci(y_true, y_pred, n_bootstraps=1000, ci=95, seed=42):
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    lower = np.percentile(sorted_scores, (100 - ci) / 2)
    upper = np.percentile(sorted_scores, 100 - (100 - ci) / 2)
    return lower, upper

def generate_and_save_gradcam(model, dataset, case_id, output_path, device):
    # Find dataset index for case_id
    idx = None
    for i in range(len(dataset)):
        if str(dataset.labels_df.iloc[i]['id']).zfill(4) == case_id:
            idx = i
            break

    if idx is None:
        idx = 0
        case_id = str(dataset.labels_df.iloc[0]['id']).zfill(4)

    series_tensor, label, _ = dataset[idx]  # (S, 3, H, W)
    input_tensor = series_tensor.unsqueeze(0).to(device)  # (1, S, 3, H, W)

    # Initialize GradCAM
    gradcam = GradCAM(model)
    heatmap, slice_idx = gradcam.generate(input_tensor)

    # Load raw npy slice for visualization
    file_path = os.path.join(dataset.series_dir, f"{case_id}.npy")
    array = np.load(file_path)
    total_slices = array.shape[0]

    # Map slice index
    if total_slices > dataset.num_slices:
        indices = np.linspace(0, total_slices - 1, dataset.num_slices, dtype=int)
        raw_slice_idx = indices[slice_idx]
    else:
        raw_slice_idx = min(slice_idx, total_slices - 1)

    raw_slice = array[raw_slice_idx]  # (H, W) 0-255

    # Convert raw slice to BGR image
    raw_img = cv2.cvtColor(raw_slice, cv2.COLOR_GRAY2BGR)

    # Apply JET color map to GradCAM heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Overlay heatmap onto MRI slice
    overlay = cv2.addWeighted(raw_img, 0.6, heatmap_colored, 0.4, 0)

    # Save figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(raw_slice, cmap='gray')
    axes[0].set_title(f"Original MRI (Case {case_id}, Slice {raw_slice_idx})")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Grad-CAM Activation Heatmap")
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Lesion / Edema Highlight (Ground Truth: {int(label.item())})")
    axes[2].axis('off')

    plt.suptitle(f"MRNet Tear & Edema Detection - Grad-CAM Explainability Map", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Grad-CAM overlay saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Baseline vs LSTM Models & Generate Grad-CAM")
    parser.add_argument('--root_dir', type=str, default=r'C:\Users\USER\OneDrive\Desktop\DEEP LEARNING\project\MRNet-v1.0')
    parser.add_argument('--task', type=str, default='acl')
    parser.add_argument('--plane', type=str, default='axial')
    parser.add_argument('--num_slices', type=int, default=16)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    valid_dataset = MRNetDataset(args.root_dir, split='valid', plane=args.plane, task=args.task, num_slices=args.num_slices)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=0)

    models_dict = {}

    # Load Baseline
    baseline_paths = [f'best_baseline_{args.plane}.pth', 'best_baseline.pth']
    baseline_path = next((p for p in baseline_paths if os.path.exists(p)), None)
    if baseline_path:
        b_model = MRNetBaseline(freeze_early=True).to(device)
        b_ckpt = torch.load(baseline_path, map_location=device)
        b_model.load_state_dict(b_ckpt['model_state_dict'])
        models_dict['CNN Max-Pooling (Baseline)'] = b_model
        print(f"Loaded baseline model from {baseline_path} successfully.", flush=True)
    else:
        print("Warning: baseline checkpoint not found!", flush=True)

    # Load LSTM
    lstm_paths = [f'best_lstm_{args.plane}.pth', 'best_lstm.pth']
    lstm_path = next((p for p in lstm_paths if os.path.exists(p)), None)
    if lstm_path:
        l_model = MRNetLSTM(hidden_dim=128, num_layers=1, freeze_early=True).to(device)
        l_ckpt = torch.load(lstm_path, map_location=device)
        l_model.load_state_dict(l_ckpt['model_state_dict'])
        models_dict['CNN + BiLSTM (Proposed)'] = l_model
        print(f"Loaded LSTM model from {lstm_path} successfully.", flush=True)
    else:
        print("Warning: LSTM checkpoint not found!", flush=True)

    if not models_dict:
        print("No trained models found! Please run train.py first.", flush=True)
        return

    plt.figure(figsize=(8, 7))
    metrics_summary = {}

    for name, model in models_dict.items():
        model.eval()
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for series, label, _ in valid_loader:
                series = series.to(device)
                label = label.to(device)

                logits = model(series)
                probs = torch.sigmoid(logits)

                all_targets.extend(label.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        preds = (all_probs >= 0.5).astype(int)

        fpr, tpr, _ = roc_curve(all_targets, all_probs)
        auc = roc_auc_score(all_targets, all_probs)
        ci_lower, ci_upper = bootstrap_auc_ci(all_targets, all_probs, n_bootstraps=1000)
        
        acc = accuracy_score(all_targets, preds)
        rec = recall_score(all_targets, preds, zero_division=0)
        prec = precision_score(all_targets, preds, zero_division=0)
        f1 = f1_score(all_targets, preds, zero_division=0)

        sens_at_80spec = calc_sensitivity_at_specificity(fpr, tpr, spec_target=0.80)
        sens_at_90spec = calc_sensitivity_at_specificity(fpr, tpr, spec_target=0.90)

        metrics_summary[name] = {
            'ROC_AUC': float(auc),
            'ROC_AUC_95_CI': [float(ci_lower), float(ci_upper)],
            'ROC_AUC_CI_Formatted': f"{auc:.3f} [{ci_lower:.3f} - {ci_upper:.3f}]",
            'Accuracy': float(acc),
            'Recall (Sensitivity)': float(rec),
            'Precision': float(prec),
            'F1_Score': float(f1),
            'Sens_at_80_Spec': float(sens_at_80spec),
            'Sens_at_90_Spec': float(sens_at_90spec)
        }

        plt.plot(fpr, tpr, lw=2.5, label=f'{name} (AUC = {auc:.3f} [{ci_lower:.3f}-{ci_upper:.3f}])')

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title(f'ROC Curve Comparison - Knee MRI Tear Detection ({args.task.upper()})', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_comparison.png', dpi=200)
    plt.close()
    print("ROC curve comparison plot saved to: roc_comparison.png")

    # Save metrics summary json
    with open('eval_results.json', 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print("Evaluation metrics saved to: eval_results.json")
    print(json.dumps(metrics_summary, indent=4))

    # Generate Grad-CAM sample using the best model
    best_model_key = 'CNN + BiLSTM (Proposed)' if 'CNN + BiLSTM (Proposed)' in models_dict else list(models_dict.keys())[0]
    best_model = models_dict[best_model_key]

    # Find a positive validation case
    pos_cases = valid_dataset.labels_df[valid_dataset.labels_df['label'] == 1]
    sample_case_id = str(pos_cases.iloc[0]['id']).zfill(4) if len(pos_cases) > 0 else '1130'

    generate_and_save_gradcam(best_model, valid_dataset, sample_case_id, 'gradcam_sample.png', device)

if __name__ == '__main__':
    main()
