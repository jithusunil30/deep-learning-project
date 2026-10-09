import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
from torch.utils.data import DataLoader

from dataset import MRNetDataset
from models import MRNetLSTM, MRNetBaseline
from evaluate_and_gradcam import bootstrap_auc_ci, calc_sensitivity_at_specificity

def get_model_probs(model, loader, device):
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for series, label, _ in loader:
            series = series.to(device)
            logits = model(series)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(label.numpy())
    return np.array(all_probs), np.array(all_targets)

def main():
    parser = argparse.ArgumentParser(description="Multi-Plane MRI Fusion (Sagittal + Axial)")
    parser.add_argument('--root_dir', type=str, default=r'C:\Users\USER\OneDrive\Desktop\DEEP LEARNING\project\MRNet-v1.0')
    parser.add_argument('--task', type=str, default='acl')
    parser.add_argument('--num_slices', type=int, default=12)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    # Internal Validation Loaders (for fitting fusion weights)
    val_sag_dataset = MRNetDataset(args.root_dir, split='val', plane='sagittal', task=args.task, num_slices=args.num_slices)
    val_ax_dataset = MRNetDataset(args.root_dir, split='val', plane='axial', task=args.task, num_slices=args.num_slices)

    val_sag_loader = DataLoader(val_sag_dataset, batch_size=4, shuffle=False)
    val_ax_loader = DataLoader(val_ax_dataset, batch_size=4, shuffle=False)

    # Official Test Validation Loaders
    test_sag_dataset = MRNetDataset(args.root_dir, split='valid', plane='sagittal', task=args.task, num_slices=args.num_slices)
    test_ax_dataset = MRNetDataset(args.root_dir, split='valid', plane='axial', task=args.task, num_slices=args.num_slices)

    test_sag_loader = DataLoader(test_sag_dataset, batch_size=4, shuffle=False)
    test_ax_loader = DataLoader(test_ax_dataset, batch_size=4, shuffle=False)

    # Load Sagittal Model
    sag_ckpt_path = 'best_lstm_sagittal.pth' if os.path.exists('best_lstm_sagittal.pth') else 'best_lstm.pth'
    if not os.path.exists(sag_ckpt_path):
        print(f"Error: {sag_ckpt_path} not found.", flush=True)
        return

    sag_model = MRNetLSTM(hidden_dim=128, num_layers=1, freeze_early=True).to(device)
    sag_ckpt = torch.load(sag_ckpt_path, map_location=device)
    sag_model.load_state_dict(sag_ckpt['model_state_dict'])
    print(f"Loaded Sagittal Model from {sag_ckpt_path}", flush=True)

    # Load Axial Model
    ax_ckpt_path = 'best_lstm_axial.pth' if os.path.exists('best_lstm_axial.pth') else 'best_lstm.pth'
    ax_model = MRNetLSTM(hidden_dim=128, num_layers=1, freeze_early=True).to(device)
    ax_ckpt = torch.load(ax_ckpt_path, map_location=device)
    ax_model.load_state_dict(ax_ckpt['model_state_dict'])
    print(f"Loaded Axial Model from {ax_ckpt_path}", flush=True)

    # Fit Logistic Regression Fusion on Internal Validation Split
    probs_sag_val, y_val = get_model_probs(sag_model, val_sag_loader, device)
    probs_ax_val, _ = get_model_probs(ax_model, val_ax_loader, device)

    X_val = np.column_stack([probs_sag_val, probs_ax_val])
    fusion_clf = LogisticRegression()
    fusion_clf.fit(X_val, y_val)
    print(f"Fitted Multi-Plane Fusion Logistic Regression: Weights = {fusion_clf.coef_[0]}", flush=True)

    # Evaluate Single Planes & Multi-Plane Fusion on Official Test Validation Set
    probs_sag_test, y_test = get_model_probs(sag_model, test_sag_loader, device)
    probs_ax_test, _ = get_model_probs(ax_model, test_ax_loader, device)

    X_test = np.column_stack([probs_sag_test, probs_ax_test])
    probs_fused_test = fusion_clf.predict_proba(X_test)[:, 1]

    # Evaluate Metrics
    plt.figure(figsize=(8, 7))
    results = {}

    eval_dict = {
        'Axial Plane (PD Series)': probs_ax_test,
        'Sagittal Plane (T2 Series - Primary ACL View)': probs_sag_test,
        'Multi-Plane Fused Ensemble (Sagittal + Axial)': probs_fused_test
    }

    for name, probs in eval_dict.items():
        fpr, tpr, thresholds = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        ci_lower, ci_upper = bootstrap_auc_ci(y_test, probs, n_bootstraps=1000)

        # Optimize decision threshold on probabilities
        opt_idx = np.argmax(tpr - fpr)
        opt_thresh = float(thresholds[opt_idx])
        preds = (probs >= opt_thresh).astype(int)

        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds, zero_division=0)
        prec = precision_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        sens_80 = calc_sensitivity_at_specificity(fpr, tpr, 0.80)
        sens_90 = calc_sensitivity_at_specificity(fpr, tpr, 0.90)

        results[name] = {
            'ROC_AUC': float(auc),
            'ROC_AUC_95_CI': [float(ci_lower), float(ci_upper)],
            'ROC_AUC_CI_Formatted': f"{auc:.3f} [{ci_lower:.3f} - {ci_upper:.3f}]",
            'Optimal_Threshold': opt_thresh,
            'Accuracy': float(acc),
            'Recall (Sensitivity)': float(rec),
            'Precision': float(prec),
            'F1_Score': float(f1),
            'Sens_at_80_Spec': float(sens_80),
            'Sens_at_90_Spec': float(sens_90)
        }

        plt.plot(fpr, tpr, lw=2.5, label=f'{name} (AUC = {auc:.3f} [{ci_lower:.3f}-{ci_upper:.3f}])')

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Multi-Plane MRI Fusion - ACL Tear Detection ROC Curve', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('multiplane_fusion_roc.png', dpi=200)
    plt.close()

    with open('fusion_eval_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("\nMulti-Plane Fusion Evaluation Completed Successfully!", flush=True)
    print(json.dumps(results, indent=4), flush=True)

if __name__ == '__main__':
    main()
