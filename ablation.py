import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from dataset import MRNetDataset
from models import SliceFeatureExtractor

class MeanPoolModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SliceFeatureExtractor(pretrained=True, freeze_early=True)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(0)
        B, S, C, H, W = x.size()
        slice_feats = self.backbone(x.view(B * S, C, H, W)).view(B, S, -1)
        pooled = torch.mean(slice_feats, dim=1)
        return self.classifier(pooled).squeeze(-1)

class AttentionOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SliceFeatureExtractor(pretrained=True, freeze_early=True)
        self.attention = nn.Linear(512, 1)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(0)
        B, S, C, H, W = x.size()
        slice_feats = self.backbone(x.view(B * S, C, H, W)).view(B, S, -1)
        weights = F.softmax(self.attention(slice_feats), dim=1)
        pooled = torch.sum(weights * slice_feats, dim=1)
        return self.classifier(pooled).squeeze(-1)

class ShuffledLSTMModel(nn.Module):
    def __init__(self, base_lstm_model):
        super().__init__()
        self.base_model = base_lstm_model

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(0)
        B, S, C, H, W = x.size()
        # Randomly shuffle slice order along sequence dimension S for each item in batch
        perm = torch.randperm(S)
        x_shuffled = x[:, perm]
        return self.base_model(x_shuffled)

def evaluate_model(model, loader, device):
    model.eval()
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for series, label, _ in loader:
            series = series.to(device)
            logits = model(series)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(label.numpy())
    auc = roc_auc_score(all_targets, all_probs)
    acc = accuracy_score(all_targets, (np.array(all_probs) >= 0.5).astype(int))
    return auc, acc

def main():
    parser = argparse.ArgumentParser(description="Ablation Study for Sequence Aggregation Architectures")
    parser.add_argument('--root_dir', type=str, default=r'C:\Users\USER\OneDrive\Desktop\DEEP LEARNING\project\MRNet-v1.0')
    parser.add_argument('--plane', type=str, default='sagittal')
    parser.add_argument('--task', type=str, default='acl')
    parser.add_argument('--num_slices', type=int, default=12)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    valid_dataset = MRNetDataset(args.root_dir, split='valid', plane=args.plane, task=args.task, num_slices=args.num_slices)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

    results = {}

    # 1. Mean-Pooling Model
    mean_model = MeanPoolModel().to(device)
    auc_mean, acc_mean = evaluate_model(mean_model, valid_loader, device)
    results['Slice Mean-Pooling'] = {'ROC_AUC': float(auc_mean), 'Accuracy': float(acc_mean)}

    # 2. Attention-Only Model
    attn_model = AttentionOnlyModel().to(device)
    auc_attn, acc_attn = evaluate_model(attn_model, valid_loader, device)
    results['Attention-Only (No LSTM)'] = {'ROC_AUC': float(auc_attn), 'Accuracy': float(acc_attn)}

    # Save results
    with open('ablation_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("\nAblation Experiment Results:", flush=True)
    print(json.dumps(results, indent=4), flush=True)

if __name__ == '__main__':
    main()
