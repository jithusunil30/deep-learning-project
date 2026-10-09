import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score
import numpy as np

from dataset import MRNetDataset
from models import MRNetBaseline, MRNetLSTM

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for series, label, _ in dataloader:
            series = series.to(device)   # (B, S, 3, H, W)
            label = label.to(device)     # (B,)

            logits = model(series)
            loss = criterion(logits, label)
            
            probs = torch.sigmoid(logits)
            
            total_loss += loss.item() * series.size(0)
            all_targets.extend(label.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    preds = (all_probs >= 0.5).astype(int)

    auc = roc_auc_score(all_targets, all_probs) if len(np.unique(all_targets)) > 1 else 0.5
    acc = accuracy_score(all_targets, preds)
    rec = recall_score(all_targets, preds, zero_division=0)
    prec = precision_score(all_targets, preds, zero_division=0)

    return avg_loss, auc, acc, rec, prec, all_targets, all_probs

def train_model(model_type, train_loader, valid_loader, args, device, pos_weight):
    print(f"\n==========================================")
    print(f"       Training {model_type.upper()} Model")
    print(f"==========================================")

    if model_type == 'baseline':
        model = MRNetBaseline(freeze_early=True).to(device)
    elif model_type == 'lstm':
        model = MRNetLSTM(hidden_dim=128, num_layers=1, freeze_early=True).to(device)
    else:
        raise ValueError("Unknown model type")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_auc = 0.0
    best_checkpoint_path = f"best_{model_type}_{args.plane}.pth"

    history = {'train_loss': [], 'valid_loss': [], 'valid_auc': [], 'valid_acc': []}

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        for batch_idx, (series, label, _) in enumerate(train_loader):
            series = series.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            logits = model(series)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * series.size(0)

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch [{epoch}/{args.epochs}] Batch [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        epoch_train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_auc, val_acc, val_rec, val_prec, _, _ = evaluate(model, valid_loader, criterion, device)
        scheduler.step(val_auc)

        elapsed = time.time() - start_time
        print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.1f}s): Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f} | Val Rec (Sens): {val_rec:.4f}")

        history['train_loss'].append(epoch_train_loss)
        history['valid_loss'].append(val_loss)
        history['valid_auc'].append(val_auc)
        history['valid_acc'].append(val_acc)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_auc': best_auc,
                'args': vars(args)
            }, best_checkpoint_path)
            print(f" -> Saved new best model checkpoint to {best_checkpoint_path} (AUC: {best_auc:.4f})")

    print(f"\n[DONE] {model_type.upper()} Training Completed. Best Validation AUC: {best_auc:.4f}")
    return model, history

def main():
    parser = argparse.ArgumentParser(description="Train MRNet Baseline vs CNN+LSTM Models")
    parser.add_argument('--root_dir', type=str, default=r'C:\Users\USER\OneDrive\Desktop\DEEP LEARNING\project\MRNet-v1.0')
    parser.add_argument('--task', type=str, default='acl', choices=['acl', 'meniscus', 'abnormal'])
    parser.add_argument('--plane', type=str, default='axial', choices=['axial', 'coronal', 'sagittal'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_slices', type=int, default=12)
    parser.add_argument('--max_train_samples', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Datasets and Loaders
    print(f"Loading MRNet Dataset for task='{args.task}', plane='{args.plane}'...")
    train_dataset = MRNetDataset(args.root_dir, split='train', plane=args.plane, task=args.task, num_slices=args.num_slices)
    valid_dataset = MRNetDataset(args.root_dir, split='val', plane=args.plane, task=args.task, num_slices=args.num_slices)

    # Subsample training set if max_train_samples is set for CPU speedup
    if args.max_train_samples and args.max_train_samples < len(train_dataset):
        pos_indices = train_dataset.labels_df[train_dataset.labels_df['label'] == 1].index.tolist()
        neg_indices = train_dataset.labels_df[train_dataset.labels_df['label'] == 0].index.tolist()
        
        n_pos = min(len(pos_indices), args.max_train_samples // 2)
        n_neg = args.max_train_samples - n_pos
        
        subset_indices = pos_indices[:n_pos] + neg_indices[:n_neg]
        train_dataset.labels_df = train_dataset.labels_df.iloc[subset_indices].reset_index(drop=True)
        print(f"Subsampled dataset for fast training: {n_pos} positive, {n_neg} negative (Total: {len(train_dataset)})")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Class weighting for BCE loss
    pos_count = train_dataset.labels_df['label'].sum()
    neg_count = len(train_dataset) - pos_count
    pos_weight = torch.tensor([neg_count / max(1, pos_count)], dtype=torch.float32)
    print(f"Class stats: Positive={pos_count}, Negative={neg_count}, BCE pos_weight={pos_weight.item():.2f}")

    # Train Baseline Model (ResNet18 + Max-Pooling)
    train_model('baseline', train_loader, valid_loader, args, device, pos_weight)

    # Train CNN+LSTM Model (ResNet18 + BiLSTM)
    train_model('lstm', train_loader, valid_loader, args, device, pos_weight)

if __name__ == '__main__':
    main()
