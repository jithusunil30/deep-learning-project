import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2

class SliceFeatureExtractor(nn.Module):
    """
    ResNet18 Feature Extractor for 2D MRI slices.
    Freezes early layers (conv1, bn1, layer1) to prevent overfitting on small medical datasets.
    """
    def __init__(self, pretrained=True, freeze_early=True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        
        if freeze_early:
            for name, param in resnet.named_parameters():
                if any(layer in name for layer in ['conv1', 'bn1', 'layer1']):
                    param.requires_grad = False
                    
        # Feature extractor up to avgpool
        self.conv_body = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        self.avgpool = resnet.avgpool
        self.feature_dim = 512

    def forward(self, x):
        # x shape: (batch_size, 3, H, W)
        features = self.conv_body(x)             # (batch_size, 512, H_out, W_out)
        pooled = self.avgpool(features)            # (batch_size, 512, 1, 1)
        flattened = torch.flatten(pooled, 1)       # (batch_size, 512)
        return flattened

class MRNetBaseline(nn.Module):
    """
    Baseline MRNet Architecture:
    Extracts features per slice using ResNet18, then performs Max-Pooling across the slice sequence.
    """
    def __init__(self, freeze_early=True):
        super().__init__()
        self.backbone = SliceFeatureExtractor(pretrained=True, freeze_early=freeze_early)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        # x shape: (batch_size, S, 3, H, W) or (S, 3, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(0)  # Add batch dimension -> (1, S, 3, H, W)
            
        B, S, C, H, W = x.size()
        x_reshaped = x.view(B * S, C, H, W)
        
        # Feature extraction
        slice_feats = self.backbone(x_reshaped)   # (B*S, 512)
        slice_feats = slice_feats.view(B, S, -1)   # (B, S, 512)
        
        # Max-pooling across sequence dimension S
        seq_pooled, _ = torch.max(slice_feats, dim=1)  # (B, 512)
        
        # Output logit
        logit = self.classifier(seq_pooled)       # (B, 1)
        return logit.squeeze(-1)

class MRNetLSTM(nn.Module):
    """
    CNN + LSTM Architecture:
    Extracts features per slice using ResNet18, then processes slice sequence with a BiLSTM.
    Captures temporal/spatial continuity across sequential MRI slices.
    """
    def __init__(self, hidden_dim=128, num_layers=1, freeze_early=True):
        super().__init__()
        self.backbone = SliceFeatureExtractor(pretrained=True, freeze_early=freeze_early)
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: (B, S, 3, H, W) or (S, 3, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(0)
            
        B, S, C, H, W = x.size()
        x_reshaped = x.view(B * S, C, H, W)
        
        slice_feats = self.backbone(x_reshaped)    # (B*S, 512)
        slice_feats = slice_feats.view(B, S, -1)    # (B, S, 512)
        
        # LSTM sequence processing
        lstm_out, _ = self.lstm(slice_feats)       # (B, S, 256)
        
        # Attention-weighted sequence pooling
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)  # (B, S, 1)
        seq_pooled = torch.sum(attn_weights * lstm_out, dim=1)      # (B, 256)
        
        logit = self.classifier(seq_pooled)        # (B, 1)
        return logit.squeeze(-1)

class GradCAM:
    """
    Grad-CAM class to generate explainability heatmaps on the final convolutional layer of ResNet18.
    """
    def __init__(self, model, target_layer_name='backbone.conv_body.7'):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        
        # Hook layer4 (index 7 in conv_body)
        target_layer = self.model.backbone.conv_body[7]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_slice_idx=None):
        """
        input_tensor: shape (1, S, 3, H, W)
        Returns heatmap of shape (H, W) for the most active or specified slice.
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Backward pass on the output score
        score = output[0] if output.dim() > 0 else output
        score.backward(retain_graph=True)
        
        # self.activations: (1*S, 512, H_feat, W_feat)
        # self.gradients: (1*S, 512, H_feat, W_feat)
        grads = self.gradients
        acts = self.activations
        
        # Pick slice index if not specified (slice with maximum activation)
        if target_slice_idx is None:
            slice_norms = acts.view(acts.size(0), -1).norm(dim=1)
            target_slice_idx = torch.argmax(slice_norms).item()
            
        target_grad = grads[target_slice_idx]  # (512, H_feat, W_feat)
        target_act = acts[target_slice_idx]    # (512, H_feat, W_feat)
        
        # Pool gradients across H, W
        weights = torch.mean(target_grad, dim=(1, 2))  # (512,)
        
        # Compute weighted sum of activations
        cam = torch.zeros(target_act.shape[1:], dtype=torch.float32, device=target_act.device)
        for i, w in enumerate(weights):
            cam += w * target_act[i]
            
        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()
        
        # Normalize between 0 and 1
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
            
        # Resize to original slice dimension H, W
        H, W = input_tensor.shape[3], input_tensor.shape[4]
        cam_resized = cv2.resize(cam, (W, H))
        return cam_resized, target_slice_idx
