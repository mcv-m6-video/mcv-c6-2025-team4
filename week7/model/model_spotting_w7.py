"""
File containing the main model.
"""

# Standard imports
import torch
from torch import nn
import timm
import numpy as np
import torchvision
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import math
import torch.nn.functional as F

# Local imports
from model.modules import BaseRGBModel, FCLayers, step
from fvcore.nn import FlopCountAnalysis, parameter_count_table

class Impl(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self._feature_arch = args.feature_arch

        if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
            features = timm.create_model({
                'rny002': 'regnety_002',
                'rny004': 'regnety_004',
                'rny008': 'regnety_008',
            }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
            feat_dim = features.head.fc.in_features
            features.head.fc = nn.Identity()
            self._d = feat_dim
        else:
            raise NotImplementedError(args._feature_arch)

        self._features = features
        self._fc = FCLayers(self._d, args.num_classes + 1)

        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
            T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.GaussianBlur(5)], p=0.25),
            T.RandomHorizontalFlip(),
        ])

        self.standarization = T.Compose([
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def forward(self, x):
        x = self.normalize(x)
        batch_size, clip_len, channels, height, width = x.shape

        if self.training:
            x = self.augment(x)

        x = self.standarize(x)

        im_feat = self._features(
            x.view(-1, channels, height, width)
        ).reshape(batch_size, clip_len, self._d)

        im_feat = self._fc(im_feat)
        return im_feat

    def normalize(self, x):
        return x / 255.

    def augment(self, x):
        for i in range(x.shape[0]):
            x[i] = self.augmentation(x[i])
        return x

    def standarize(self, x):
        for i in range(x.shape[0]):
            x[i] = self.standarization(x[i])
        return x

    def print_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        try:
            dummy_input = torch.randn(4, 50, 3, 224, 224).to(next(self.parameters()).device)
            self.eval()
            flops = FlopCountAnalysis(self, dummy_input)
            print(f"FLOPs: {flops.total():,}")
            print(parameter_count_table(self))
        except Exception as e:
            print(f"Could not calculate FLOPs: {e}")


class TemporalAnchorSpotting(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self._feature_arch = args.feature_arch
        self.num_classes = args.num_classes

        if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
            features = timm.create_model({
                'rny002': 'regnety_002',
                'rny004': 'regnety_004',
                'rny008': 'regnety_008',
            }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
            feat_dim = features.head.fc.in_features
            features.head.fc = nn.Identity()
            self._d = feat_dim
        else:
            raise NotImplementedError(args._feature_arch)

        self._features = features

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self._d, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(512, self.num_classes + 1, kernel_size=1)
        )

        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
            T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.GaussianBlur(5)], p=0.25),
            T.RandomHorizontalFlip(),
        ])
        self.standarization = T.Compose([
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def forward(self, x):
        x = self.normalize(x)
        B, T, C, H, W = x.shape

        if self.training:
            x = self.augment(x)

        x = self.standarize(x)

        x = x.view(-1, C, H, W)
        feats = self._features(x)
        feats = feats.view(B, T, self._d).permute(0, 2, 1)

        out = self.temporal_conv(feats)
        out = out.permute(0, 2, 1)  # (B, T, num_classes+1)
        return out

    def normalize(self, x):
        return x / 255.

    def augment(self, x):
        for i in range(x.shape[0]):
            x[i] = self.augmentation(x[i])
        return x

    def standarize(self, x):
        for i in range(x.shape[0]):
            x[i] = self.standarization(x[i])
        return x

    def print_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        try:
            dummy_input = torch.randn(4, 50, 3, 224, 224).to(next(self.parameters()).device)
            self.eval()
            flops = FlopCountAnalysis(self, dummy_input)
            print(f"FLOPs: {flops.total():,}")
            print(parameter_count_table(self))
        except Exception as e:
            print(f"Could not calculate FLOPs: {e}")
            
# Positional Encoding para el orden temporal
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))  # [D/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Modelo Híbrido con Transformer y LSTM
class HybridTransformerDownscale(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self._feature_arch = args.feature_arch

        # Backbone (RegNetY)
        if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
            features = timm.create_model({
                'rny002': 'regnety_002',
                'rny004': 'regnety_004',
                'rny008': 'regnety_008',
            }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
            feat_dim = features.head.fc.in_features
            features.head.fc = nn.Identity()
            self._d = feat_dim
        else:
            raise NotImplementedError(args._feature_arch)

        self._features = features

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(self._d)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._d, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # BiLSTM
        self.temporal_rnn = nn.LSTM(
            input_size=self._d, hidden_size=512, num_layers=1,
            batch_first=True, bidirectional=True
        )

        # Downscale temporal
        self.temporal_downscale = nn.Conv1d(self._d, self._d, kernel_size=2, stride=2)

        # Offset head (opcional)
        self.temporal_offset_head = nn.Conv1d(self._d, 1, kernel_size=1)

        # Clasificador
        self._fc = FCLayers(512 * 2, args.num_classes + 1)

        # Augmentaciones
        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
            T.RandomHorizontalFlip(),
        ])
        self.standarization = T.Normalize(mean=(0.485, 0.456, 0.406),
                                          std=(0.229, 0.224, 0.225))

    def forward(self, x):
        x = x / 255.
        B, T, C, H, W = x.shape

        if self.training:
            for i in range(B):
                x[i] = self.augmentation(x[i])
        for i in range(B):
            x[i] = self.standarization(x[i])

        # Features
        x = x.view(-1, C, H, W)
        feat = self._features(x).view(B, T, self._d)

        feat = self.pos_encoder(feat)
        encoded = self.temporal_encoder(feat)

        # Downscale temporal
        downscaled = self.temporal_downscale(encoded.permute(0, 2, 1)).permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.temporal_rnn(downscaled)

        # Offset prediction
        temporal_offsets = self.temporal_offset_head(downscaled.permute(0, 2, 1))

        # Clasificación
        out = self._fc(lstm_out)
        return out

    def print_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        try:
            dummy_input = torch.randn(4, 50, 3, 224, 224).to(next(self.parameters()).device)
            self.eval()
            flops = FlopCountAnalysis(self, dummy_input)
            print(f"FLOPs: {flops.total():,}")
            print(parameter_count_table(self))
        except Exception as e:
            print(f"Could not calculate FLOPs: {e}")

class HybridTransformerDownscale2(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self._feature_arch = args.feature_arch

        # Backbone (RegNetY)
        if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
            features = timm.create_model({
                'rny002': 'regnety_002',
                'rny004': 'regnety_004',
                'rny008': 'regnety_008',
            }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
            feat_dim = features.head.fc.in_features
            features.head.fc = nn.Identity()
            self._d = feat_dim
        else:
            raise NotImplementedError(args._feature_arch)

        self._features = features

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(self._d)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._d, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Downscaling temporal (antes del LSTM)
        self.downscale_pre_lstm = nn.Conv1d(self._d, self._d, kernel_size=2, stride=2)

        # LSTM
        self.temporal_rnn = nn.LSTM(
            input_size=self._d, hidden_size=512, num_layers=1,
            batch_first=True, bidirectional=True
        )

        # Downscaling (después del LSTM)
        self.downscale_post_lstm = nn.Conv1d(512 * 2, 512 * 2, kernel_size=2, stride=2)

        # Clasificador final
        self._fc = FCLayers(512 * 2, args.num_classes + 1)

        # Augmentations
        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
            T.RandomHorizontalFlip(),
        ])
        self.standarization = T.Normalize(mean=(0.485, 0.456, 0.406),
                                          std=(0.229, 0.224, 0.225))

    def forward(self, x):
        x = x / 255.
        B, T, C, H, W = x.shape

        if self.training:
            for i in range(B):
                x[i] = self.augmentation(x[i])
        for i in range(B):
            x[i] = self.standarization(x[i])

        # Feature extraction
        x = x.view(-1, C, H, W)
        feat = self._features(x).view(B, T, self._d)
        feat = self.pos_encoder(feat)
        feat = self.temporal_encoder(feat)

        # Downscaling before LSTM
        feat = self.downscale_pre_lstm(feat.permute(0, 2, 1)).permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.temporal_rnn(feat)

        # Downscaling after LSTM
        lstm_down = self.downscale_post_lstm(lstm_out.permute(0, 2, 1)).permute(0, 2, 1)

        # Final classification
        out = self._fc(lstm_down)
        return out
        
    def print_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        try:
            dummy_input = torch.randn(4, 50, 3, 224, 224).to(next(self.parameters()).device)
            self.eval()
            flops = FlopCountAnalysis(self, dummy_input)
            print(f"FLOPs: {flops.total():,}")
            print(parameter_count_table(self))
        except Exception as e:
            print(f"Could not calculate FLOPs: {e}")

class TemporalPooling(nn.Module):
    def __init__(self, in_dim, num_clusters=8):
        super().__init__()
        self.num_clusters = num_clusters
        self.in_dim = in_dim

        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, in_dim))
        self.attention = nn.Linear(in_dim, num_clusters)

    def forward(self, x):
        # x: [B, T, D]
        soft_assign = F.softmax(self.attention(x), dim=-1)  # [B, T, K]
        residuals = x.unsqueeze(2) - self.cluster_centers  # [B, T, K, D]
        residuals = residuals * soft_assign.unsqueeze(-1)  # [B, T, K, D]
        pooled = residuals.sum(dim=1)  # [B, K, D]
        pooled = F.normalize(pooled.view(x.shape[0], -1), p=2, dim=1)  # [B, K*D]
        return pooled


class HybridTransformerLSTMModel(nn.Module):
    def __init__(self, args=None):  # Ajustado __init__
        super().__init__()
        self._feature_arch = args.feature_arch

        # Backbone (RegNetY)
        if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
            features = timm.create_model({
                'rny002': 'regnety_002',
                'rny004': 'regnety_004',
                'rny008': 'regnety_008',
            }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
            feat_dim = features.head.fc.in_features
            features.head.fc = nn.Identity()
            self._d = feat_dim
        else:
            raise NotImplementedError(args._feature_arch)

        self._features = features

        # Positional Encoding for temporal order
        self.pos_encoder = PositionalEncoding(self._d)

        # Transformer encoder (contexto global)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._d, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # LSTM para modelar las relaciones temporales más finas
        self.lstm = nn.LSTM(self._d, self._d, batch_first=True)

        # Temporal Pooling layer
        self.temporal_pooling = TemporalPooling(in_dim=self._d, num_clusters=8)

        # Clasificador final
        self._fc = FCLayers(8 * self._d, args.num_classes + 1)

        # Augmentations
        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
            T.RandomHorizontalFlip(),
        ])
        self.standarization = T.Normalize(mean=(0.485, 0.456, 0.406),
                                          std=(0.229, 0.224, 0.225))

    def forward(self, x):
        x = x / 255.
        B, T, C, H, W = x.shape
    
        # Aplicar aumentaciones solo en el entrenamiento
        if self.training:
            for i in range(B):
                x[i] = self.augmentation(x[i])
        
        # Normalización estándar
        for i in range(B):
            x[i] = self.standarization(x[i])
    
        # Pasar la entrada por el modelo de características
        x = x.view(-1, C, H, W)
        feat = self._features(x).view(B, T, self._d)
    
        # Añadir codificación posicional antes del Transformer
        feat = self.pos_encoder(feat)
    
        # Pasar por el Transformer encoder
        encoded = self.temporal_encoder(feat)

        # Aplicar LSTM después del Transformer
        lstm_out, _ = self.lstm(encoded)

        # Asegurarnos de que el "pooling" no reduzca la dimensión temporal
        pooled = self.temporal_pooling(lstm_out)  # [B, K*D] (asumiendo que K es 8 en tu caso)
    
        # Si después del pooling no tenemos la dimensión temporal, agregamos T_pred artificialmente
        if len(pooled.shape) == 2:
            pooled = pooled.unsqueeze(1)  # Añadimos la dimensión temporal T_pred = 1
    
        # Asegurarnos de que la forma sea [B, T_pred, C]
        out = self._fc(pooled)
        
        return out

    def print_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        try:
            dummy_input = torch.randn(4, 50, 3, 224, 224).to(next(self.parameters()).device)
            self.eval()
            flops = FlopCountAnalysis(self, dummy_input)
            print(f"FLOPs: {flops.total():,}")
            print(parameter_count_table(self))
        except Exception as e:
            print(f"Could not calculate FLOPs: {e}")

            
class TemporalDownsampler(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=4):
        super().__init__()
        self.pool = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)

    def forward(self, x):
        # x: [B, T, D]
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.pool(x)       # [B, D, T']
        x = x.transpose(1, 2)  # [B, T', D]
        return x
            
# TCN Block (Temporal Convolutional Network)
class TemporalConvolutionalNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, num_layers=5, stride=1, dilation_growth=1):
        super(TemporalConvolutionalNetwork, self).__init__()

        layers = []
        dilation = 1  # Inicializamos la dilatación en 1
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else output_dim
            layers.append(
                nn.Conv1d(in_channels, output_dim, kernel_size=kernel_size, stride=stride, 
                          padding=(kernel_size // 2) * dilation, dilation=dilation)
            )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(output_dim))

            # Incrementamos la dilatación en función del parámetro de dilatación
            dilation *= dilation_growth  # El crecimiento de la dilatación

        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, T, D] -> [B, D, T] (para la convolución 1D)
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.tcn(x)        # [B, D', T']
        x = x.transpose(1, 2)  # [B, T', D']
        return x

class TemporalAnchorSpottingTCN(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.num_classes = args.num_classes
        self._feature_arch = args.feature_arch

        # Backbone visual (como en los otros modelos)
        if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
            features = timm.create_model({
                'rny002': 'regnety_002',
                'rny004': 'regnety_004',
                'rny008': 'regnety_008',
            }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
            feat_dim = features.head.fc.in_features
            features.head.fc = nn.Identity()
            self._d = feat_dim
        else:
            raise NotImplementedError(args._feature_arch)

        self._features = features

        # TCN que opera sobre las features visuales (por frame)
        self.tcn = TemporalConvolutionalNetwork(
            input_dim=self._d, output_dim=256, num_layers=5, kernel_size=3, stride=1
        )

        # Cabeza de predicción temporal
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(512, self.num_classes + 1, kernel_size=1)
        )

        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
            T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.GaussianBlur(5)], p=0.25),
            T.RandomHorizontalFlip(),
        ])
        self.standarization = T.Compose([
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def forward(self, x):
        x = self.normalize(x)
        B, T, C, H, W = x.shape

        if self.training:
            x = self.augment(x)

        x = self.standarize(x)

        # Backbone visual frame a frame
        x = x.view(-1, C, H, W)  # [B*T, C, H, W]
        feats = self._features(x).view(B, T, self._d)  # [B, T, D]

        # TCN
        feats = self.tcn(feats)  # [B, T, 256]

        # Clasificador temporal
        out = self.temporal_conv(feats.transpose(1, 2))  # [B, 256, T] -> [B, num_classes+1, T]
        out = out.transpose(1, 2)  # [B, T, num_classes+1]

        return out

    def normalize(self, x):
        return x / 255.

    def augment(self, x):
        for i in range(x.shape[0]):
            x[i] = self.augmentation(x[i])
        return x

    def standarize(self, x):
        for i in range(x.shape[0]):
            x[i] = self.standarization(x[i])
        return x

    def print_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        try:
            dummy_input = torch.randn(4, 50, 3, 224, 224).to(next(self.parameters()).device)
            self.eval()
            flops = FlopCountAnalysis(self, dummy_input)
            print(f"FLOPs: {flops.total():,}")
            print(parameter_count_table(self))
        except Exception as e:
            print(f"Could not calculate FLOPs: {e}")
            
class TCNMultiscaleYOLO(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.num_classes = args.num_classes
        self._feature_arch = args.feature_arch

        # Backbone visual (RegNetY)
        if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
            features = timm.create_model({
                'rny002': 'regnety_002',
                'rny004': 'regnety_004',
                'rny008': 'regnety_008',
            }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
            feat_dim = features.head.fc.in_features
            features.head.fc = nn.Identity()
            self._d = feat_dim
        else:
            raise NotImplementedError(args._feature_arch)

        self._features = features

        # Multi-Scale Temporal Convolutional Network (TCN)
        self.tcn1 = TemporalConvolutionalNetwork(
            input_dim=self._d, output_dim=128, num_layers=3, kernel_size=3, stride=1, dilation_growth=1
        )
        self.tcn2 = TemporalConvolutionalNetwork(
            input_dim=self._d, output_dim=128, num_layers=3, kernel_size=3, stride=1, dilation_growth=2
        )
        self.tcn3 = TemporalConvolutionalNetwork(
            input_dim=self._d, output_dim=128, num_layers=3, kernel_size=3, stride=1, dilation_growth=4
        )

        # Fusion layer to combine multi-scale features
        self.fusion = nn.Conv1d(384, 256, kernel_size=1)  # 128 + 128 + 128 = 384

        # Temporal classification head
        self.temporal_class_head = nn.Conv1d(256, self.num_classes + 1, kernel_size=1)

        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
            T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.GaussianBlur(5)], p=0.25),
            T.RandomHorizontalFlip(),
        ])
        self.standarization = T.Compose([
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def forward(self, x):
        x = self.normalize(x)
        B, T, C, H, W = x.shape

        # Augmentations
        if self.training:
            for i in range(B):
                x[i] = self.augmentation(x[i])
        for i in range(B):
            x[i] = self.standarization(x[i])

        # Backbone visual frame a frame
        x = x.view(-1, C, H, W)  # [B*T, C, H, W]
        feats = self._features(x).view(B, T, self._d)  # [B, T, D]

        # TCNs
        feats_tcn1 = self.tcn1(feats)  # [B, T, 128]
        feats_tcn2 = self.tcn2(feats)  # [B, T, 128]
        feats_tcn3 = self.tcn3(feats)  # [B, T, 128]

        # Fusion of multi-scale features
        fused_feats = torch.cat([feats_tcn1, feats_tcn2, feats_tcn3], dim=-1)  # [B, T, 384]

        # Apply fusion layer to combine features
        fused_feats = self.fusion(fused_feats.transpose(1, 2))  # [B, 256, T]
        fused_feats = fused_feats.transpose(1, 2)  # [B, T, 256]

        # Temporal classification head
        out = self.temporal_class_head(fused_feats.transpose(1, 2))  # [B, num_classes+1, T]
        out = out.transpose(1, 2)  # [B, T, num_classes+1]

        return out

    def normalize(self, x):
        return x / 255.

    def augment(self, x):
        for i in range(x.shape[0]):
            x[i] = self.augmentation(x[i])
        return x

    def standarize(self, x):
        for i in range(x.shape[0]):
            x[i] = self.standarization(x[i])
        return x

    def print_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        try:
            dummy_input = torch.randn(4, 50, 3, 224, 224).to(next(self.parameters()).device)
            self.eval()
            flops = FlopCountAnalysis(self, dummy_input)
            print(f"FLOPs: {flops.total():,}")
            print(parameter_count_table(self))
        except Exception as e:
            print(f"Could not calculate FLOPs: {e}")
    
class TCNTemporalOffset(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.num_classes = args.num_classes
        self._feature_arch = args.feature_arch

        # Visual Backbone
        if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
            features = timm.create_model({
                'rny002': 'regnety_002',
                'rny004': 'regnety_004',
                'rny008': 'regnety_008',
            }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
            feat_dim = features.head.fc.in_features
            features.head.fc = nn.Identity()
            self._d = feat_dim
        else:
            raise NotImplementedError(args._feature_arch)

        self._features = features

        # Multi-Scale TCNs
        self.tcn1 = TemporalConvolutionalNetwork(input_dim=self._d, output_dim=128, num_layers=3, kernel_size=3, stride=1, dilation_growth=1)
        self.tcn2 = TemporalConvolutionalNetwork(input_dim=self._d, output_dim=128, num_layers=3, kernel_size=3, stride=1, dilation_growth=2)
        self.tcn3 = TemporalConvolutionalNetwork(input_dim=self._d, output_dim=128, num_layers=3, kernel_size=3, stride=1, dilation_growth=4)

        # Fusion
        self.fusion = nn.Conv1d(384, 256, kernel_size=1)

        # Output heads
        self.temporal_class_head = nn.Conv1d(256, self.num_classes + 1, kernel_size=1)
        self.temporal_offset_head = nn.Conv1d(256, self.num_classes, kernel_size=1)

        # Augmentations
        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
            T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.GaussianBlur(5)], p=0.25),
            T.RandomHorizontalFlip(),
        ])
        self.standarization = T.Compose([
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def forward(self, x):
        x = x / 255.
        B, T, C, H, W = x.shape

        if self.training:
            for i in range(B):
                x[i] = self.augmentation(x[i])
        for i in range(B):
            x[i] = self.standarization(x[i])

        x = x.view(-1, C, H, W)
        feats = self._features(x).view(B, T, self._d)

        # Multiscale TCNs
        feats1 = self.tcn1(feats)
        feats2 = self.tcn2(feats)
        feats3 = self.tcn3(feats)
        fused = torch.cat([feats1, feats2, feats3], dim=-1)  # [B, T, 384]

        # Fusion
        fused = self.fusion(fused.transpose(1, 2)).transpose(1, 2)  # [B, T, 256]

        # Heads
        cls_logits = self.temporal_class_head(fused.transpose(1, 2)).transpose(1, 2)  # [B, T, num_classes + 1]
        offsets = self.temporal_offset_head(fused.transpose(1, 2)).transpose(1, 2)    # [B, T, num_classes]

        return cls_logits
            
    def print_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        try:
            dummy_input = torch.randn(4, 50, 3, 224, 224).to(next(self.parameters()).device)
            self.eval()
            flops = FlopCountAnalysis(self, dummy_input)
            print(f"FLOPs: {flops.total():,}")
            print(parameter_count_table(self))
        except Exception as e:
            print(f"Could not calculate FLOPs: {e}")

class MultiScaleDownscaleYOLO(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self._feature_arch = args.feature_arch

        # Backbone (RegNetY)
        if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
            features = timm.create_model({
                'rny002': 'regnety_002',
                'rny004': 'regnety_004',
                'rny008': 'regnety_008',
            }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
            feat_dim = features.head.fc.in_features
            features.head.fc = nn.Identity()
            self._d = feat_dim
        else:
            raise NotImplementedError(args._feature_arch)

        self._features = features

        # Downscaling layers
        self.downscale2x = nn.Conv1d(self._d, self._d, kernel_size=2, stride=2)
        self.downscale4x = nn.Conv1d(self._d, self._d, kernel_size=4, stride=4)
        self.downscale8x = nn.Conv1d(self._d, self._d, kernel_size=8, stride=8)

        # YOLO-style prediction heads for each scale
        self.head2x = nn.Conv1d(self._d, args.num_classes + 1, kernel_size=1)
        self.head4x = nn.Conv1d(self._d, args.num_classes + 1, kernel_size=1)
        self.head8x = nn.Conv1d(self._d, args.num_classes + 1, kernel_size=1)

        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
            T.RandomHorizontalFlip(),
        ])
        self.standarization = T.Normalize(mean=(0.485, 0.456, 0.406),
                                          std=(0.229, 0.224, 0.225))

    def forward(self, x):
        x = x / 255.
        B, T, C, H, W = x.shape

        # Data augmentation and standarization
        if self.training:
            for i in range(B):
                x[i] = self.augmentation(x[i])
        for i in range(B):
            x[i] = self.standarization(x[i])

        # Backbone visual feature extraction
        x = x.view(-1, C, H, W)
        feat = self._features(x).view(B, T, self._d).transpose(1, 2)  # [B, D, T]

        # Multi-scale downscaling
        feat2x = self.downscale2x(feat)  # [B, D, T/2]
        feat4x = self.downscale4x(feat)  # [B, D, T/4]
        feat8x = self.downscale8x(feat)  # [B, D, T/8]

        # Prediction heads
        out2x = self.head2x(feat2x).transpose(1, 2)  # [B, T/2, C]
        out4x = self.head4x(feat4x).transpose(1, 2)  # [B, T/4, C]
        out8x = self.head8x(feat8x).transpose(1, 2)  # [B, T/8, C]

        # Combine predictions (for example, concatenate or take an average)
        # Here we concatenate the outputs along the temporal dimension (T)
        out = torch.cat([out2x, out4x, out8x], dim=1)  # [B, T/2 + T/4 + T/8, C]

        return out

    def print_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        try:
            dummy_input = torch.randn(4, 50, 3, 224, 224).to(next(self.parameters()).device)
            self.eval()
            flops = FlopCountAnalysis(self, dummy_input)
            print(f"FLOPs: {flops.total():,}")
            print(parameter_count_table(self))
        except Exception as e:
            print(f"Could not calculate FLOPs: {e}")

class AttentionFusion(nn.Module):
    def __init__(self, input_dims, hidden_dim):
        super().__init__()
        self.query = nn.Linear(input_dims, hidden_dim)
        self.key = nn.Linear(input_dims, hidden_dim)
        self.value = nn.Linear(input_dims, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 256)

    def forward(self, x):  # x: [B, T, C]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (k.shape[-1] ** 0.5), dim=-1)
        out = torch.bmm(attn, v)
        return self.out_proj(out)

class TCNMultiscaleAttention(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.num_classes = args.num_classes
        self._feature_arch = args.feature_arch

        # Backbone visual (RegNetY)
        if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
            features = timm.create_model({
                'rny002': 'regnety_002',
                'rny004': 'regnety_004',
                'rny008': 'regnety_008',
            }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
            feat_dim = features.head.fc.in_features
            features.head.fc = nn.Identity()
            self._d = feat_dim
        else:
            raise NotImplementedError(args._feature_arch)

        self._features = features

        # Proyección a dimensión fija para TCNs
        self.input_proj = nn.Linear(self._d, 128)

        # Multi-Scale Temporal Convolutional Networks
        self.tcn1 = TemporalConvolutionalNetwork(
            input_dim=128, output_dim=128, num_layers=3, kernel_size=3, stride=1, dilation_growth=1
        )
        self.tcn2 = TemporalConvolutionalNetwork(
            input_dim=128, output_dim=128, num_layers=3, kernel_size=3, stride=1, dilation_growth=2
        )
        self.tcn3 = TemporalConvolutionalNetwork(
            input_dim=128, output_dim=128, num_layers=3, kernel_size=3, stride=1, dilation_growth=4
        )

        # Fusion layer to combine multi-scale features
        self.fusion = nn.Conv1d(384, 256, kernel_size=1)  # 128 * 3 = 384

        # Temporal classification head
        self.temporal_class_head = nn.Conv1d(256, self.num_classes + 1, kernel_size=1)

        # Augmentations
        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
            T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.GaussianBlur(5)], p=0.25),
            T.RandomHorizontalFlip(),
        ])
        self.standarization = T.Compose([
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])


    def forward(self, x):
        x = self.normalize(x)
        B, T, C, H, W = x.shape

        # Augmentations
        if self.training:
            for i in range(B):
                x[i] = self.augmentation(x[i])
        for i in range(B):
            x[i] = self.standarization(x[i])

        # Backbone visual
        x = x.view(-1, C, H, W)  # [B*T, C, H, W]
        feats = self._features(x).view(B, T, self._d)  # [B, T, D]

        # Proyección a 128
        feats_proj = self.input_proj(feats)  # [B, T, 128]

        # TCNs
        tcn1 = self.tcn1(feats_proj)
        tcn2 = self.tcn2(feats_proj)
        tcn3 = self.tcn3(feats_proj)

        # Residuales
        tcn1 += feats_proj
        tcn2 += feats_proj
        tcn3 += feats_proj

        # Fusion
        fused_feats = torch.cat([tcn1, tcn2, tcn3], dim=-1)  # [B, T, 384]
        fused_feats = self.fusion(fused_feats.transpose(1, 2))  # [B, 256, T]
        fused_feats = fused_feats.transpose(1, 2)  # [B, T, 256]

        # Output
        out = self.temporal_class_head(fused_feats.transpose(1, 2))  # [B, num_classes+1, T]
        return out.transpose(1, 2)  # [B, T, num_classes+1]

    def normalize(self, x):
        return x / 255.

    def augment(self, x):
        for i in range(x.shape[0]):
            x[i] = self.augmentation(x[i])
        return x

    def standarize(self, x):
        for i in range(x.shape[0]):
            x[i] = self.standarization(x[i])
        return x

    def print_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        try:
            dummy_input = torch.randn(4, 50, 3, 224, 224).to(next(self.parameters()).device)
            self.eval()
            flops = FlopCountAnalysis(self, dummy_input)
            print(f"FLOPs: {flops.total():,}")
            print(parameter_count_table(self))
        except Exception as e:
            print(f"Could not calculate FLOPs: {e}")

class TCNMultiscaleYOLOImproved(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.num_classes = args.num_classes
        self._feature_arch = args.feature_arch

        # Backbone visual
        if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
            features = timm.create_model({
                'rny002': 'regnety_002',
                'rny004': 'regnety_004',
                'rny008': 'regnety_008',
            }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
            feat_dim = features.head.fc.in_features
            features.head.fc = nn.Identity()
            self._d = feat_dim
        else:
            raise NotImplementedError(args._feature_arch)

        self._features = features

        # TCN multi-escala
        self.tcn1 = TemporalConvolutionalNetwork(self._d, 128, num_layers=3, dilation_growth=1)
        self.tcn2 = TemporalConvolutionalNetwork(self._d, 128, num_layers=3, dilation_growth=2)
        self.tcn3 = TemporalConvolutionalNetwork(self._d, 128, num_layers=3, dilation_growth=4)

        # Fusion multiescala
        self.fusion = nn.Conv1d(384, 256, kernel_size=1)

        # Atención sobre features fusionadas
        self.attn = AttentionFusion(256, hidden_dim=128)

        # Cabeza de clasificación temporal
        self.temporal_class_head = nn.Conv1d(256, self.num_classes + 1, kernel_size=1)

        # Head adicional: offset temporal
        self.temporal_offset_head = nn.Conv1d(256, 1, kernel_size=1)

        # Augmentaciones
        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
            T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.GaussianBlur(5)], p=0.25),
            T.RandomHorizontalFlip(),
        ])
        self.standarization = T.Compose([
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def forward(self, x):
        x = self.normalize(x)
        B, T, C, H, W = x.shape

        if self.training:
            for i in range(B):
                x[i] = self.augmentation(x[i])
        for i in range(B):
            x[i] = self.standarization(x[i])

        x = x.view(-1, C, H, W)
        feats = self._features(x).view(B, T, self._d)

        feats_tcn1 = self.tcn1(feats)
        feats_tcn2 = self.tcn2(feats)
        feats_tcn3 = self.tcn3(feats)

        fused_feats = torch.cat([feats_tcn1, feats_tcn2, feats_tcn3], dim=-1)
        fused_feats = self.fusion(fused_feats.transpose(1, 2)).transpose(1, 2)

        # Atención
        fused_feats = self.attn(fused_feats)

        # Clasificación
        class_logits = self.temporal_class_head(fused_feats.transpose(1, 2)).transpose(1, 2)

        # Offset (opcional: para detección más fina)
        offsets = self.temporal_offset_head(fused_feats.transpose(1, 2)).transpose(1, 2)

        return class_logits

    def normalize(self, x):
        return x / 255.

    def augment(self, x):
        for i in range(x.shape[0]):
            x[i] = self.augmentation(x[i])
        return x

    def standarize(self, x):
        for i in range(x.shape[0]):
            x[i] = self.standarization(x[i])
        return x
    
    def print_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        try:
            dummy_input = torch.randn(4, 50, 3, 224, 224).to(next(self.parameters()).device)
            self.eval()
            class_logits, _ = self(dummy_input)
            from fvcore.nn import FlopCountAnalysis, parameter_count_table
            flops = FlopCountAnalysis(self, dummy_input)
            print(f"FLOPs: {flops.total():,}")
            print(parameter_count_table(self))
        except Exception as e:
            print(f"Could not calculate FLOPs: {e}")
            
class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.GroupNorm(8, out_channels)  # O LayerNorm si prefieres
        self.act = nn.SiLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out + identity

class TemporalConvolutionalNetwork2(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, num_layers=5, stride=1, dilation_growth=2):
        super().__init__()
        layers = []
        dilation = 1
        for i in range(num_layers):
            in_ch = input_dim if i == 0 else output_dim
            layers.append(TemporalConvBlock(in_ch, output_dim, kernel_size, dilation))
            dilation *= dilation_growth
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, T, D]
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.tcn(x)
        x = x.transpose(1, 2)  # [B, T, D]
        return x

# ---- Modelo Principal Mejorado ----

class TCNMultiscaleYOLO2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_classes = args.num_classes
        self._feature_arch = args.feature_arch

        if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
            features = timm.create_model({
                'rny002': 'regnety_002',
                'rny004': 'regnety_004',
                'rny008': 'regnety_008',
            }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
            feat_dim = features.head.fc.in_features
            features.head.fc = nn.Identity()
            self._d = feat_dim
        else:
            raise NotImplementedError(args.feature_arch)

        self._features = features

        self.tcn1 = TemporalConvolutionalNetwork2(input_dim=self._d, output_dim=128, num_layers=3, kernel_size=3, dilation_growth=1)
        self.tcn2 = TemporalConvolutionalNetwork2(input_dim=self._d, output_dim=128, num_layers=3, kernel_size=3, dilation_growth=2)
        self.tcn3 = TemporalConvolutionalNetwork2(input_dim=self._d, output_dim=128, num_layers=3, kernel_size=3, dilation_growth=4)

        self.fusion = nn.Sequential(
            nn.Conv1d(384, 256, kernel_size=1),
            nn.GroupNorm(8, 256),
            nn.SiLU()
        )

        self.temporal_class_head = nn.Sequential(
            nn.Conv1d(256, self.num_classes + 1, kernel_size=1)
        )

        # Mejor Augmentation
        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.1, brightness=0.1, contrast=0.1, saturation=0.1)], p=0.5),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.GaussianBlur(5)], p=0.2),
        ])

        self.standardization = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def forward(self, x):
        x = self.normalize(x)
        B, T, C, H, W = x.shape

        if self.training:
            for i in range(B):
                x[i] = self.augmentation(x[i])

        for i in range(B):
            x[i] = self.standardization(x[i])

        x = x.view(-1, C, H, W)  # [B*T, C, H, W]
        feats = self._features(x).view(B, T, self._d)  # [B, T, D]

        feats_tcn1 = self.tcn1(feats)
        feats_tcn2 = self.tcn2(feats)
        feats_tcn3 = self.tcn3(feats)

        fused_feats = torch.cat([feats_tcn1, feats_tcn2, feats_tcn3], dim=-1)  # [B, T, 384]

        fused_feats = self.fusion(fused_feats.transpose(1, 2))  # [B, 256, T]
        fused_feats = fused_feats.transpose(1, 2)  # [B, T, 256]

        out = self.temporal_class_head(fused_feats.transpose(1, 2))  # [B, num_classes+1, T]
        out = out.transpose(1, 2)  # [B, T, num_classes+1]

        return out

    def normalize(self, x):
        return x / 255.

    def print_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        try:
            dummy_input = torch.randn(4, 50, 3, 224, 224).to(next(self.parameters()).device)
            self.eval()
            flops = FlopCountAnalysis(self, dummy_input)
            print(f"FLOPs: {flops.total():,}")
            print(parameter_count_table(self))
        except Exception as e:
            print(f"Could not calculate FLOPs: {e}")
    
class TCNMultiscaleYOLO5(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_classes = args.num_classes
        self._feature_arch = args.feature_arch

        if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
            features = timm.create_model({
                'rny002': 'regnety_002',
                'rny004': 'regnety_004',
                'rny008': 'regnety_008',
            }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
            feat_dim = features.head.fc.in_features
            features.head.fc = nn.Identity()
            self._d = feat_dim
        else:
            raise NotImplementedError(args.feature_arch)

        self._features = features

        self.tcn1 = TemporalConvolutionalNetwork2(input_dim=self._d, output_dim=128, num_layers=3, kernel_size=3, dilation_growth=1)
        self.tcn2 = TemporalConvolutionalNetwork2(input_dim=self._d, output_dim=128, num_layers=3, kernel_size=3, dilation_growth=2)
        self.tcn3 = TemporalConvolutionalNetwork2(input_dim=self._d, output_dim=128, num_layers=3, kernel_size=3, dilation_growth=4)

        self.fusion = nn.Sequential(
            nn.Dropout(0.3),  # <<<<<< Dropout aquí
            nn.Conv1d(384, 256, kernel_size=1),
            nn.GroupNorm(8, 256),
            nn.SiLU()
        )

        self.temporal_class_head = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv1d(256, self.num_classes + 1, kernel_size=1)
        )

        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.1, brightness=0.1, contrast=0.1, saturation=0.1)], p=0.5),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.GaussianBlur(5)], p=0.2),
        ])

        self.standardization = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def forward(self, x):
        x = self.normalize(x)
        B, T, C, H, W = x.shape

        if self.training:
            for i in range(B):
                x[i] = self.augmentation(x[i])

            # Temporal Augment: Random Frame Dropping
            if torch.rand(1).item() < 0.3:
                x = x[:, ::2]  # Downsample temporalmente
                T = x.size(1)  # Update T

        for i in range(B):
            x[i] = self.standardization(x[i])

        x = x.view(-1, C, H, W)  # [B*T, C, H, W]
        feats = self._features(x).view(B, T, self._d)  # [B, T, D]

        # MixUp en espacio de features
        if self.training and torch.rand(1).item() < 0.5:
            lam = np.random.beta(0.2, 0.2)
            index = torch.randperm(B).to(feats.device)
            feats = lam * feats + (1 - lam) * feats[index]

        feats_tcn1 = self.tcn1(feats)
        feats_tcn2 = self.tcn2(feats)
        feats_tcn3 = self.tcn3(feats)

        fused_feats = torch.cat([feats_tcn1, feats_tcn2, feats_tcn3], dim=-1)  # [B, T, 384]

        fused_feats = self.fusion(fused_feats.transpose(1, 2))  # [B, 256, T]
        fused_feats = fused_feats.transpose(1, 2)  # [B, T, 256]

        out = self.temporal_class_head(fused_feats.transpose(1, 2))  # [B, num_classes+1, T]
        out = out.transpose(1, 2)  # [B, T, num_classes+1]

        return out

    def normalize(self, x):
        return x / 255.

    def print_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        try:
            dummy_input = torch.randn(4, 50, 3, 224, 224).to(next(self.parameters()).device)
            self.eval()
            flops = FlopCountAnalysis(self, dummy_input)
            print(f"FLOPs: {flops.total():,}")
            print(parameter_count_table(self))
        except Exception as e:
            print(f"Could not calculate FLOPs: {e}")
            
class TCNMultiscaleYOLO4(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_classes = args.num_classes
        self._feature_arch = args.feature_arch

        if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
            features = timm.create_model({
                'rny002': 'regnety_002',
                'rny004': 'regnety_004',
                'rny008': 'regnety_008',
            }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
            feat_dim = features.head.fc.in_features
            features.head.fc = nn.Identity()
            self._d = feat_dim
        else:
            raise NotImplementedError(args.feature_arch)

        self._features = features

        self.tcn1 = TemporalConvolutionalNetwork2(input_dim=self._d, output_dim=128, num_layers=3, kernel_size=3, dilation_growth=1)
        self.tcn2 = TemporalConvolutionalNetwork2(input_dim=self._d, output_dim=128, num_layers=3, kernel_size=3, dilation_growth=2)
        self.tcn3 = TemporalConvolutionalNetwork2(input_dim=self._d, output_dim=128, num_layers=3, kernel_size=3, dilation_growth=4)

        self.fusion = nn.Sequential(
            nn.Dropout(0.3),
            nn.Conv1d(384, 256, kernel_size=1),
            nn.GroupNorm(8, 256),
            nn.SiLU()
        )

        self.temporal_class_head = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv1d(256, self.num_classes + 1, kernel_size=1)
        )

        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.1, brightness=0.1, contrast=0.1, saturation=0.1)], p=0.5),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.GaussianBlur(5)], p=0.2),
        ])

        self.standardization = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def forward(self, x):
        x = self.normalize(x)
        B, T, C, H, W = x.shape

        if self.training:
            for i in range(B):
                x[i] = self.augmentation(x[i])

        for i in range(B):
            x[i] = self.standardization(x[i])

        x = x.view(-1, C, H, W)  # [B*T, C, H, W]
        feats = self._features(x).view(B, T, self._d)  # [B, T, D]

        # --- MixUp en Features ---
        if self.training and torch.rand(1).item() < 0.5:
            lam = np.random.beta(0.2, 0.2)
            index = torch.randperm(B).to(feats.device)
            feats = lam * feats + (1 - lam) * feats[index]

        feats_tcn1 = self.tcn1(feats)
        feats_tcn2 = self.tcn2(feats)
        feats_tcn3 = self.tcn3(feats)

        fused_feats = torch.cat([feats_tcn1, feats_tcn2, feats_tcn3], dim=-1)  # [B, T, 384]
        fused_feats = self.fusion(fused_feats.transpose(1, 2))  # [B, 256, T]
        fused_feats = fused_feats.transpose(1, 2)  # [B, T, 256]

        out = self.temporal_class_head(fused_feats.transpose(1, 2))  # [B, num_classes+1, T]
        out = out.transpose(1, 2)  # [B, T, num_classes+1]

        return out

    def normalize(self, x):
        return x / 255.

    def print_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        try:
            dummy_input = torch.randn(4, 50, 3, 224, 224).to(next(self.parameters()).device)
            self.eval()
            flops = FlopCountAnalysis(self, dummy_input)
            print(f"FLOPs: {flops.total():,}")
            print(parameter_count_table(self))
        except Exception as e:
            print(f"Could not calculate FLOPs: {e}")
            
# Temporal ASPP (Atrous Spatial Pyramid Pooling Adaptado)
class TemporalASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=d*dilations[0], dilation=dilations[0], bias=False),
                nn.GroupNorm(8, out_channels),
                nn.SiLU()
            ) for d in dilations
        ])
        self.project = nn.Sequential(
            nn.Conv1d(len(dilations)*out_channels, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        feats = [branch(x) for branch in self.branches]
        # Encontrar el tamaño temporal mínimo
        min_len = min(f.shape[-1] for f in feats)
        # Cortar todas las salidas al mismo tamaño
        feats = [f[..., :min_len] for f in feats]
        out = torch.cat(feats, dim=1)
        out = self.project(out)
        return out


# Simple Temporal Attention
class TemporalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return attn_out + x

# Modelo Principal Mejorado
class TCNMultiscaleYOLOV3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_classes = args.num_classes
        self._feature_arch = args.feature_arch

        if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
            features = timm.create_model({
                'rny002': 'regnety_002',
                'rny004': 'regnety_004',
                'rny008': 'regnety_008',
            }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
            feat_dim = features.head.fc.in_features
            features.head.fc = nn.Identity()
            self._d = feat_dim
        else:
            raise NotImplementedError(args.feature_arch)

        self._features = features

        self.tcn_backbone = TemporalASPP(in_channels=self._d, out_channels=256)
        self.attention = TemporalSelfAttention(dim=256, num_heads=4)

        self.head = nn.Sequential(
            nn.Conv1d(256, 256, 3, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv1d(256, self.num_classes + 1, kernel_size=1)
        )

        # Mejor Augmentation
        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.1, brightness=0.1, contrast=0.1, saturation=0.1)], p=0.5),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.GaussianBlur(5)], p=0.2),
        ])

        self.standardization = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def forward(self, x):
        x = self.normalize(x)
        B, T, C, H, W = x.shape

        if self.training:
            for i in range(B):
                x[i] = self.augmentation(x[i])

        for i in range(B):
            x[i] = self.standardization(x[i])

        x = x.view(-1, C, H, W)  # [B*T, C, H, W]
        feats = self._features(x).view(B, T, self._d)  # [B, T, D]

        x = feats.transpose(1, 2)  # [B, D, T]
        x = self.tcn_backbone(x)   # [B, 256, T]
        x = x.transpose(1, 2)      # [B, T, 256]
        x = self.attention(x)      # [B, T, 256]

        x = x.transpose(1, 2)      # [B, 256, T]
        out = self.head(x)         # [B, num_classes+1, T]
        out = out.transpose(1, 2)  # [B, T, num_classes+1]

        return out

    def normalize(self, x):
        return x / 255.

    def print_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        try:
            dummy_input = torch.randn(4, 50, 3, 224, 224).to(next(self.parameters()).device)
            self.eval()
            flops = FlopCountAnalysis(self, dummy_input)
            print(f"FLOPs: {flops.total():,}")
            print(parameter_count_table(self))
        except Exception as e:
            print(f"Could not calculate FLOPs: {e}")

class NetVLAD(nn.Module):
    def __init__(self, num_clusters=16, dim=512, normalize_input=True):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.normalize_input = normalize_input

        self.conv = nn.Conv1d(dim, num_clusters, kernel_size=1, bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def forward(self, x):  # x: [B, T, D]
        N, T, D = x.shape

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=2)  # Across descriptor dim

        x_flat = x.permute(0, 2, 1)  # [B, D, T]
        soft_assign = self.conv(x_flat)  # [B, num_clusters, T]
        soft_assign = F.softmax(soft_assign, dim=1)

        x_expand = x.unsqueeze(1).expand(-1, self.num_clusters, -1, -1)  # [B, C, T, D]
        c_expand = self.centroids.unsqueeze(0).unsqueeze(2)  # [1, C, 1, D]
        residual = x_expand - c_expand  # [B, C, T, D]

        residual *= soft_assign.unsqueeze(3)  # [B, C, T, D]
        vlad = residual.sum(dim=2)  # [B, C, D]

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # [B, C * D]
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad  # [B, C * D]

class VLADNetModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_classes = args.num_classes
        self._feature_arch = args.feature_arch

        # Visual backbone
        if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
            features = timm.create_model({
                'rny002': 'regnety_002',
                'rny004': 'regnety_004',
                'rny008': 'regnety_008',
            }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
            feat_dim = features.head.fc.in_features
            features.head.fc = nn.Identity()
            self._d = feat_dim
        else:
            raise NotImplementedError(args._feature_arch)

        self._features = features

        # NetVLAD aggregation
        self.vlad = NetVLAD(num_clusters=16, dim=self._d, normalize_input=True)

        # Classification + offset regression (YOLO-style head)
        self.head = nn.Sequential(
            nn.Linear(16 * self._d, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, (self.num_classes + 1) * 2)  # class + offset per class (incl. background)
        )

        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
            T.RandomHorizontalFlip(),
        ])
        self.standarization = T.Normalize(mean=(0.485, 0.456, 0.406),
                                          std=(0.229, 0.224, 0.225))

    def forward(self, x):
        x = x / 255.
        B, T, C, H, W = x.shape
    
        if self.training:
            for i in range(B):
                x[i] = self.augmentation(x[i])
        for i in range(B):
            x[i] = self.standarization(x[i])
    
        x = x.view(-1, C, H, W)
        feat = self._features(x).view(B, T, self._d)
    
        vlad_out = self.vlad(feat)
    
        head_out = self.head(vlad_out)  # [B, (C+1)*2]
        head_out = head_out.view(B, self.num_classes + 1, 2)  # [B, C+1, 2]
    
        class_scores = head_out[..., 0]  # [B, C+1]
        return class_scores.unsqueeze(1).repeat(1, T, 1)  # [B, T, C+1]

    def print_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        try:
            dummy_input = torch.randn(4, 50, 3, 224, 224).to(next(self.parameters()).device)
            self.eval()
            class_logits, _ = self(dummy_input)
            from fvcore.nn import FlopCountAnalysis, parameter_count_table
            flops = FlopCountAnalysis(self, dummy_input)
            print(f"FLOPs: {flops.total():,}")
            print(parameter_count_table(self))
        except Exception as e:
            print(f"Could not calculate FLOPs: {e}")
            
class TAPPooling(nn.Module):
    def __init__(self, in_dim, num_clusters=8):
        super().__init__()
        self.num_clusters = num_clusters
        self.in_dim = in_dim

        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, in_dim))
        self.attention = nn.Linear(in_dim, num_clusters)

    def forward(self, x):
        # x: [B, T, D]
        soft_assign = F.softmax(self.attention(x), dim=-1)  # [B, T, K]
        residuals = x.unsqueeze(2) - self.cluster_centers  # [B, T, K, D]
        residuals = residuals * soft_assign.unsqueeze(-1)  # [B, T, K, D]
        vlad = residuals.sum(dim=1)  # [B, K, D]
        vlad = F.normalize(vlad.view(x.shape[0], -1), p=2, dim=1)  # [B, K*D]
        return vlad


class TemporalDisplacementHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.offset = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.offset(x).squeeze(-1)  # [B, T]


class TAVLADNetModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Backbone: RegNetY
        feature_arch = getattr(args, 'feature_arch', 'rny008')
        self._feature_arch = feature_arch

        features = timm.create_model({
            'rny002': 'regnety_002',
            'rny004': 'regnety_004',
            'rny008': 'regnety_008',
        }[feature_arch], pretrained=True)
        feat_dim = features.head.fc.in_features
        features.head.fc = nn.Identity()

        self._features = features
        self._feat_dim = feat_dim

        # Pre-procesamiento
        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
            T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.GaussianBlur(5)], p=0.25),
            T.RandomHorizontalFlip(),
        ])
        self.standarization = T.Compose([
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        # Temporal branch
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self._feat_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.tap = TAPPooling(in_dim=256, num_clusters=8)
        self.classifier = nn.Linear(256 * 8, args.num_classes+1)
        self.offset_head = TemporalDisplacementHead(256)

    def forward(self, x):
        x = self.normalize(x)
        batch_size, clip_len, channels, height, width = x.shape

        if self.training:
            x = self.augment(x)
        x = self.standarize(x)

        # Backbone
        x = x.view(-1, channels, height, width)  # [B*T, C, H, W]
        features = self._features(x)  # [B*T, D]
        features = features.view(batch_size, clip_len, self._feat_dim)  # [B, T, D]

        # Temporal conv
        x = features.permute(0, 2, 1)  # [B, D, T]
        x = self.temporal_conv(x)  # [B, 256, T]
        x = x.permute(0, 2, 1)  # [B, T, 256]

        pooled = self.tap(x)  # [B, 256 * K]
        class_logits = self.classifier(pooled)  # [B, num_classes]
        class_logits = class_logits.unsqueeze(1).repeat(1, clip_len, 1)  # [B, T, C]

        offsets = self.offset_head(x)  # [B, T]

        return class_logits, offsets

    def normalize(self, x):
        return x / 255.

    def augment(self, x):
        for i in range(x.shape[0]):
            x[i] = self.augmentation(x[i])
        return x

    def standarize(self, x):
        for i in range(x.shape[0]):
            x[i] = self.standarization(x[i])
        return x

    def print_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        try:
            dummy_input = torch.randn(4, 50, 3, 224, 224).to(next(self.parameters()).device)
            self.eval()
            with torch.no_grad():
                logits, offsets = self(dummy_input)
            from fvcore.nn import FlopCountAnalysis, parameter_count_table
            flops = FlopCountAnalysis(self, dummy_input)
            print(f"FLOPs: {flops.total():,}")
            print(parameter_count_table(self))
        except Exception as e:
            print(f"Could not calculate FLOPs: {e}")
class X3DTransformerModel(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.num_classes = args.num_classes
        self.clip_len = getattr(args, 'clip_len', 50)
        self.use_learnable_pe = getattr(args, 'use_learnable_pe', False)

        if args.feature_arch == "x3d_m":
            print("Using X3D_M")
            self.backbone = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
            self.feature_dim = 192
        else:
            raise NotImplementedError("Only x3d_m is supported in this config")

        # Positional encoding
        if self.use_learnable_pe:
            self.pe = nn.Embedding(self.clip_len, self.feature_dim)
        else:
            self.pe = PositionalEncoding(d_model=self.feature_dim, max_len=self.clip_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Final classifier
        self.head = FCLayers(self.feature_dim, self.num_classes + 1)

        # Augmentaciones
        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
            T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.GaussianBlur(5)], p=0.25),
            T.RandomHorizontalFlip(),
        ])
        self.standarization = T.Compose([
            T.Normalize(mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225))
        ])

    def forward(self, x):
        # Entrada esperada: [B, T, C, H, W]
        if x.ndim != 5:
            raise ValueError(f"Expected input of shape [B, T, C, H, W], but got {x.shape}")

        B, T, C, H, W = x.shape

        x = self.normalize(x)

        if self.training:
            x = self.augment(x)
        x = self.standarize(x)

        # (B,T,C,H,W) → (B,C,T,H,W) para X3D
        x = x.permute(0, 2, 1, 3, 4)

        # Solo hasta bloque 4
        for i, block in enumerate(self.backbone.blocks):
            if i == 5:
                break
            x = block(x)

        # Pooling espacial: [B, C, T’, H’, W’] → [B, C, T’]
        x = F.adaptive_avg_pool3d(x, (x.size(2), 1, 1)).squeeze(-1).squeeze(-1)  # [B, C, T’]
        x = x.permute(0, 2, 1)  # [B, T’, C]

        T_prime = x.size(1)

        # Positional Encoding
        if self.use_learnable_pe:
            positions = torch.arange(T_prime, device=x.device).unsqueeze(0).expand(B, -1)
            x = x + self.pe(positions)
        else:
            x = self.pe(x)

        x = self.transformer(x)  # [B, T’, C]

        # Si T’ != T, interpolamos para que coincidan
        if T_prime != T:
            x = x.permute(0, 2, 1)  # [B, C, T’]
            x = F.interpolate(x, size=T, mode='linear', align_corners=False)
            x = x.permute(0, 2, 1)  # [B, T, C]
        
        return self.head(x)

    def normalize(self, x):
        return x / 255.

    def augment(self, x):
        for i in range(x.shape[0]):
            x[i] = self.augmentation(x[i])
        return x

    def standarize(self, x):
        for i in range(x.shape[0]):
            x[i] = self.standarization(x[i])
        return x

    def print_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        try:
            dummy_input = torch.randn(4, 50, 3, 224, 224).to(next(self.parameters()).device)
            self.eval()
            with torch.no_grad():
                logits = self(dummy_input)
            from fvcore.nn import FlopCountAnalysis, parameter_count_table
            flops = FlopCountAnalysis(self, dummy_input)
            print(f"FLOPs: {flops.total():,}")
            print(parameter_count_table(self))
        except Exception as e:
            print(f"Could not calculate FLOPs: {e}")
            
class PositionalEncoding2(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return self.dropout(x)
    
class X3DTransformerModel2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.feature_dim = 192

        # Branch A: full resolution or regular X3D
        self.backbone_a = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
        self.backbone_a.blocks[-1] = nn.Identity()
        self.pool_a = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pos_encoding_a = PositionalEncoding2(self.feature_dim)
        encoder_layer_a = nn.TransformerEncoderLayer(
            d_model=self.feature_dim, nhead=8, dim_feedforward=2048,
            dropout=0.1, batch_first=True)
        self.transformer_a = nn.TransformerEncoder(encoder_layer_a, num_layers=1)
        self.classifier_a = nn.Linear(self.feature_dim, args.num_classes + 1)

        # Branch B: alternative view (e.g., cropped input, blurred input, or low res)
        self.backbone_b = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
        self.backbone_b.blocks[-1] = nn.Identity()
        self.pool_b = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pos_encoding_b = PositionalEncoding2(self.feature_dim)
        encoder_layer_b = nn.TransformerEncoderLayer(
            d_model=self.feature_dim, nhead=8, dim_feedforward=2048,
            dropout=0.1, batch_first=True)
        self.transformer_b = nn.TransformerEncoder(encoder_layer_b, num_layers=1)
        self.classifier_b = nn.Linear(self.feature_dim, args.num_classes + 1)

        # Fusion layer (late fusion)
        self.fusion_classifier = nn.Linear(self.feature_dim, args.num_classes + 1)

        self.normalize = T.Normalize(mean=[0.45]*3, std=[0.225]*3)

    def forward_branch(self, x, backbone, pool, pos_encoding, transformer):
        B, T, C, H, W = x.shape
        x = x / 255.0
        x = x.view(-1, C, H, W)
        x = torch.stack([self.normalize(xi) for xi in x])
        x = x.view(B, T, C, H, W).transpose(1, 2)  # [B, C, T, H, W]

        features = backbone(x)
        features = pool(features).squeeze(-1).squeeze(-1).transpose(1, 2)  # [B, T, C]
        features = pos_encoding(features)
        features = transformer(features)
        return features

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        if x.ndim == 4:
            x = x.unsqueeze(0)
        x = x.float()

        # Shared input to both heads (can be modified to be different views)
        features_a = self.forward_branch(x, self.backbone_a, self.pool_a, self.pos_encoding_a, self.transformer_a)
        features_b = self.forward_branch(x, self.backbone_b, self.pool_b, self.pos_encoding_b, self.transformer_b)

        # Average feature fusion
        fused = (features_a + features_b) / 2
        logits = self.fusion_classifier(fused)  # [B, T, num_classes+1]

        return logits

    def augment(self, x):
            for i in range(x.shape[0]):
                x[i] = self.augmentation(x[i])
            return x

    def standarize(self, x):
        for i in range(x.shape[0]):
            x[i] = self.standarization(x[i])
        return x
    
    def print_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        try:
            dummy_input = torch.randn(4, 50, 3, 224, 224).to(next(self.parameters()).device)
            self.eval()
            with torch.no_grad():
                logits = self(dummy_input)
            from fvcore.nn import FlopCountAnalysis, parameter_count_table
            flops = FlopCountAnalysis(self, dummy_input)
            print(f"FLOPs: {flops.total():,}")
            print(parameter_count_table(self))
        except Exception as e:
            print(f"Could not calculate FLOPs: {e}")

class Model(BaseRGBModel):
    def __init__(self, args=None):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() and getattr(args, "device", "cpu") == "cuda" else "cpu"

        self._model = HybridTransformerLSTMModel(args)
        self._model.print_stats()
        self._args = args
        self._model.to(self.device)
        self._num_classes = args.num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):
        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        weights = torch.tensor([0.05] + [5.0] * self._num_classes, dtype=torch.float32).to(self.device)

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
#                if batch_idx > 0:
#                    break
                frame = batch['frame'].to(self.device).float()
                label = batch['label'].to(self.device).long()

                with torch.amp.autocast(device_type='cuda'):
                    #pred = self._model(frame)  # [B, T', C]
                    class_logits = self._model(frame)
                    pred = class_logits  # esto es lo que se usa para calcular el loss

                    B, T_pred, C = pred.shape
                    T_label = label.shape[1]

                    # Ajustar labels si el modelo hace downscaling
                    if T_pred != T_label:
                        factor = T_label // T_pred
                        if T_label % T_pred == 0:
                            label = label[:, ::factor]
                        else:
                            # Recortar al mínimo común
                            min_len = min(T_label, T_pred)
                            label = label[:, :min_len]
                            pred = pred[:, :min_len]

                    # Flatten para cross entropy
                    pred = pred.reshape(-1, C)
                    label = label.reshape(-1)

                    loss = F.cross_entropy(pred, label, reduction='mean', weight=weights)

                if optimizer is not None:
                    step(optimizer, scaler, loss, lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)


    def predict(self, seq):
        # if not isinstance(seq, torch.Tensor):
        #     seq = torch.FloatTensor(seq)
        # if len(seq.shape) == 4:  # [T, C, H, W]
        #     seq = seq.unsqueeze(0)  # → [1, T, C, H, W]
        # elif len(seq.shape) == 5:  # Already batched
        #     pass
        # else:
        #     raise ValueError(f"Input must have shape [T, C, H, W] or [B, T, C, H, W], got {seq.shape}")
        # seq = seq.to(self.device).float()
        # self._model.eval()
        # with torch.no_grad():
        #     logits = self._model(seq)  # [B, T, C+1]
        #     probs = torch.softmax(logits, dim=-1)
        # return probs.cpu().numpy()              


        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4: # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                
                pred = self._model(seq)

            # apply sigmoid
            pred = torch.softmax(pred, dim=-1)
            
            return pred.cpu().numpy()
        

    def get_optimizer(self, optim_args):
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=optim_args['lr'],
            weight_decay=1e-6
        )
        scaler = torch.cuda.amp.GradScaler()
        return optimizer, scaler

    def state_dict(self):
        return self._model.state_dict()

    def load(self, state_dict):
        return self._model.load_state_dict(state_dict)

