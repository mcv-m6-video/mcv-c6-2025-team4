"""
File containing the main model.
"""

# Standard imports
import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F


# Local imports
from model.modules import BaseRGBModel, FCLayers, step
from fvcore.nn import FlopCountAnalysis, parameter_count_table

# Implementation classes moved outside to be standalone
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

            # Remove final classification layer
            features.head.fc = nn.Identity()
            self._d = feat_dim

        else:
            raise NotImplementedError(args._feature_arch)

        self._features = features

        # MLP for classification
        self._fc = FCLayers(self._d, args.num_classes+1) # +1 for background class (we now perform per-frame classification with softmax, therefore we have the extra background class)

        #Augmentations and crop
        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
            T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.GaussianBlur(5)], p=0.25),
            T.RandomHorizontalFlip(),
        ])

        #Standarization
        self.standarization = T.Compose([
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) #Imagenet mean and std
        ])

    def forward(self, x):
        x = self.normalize(x) #Normalize to 0-1
        batch_size, clip_len, channels, height, width = x.shape #B, T, C, H, W

        if self.training:
            x = self.augment(x) #augmentation per-batch

        x = self.standarize(x) #standarization imagenet stats
                    
        im_feat = self._features(
            x.view(-1, channels, height, width)
        ).reshape(batch_size, clip_len, self._d) #B, T, D

        #MLP
        im_feat = self._fc(im_feat) #B, T, num_classes+1

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
        

class HybridTransformerLSTMModel2(nn.Module):
    def __init__(self, args=None):  # Arreglado __init__
        super().__init__()
        self._feature_arch = args.feature_arch

        # Backbone (RegNetY)
        if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
            features = timm.create_model({
                'rny002': 'regnety_002',
                'rny004': 'regnety_004',
                'rny008': 'regnety_008',
            }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)  # Arreglado _feature_arch
            feat_dim = features.head.fc.in_features
            features.head.fc = nn.Identity()
            self._d = feat_dim
        else:
            raise NotImplementedError(args._feature_arch)

        self._features = features

        # Positional Encoding for temporal order
        self.pos_encoder = PositionalEncoding(self._d)

        # Transformer encoder (global context)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._d, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # BiLSTM (local continuity)
        self.temporal_rnn = nn.LSTM(
            input_size=self._d, hidden_size=512, num_layers=1,
            batch_first=True, bidirectional=True
        )

        # Final classifier
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
    
        # Visual features
        x = x.view(-1, C, H, W)
        feat = self._features(x).view(B, T, self._d)
    
        # Add positional encoding before Transformer
        feat = self.pos_encoder(feat)
    
        # Transformer encoder
        encoded = self.temporal_encoder(feat)
    
        # LSTM
        lstm_out, _ = self.temporal_rnn(encoded)
    
        # Classifier
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

class ImplLSTMs(nn.Module):
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

            # Remove final classification layer
            features.head.fc = nn.Identity()
            self._d = feat_dim

        else:
            raise NotImplementedError(args._feature_arch)

        self._features = features

        # Temporal Modeling Layer (LSTM)
        self.temporal_model = nn.LSTM(input_size=self._d, hidden_size=512, num_layers=2, batch_first=True)
        
        # MLP for classification
        self._fc = FCLayers(512, args.num_classes + 1)  # +1 for background class

        # Augmentations and crop
        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
            T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.GaussianBlur(5)], p=0.25),
            T.RandomHorizontalFlip(),
        ])

        # Standardization
        self.standarization = T.Compose([
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # Imagenet mean and std
        ])

    def forward(self, x):
        x = self.normalize(x)  # Normalize to 0-1
        batch_size, clip_len, channels, height, width = x.shape  # B, T, C, H, W

        if self.training:
            x = self.augment(x)  # augmentation per-batch

        x = self.standarize(x)  # standardization imagenet stats

        im_feat = self._features(x.view(-1, channels, height, width)).reshape(batch_size, clip_len, self._d)  # B, T, D

        # Temporal modeling via LSTM (keeping temporal resolution intact)
        lstm_out, _ = self.temporal_model(im_feat)  # B, T, 512

        # MLP
        output = self._fc(lstm_out)  # B, T, num_classes + 1

        return output

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
        

class TemporalTransformerModel(nn.Module):
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
        
        # Transformer temporal
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._d,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.temporal_model = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Clasificador
        self._fc = FCLayers(self._d, args.num_classes + 1)

        # Augmentaciones y normalización
        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
            T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.GaussianBlur(5)], p=0.25),
            T.RandomHorizontalFlip(),
        ])
        self.standarization = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def forward(self, x):
        x = x / 255.
        B, T, C, H, W = x.shape

        if self.training:
            for i in range(B):
                x[i] = self.augmentation(x[i])
        for i in range(B):
            x[i] = self.standarization(x[i])

        x = self._features(x.view(-1, C, H, W)).view(B, T, self._d)
        x = self.temporal_model(x)
        out = self._fc(x)

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
            
class HybridTransformerLSTMModel(nn.Module):
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

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._d, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Bidirectional LSTM after Transformer
        self.temporal_rnn = nn.LSTM(
            input_size=self._d, hidden_size=512, num_layers=1, batch_first=True, bidirectional=True
        )

        # Clasificador final
        self._fc = FCLayers(512 * 2, args.num_classes + 1)

        # Augmentaciones
        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
            T.RandomHorizontalFlip(),
        ])

        # Normalización
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
        features = self._features(x).view(B, T, self._d)

        encoded = self.temporal_encoder(features)
        lstm_out, _ = self.temporal_rnn(encoded)

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
            
class PyramidTransformerModel(nn.Module):
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

        self.temporal_transformers = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self._d,
                    nhead=4,
                    dim_feedforward=512,
                    dropout=0.1,
                    batch_first=True
                ),
                num_layers=1
            )
            for _ in range(3)
        ])

        self._fc = FCLayers(self._d, args.num_classes + 1)

        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
            T.RandomHorizontalFlip(),
        ])

        self.standarization = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )

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

        outputs = [transformer(feat) for transformer in self.temporal_transformers]
        fused = sum(outputs) / len(outputs)

        out = self._fc(fused)
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
            
class PerceiverTransformerModel(nn.Module):
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

        # Learnable latent queries
        self.latent_queries = nn.Parameter(torch.randn(1, 8, self._d))

        # Transformer decoder to attend to frame features using latent queries
        self.cross_attention = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self._d,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )

        # Project 8 latent slots back to T-length
        self.expand_back = nn.Linear(8, args.clip_len)

        # Final classifier
        self.output_layer = FCLayers(self._d, args.num_classes + 1)

        # Data augmentation and normalization
        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
            T.RandomHorizontalFlip(),
        ])
        self.standarization = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )

    def forward(self, x):
        x = x / 255.
        B, T, C, H, W = x.shape

        if self.training:
            for i in range(B):
                x[i] = self.augmentation(x[i])
        for i in range(B):
            x[i] = self.standarization(x[i])

        # Extract spatial features
        x = x.view(-1, C, H, W)
        features = self._features(x).view(B, T, self._d)

        # Cross-attend using learnable latent queries
        queries = self.latent_queries.expand(B, -1, -1)
        attended = self.cross_attention(tgt=queries, memory=features)

        # Project back to sequence length
        attended = attended.permute(0, 2, 1)
        attended = self.expand_back(attended).permute(0, 2, 1)

        # Final classification
        out = self.output_layer(attended)
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


class X3DSpottingModel(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.num_classes = args.num_classes
        self.backbone = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)

        # Obtener dimensión del penúltimo bloque
        self.feat_dim = self.backbone.blocks[-1].proj.in_features

        # Quitar la capa final (fc)
        self.backbone.blocks[-1].proj = nn.Identity()

        # Modelo temporal (BiGRU)
        self.temporal_model = nn.GRU(
            input_size=self.feat_dim,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )

        # Clasificador por frame
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes + 1)  # +1 background
        )

        # Augmentations
        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.1, brightness=0.8, contrast=0.8)], p=0.3),
            T.RandomHorizontalFlip(),
        ])

        self.standarization = T.Normalize(mean=(0.485, 0.456, 0.406),
                                          std=(0.229, 0.224, 0.225))

        # Freeze everything except last block
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.blocks[-1].parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.normalize(x)  # [B, T, C, H, W]

        if self.training:
            x = self.augment(x)

        x = self.standarize(x)
        x = x.permute(0, 2, 1, 3, 4)  # -> [B, C, T, H, W]

        feat = self.backbone(x)  # -> [B, D, 1, 1, 1]
        feat = feat.view(feat.shape[0], self.feat_dim)  # -> [B, D]

        # Expandir a T frames
        T_len = x.shape[2]  # dimensión temporal original
        feat = feat.unsqueeze(1).repeat(1, T_len, 1)  # -> [B, T, D]

        # Modelado temporal
        feat, _ = self.temporal_model(feat)  # -> [B, T, 512]

        # Clasificación por frame
        out = self.classifier(feat)  # -> [B, T, num_classes + 1]
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

class X3DLSTM(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.num_classes = args.num_classes
        self.backbone = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)

        # Feature dim
        self.feat_dim = self.backbone.blocks[-1].proj.in_features

        # Remove final classifier
        self.backbone.blocks[-1].proj = nn.Identity()

        # BiLSTM for temporal modeling
        self.temporal_model = nn.LSTM(
            input_size=self.feat_dim,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )

        # Frame-wise classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes + 1)  # +1 background class
        )

        # Augmentations
        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.1, brightness=0.8, contrast=0.8)], p=0.3),
            T.RandomHorizontalFlip(),
        ])

        self.standarization = T.Normalize(mean=(0.485, 0.456, 0.406),
                                          std=(0.229, 0.224, 0.225))

        # Freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.blocks[-1].parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.normalize(x)  # [B, T, C, H, W]

        if self.training:
            x = self.augment(x)

        x = self.standarize(x)
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]

        feat = self.backbone(x)  # [B, D, 1, 1, 1]
        feat = feat.view(feat.shape[0], self.feat_dim)  # [B, D]

        T_len = x.shape[2]
        feat = feat.unsqueeze(1).repeat(1, T_len, 1)  # [B, T, D]

        feat, _ = self.temporal_model(feat)  # [B, T, 512]
        out = self.classifier(feat)  # [B, T, num_classes + 1]
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
            
class SnatcherLSTMTransformerModel(nn.Module):
    def _init_(self, args=None):
        super()._init_()
        self._feature_arch = args.feature_arch

        # Visual feature extractor
        if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
            features = timm.create_model({
                'rny002': 'regnety_002',
                'rny004': 'regnety_004',
                'rny008': 'regnety_008',
            }[self.feature_arch.rsplit('', 1)[0]], pretrained=True)
            feat_dim = features.head.fc.in_features
            features.head.fc = nn.Identity()
            self._d = feat_dim
        else:
            raise NotImplementedError(args._feature_arch)

        self._features = features

        # Time-conditioned embedding (SnaTCHer style)
        self.pos_enc = nn.Parameter(torch.randn(1, 1, self._d))

        # Transformer block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._d, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # BiLSTM layer after transformer
        self.lstm = nn.LSTM(input_size=self._d, hidden_size=512, num_layers=1,
                            batch_first=True, bidirectional=True)

        # Final classifier
        self._fc = FCLayers(512 * 2, args.num_classes + 1)

        # Augmentation & standardization
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
    
        # Visual feature extraction
        x = x.view(-1, C, H, W)
        feat = self._features(x).view(B, T, self._d)
    
        # SnaTCHer-style time conditioning
        time_idx = torch.arange(T, device=feat.device).unsqueeze(0).unsqueeze(-1).float() / 100
        feat = feat + self.pos_enc * time_idx  # Add time-conditioned embedding
    
        # Transformer
        encoded = self.temporal_transformer(feat)  # [B, T, D]
    
        # LSTM
        lstm_out, _ = self.lstm(encoded)  # [B, T, 2*512]
    
        # Final classification
        out = self._fc(lstm_out)  # [B, T, num_classes+1]
        
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
            
class SnaTCHerTransformerModel(nn.Module):
    def _init_(self, args=None):
        super()._init_()
        self._feature_arch = args.feature_arch

        # Visual backbone (e.g. RegNetY)
        if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
            features = timm.create_model({
                'rny002': 'regnety_002',
                'rny004': 'regnety_004',
                'rny008': 'regnety_008',
            }[self.feature_arch.rsplit('', 1)[0]], pretrained=True)
            feat_dim = features.head.fc.in_features
            features.head.fc = nn.Identity()
            self._d = feat_dim
        else:
            raise NotImplementedError(args._feature_arch)

        self._features = features

        # Time-conditioned Transformer
        self.pos_enc = nn.Parameter(torch.randn(1, 1, self._d))  # Learned scale for time embedding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._d, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Classifier
        self._fc = FCLayers(self._d, args.num_classes + 1)

        # Augmentation and normalization
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
    
        # Visual features
        x = x.view(-1, C, H, W)
        feat = self._features(x).view(B, T, self._d)
    
        # Time encoding: normalized timestep
        time_idx = torch.arange(T, device=feat.device).unsqueeze(0).unsqueeze(-1).float() / 100  # [1, T, 1]
        time_embed = self.pos_enc * time_idx  # [1, T, D]
        feat = feat + time_embed  # [B, T, D]
    
        # Temporal Transformer
        encoded = self.temporal_transformer(feat)
    
        # Classifier
        out = self._fc(encoded)
        
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
  
class Model(BaseRGBModel):
    def __init__(self, args=None):
        super().__init__()  # Call the parent class constructor
        self.device = "cpu"
        if torch.cuda.is_available() and hasattr(args, "device") and args.device == "cuda":
            self.device = "cuda"

        # Select the model implementation based on args or config
        if hasattr(args, "model_type"):
            if args.model_type == "lstm":
                self._model = ImplLSTMs(args)
            elif args.model_type == "x3d":
                self._model = X3DSpottingModel(args)
            elif args.model_type == "x3d_lstm":
                self._model = X3DLSTM(args)
            elif args.model_type == "transformer":
                self._model = TemporalTransformerModel(args)
            else:
                self._model = HybridTransformerLSTMModel2(args)
        else:
            # Default to Transformer model
            self._model = HybridTransformerLSTMModel2(args)
        
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

        weights = torch.tensor([1.0] + [5.0] * (self._num_classes), dtype=torch.float32).to(self.device)

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label']
                label = label.to(self.device).long()

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)
                    pred = pred.view(-1, self._num_classes + 1)  # B*T, num_classes+1
                    label = label.view(-1)  # B*T
                    loss = F.cross_entropy(
                            pred, label, reduction='mean', weight=weights)

                if optimizer is not None:
                    step(optimizer, scaler, loss,
                        lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)     # Avg loss

    def predict(self, seq):
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4:  # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = self._model(seq)

            # apply softmax
            pred = torch.softmax(pred, dim=-1)
            
            return pred.cpu().numpy()
    
    def get_optimizer(self, optim_args):
        # Set up optimizer and scaler
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=optim_args['lr'],
            weight_decay=1e-4
        )
        scaler = torch.cuda.amp.GradScaler()
        return optimizer, scaler
        
    def state_dict(self):
        return self._model.state_dict()
        
    def load(self, state_dict):
        return self._model.load_state_dict(state_dict)