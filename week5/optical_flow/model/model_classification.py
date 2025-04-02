"""
File containing the main model.
"""

#Standard imports
import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F

import torchvision.models.video as models

#Local imports
from model.modules import BaseRGBModel, FCLayers, step
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def free_memory():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args = None):
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
            self._fc = FCLayers(self._d, args.num_classes)

            #Augmentations and crop
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
                T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.GaussianBlur(5)], p = 0.25),
                T.RandomHorizontalFlip(),
            ])

            #Standarization
            self.standarization = T.Compose([
                T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) #Imagenet mean and std
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

            #Max pooling
            im_feat = torch.max(im_feat, dim=1)[0] #B, D

            #MLP
            im_feat = self._fc(im_feat) #B, num_classes

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
            print('Model params:',
                sum(p.numel() for p in self.parameters()))
    


    class two_stream(nn.Module):
        def __init__(self, args=None):
            super().__init__()
            self.rgb_model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
            self.rgb_model.blocks[-1].proj = nn.Linear(2048, 512)
            #self.rgb_model.blocks[-1].proj = nn.Identity()
            #self.rgb_model.blocks[-1].output_pool = nn.Identity()  
            

            for param in self.rgb_model.parameters():
                param.requires_grad = False

            for param in self.rgb_model.blocks[-1].parameters():
                param.requires_grad = True

            self.flow_model=torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
            self.flow_model.blocks[0].conv.conv_t = nn.Conv3d(in_channels=2,  # Change from 3 to 2
                                                    out_channels=24,
                                                    kernel_size=(1, 3, 3),
                                                    stride=(1, 2, 2),
                                                    padding=(0, 1, 1),
                                                    bias=False
                                                )
            self.flow_model.blocks[-1].proj = nn.Linear(2048, 512)
            
            for param in self.flow_model.parameters():
                param.requires_grad = False

            for param in self.flow_model.blocks[-1].parameters():
                param.requires_grad = True
                            
            #self.flow_model.blocks[-1].proj = nn.Identity()
            #self.flow_model.blocks[-1].output_pool = nn.Identity() 

#            self.fc=self.fc_fusion = nn.Linear(2048 * 2, args.num_classes)
#            self.output_pool= nn.AdaptiveAvgPool3d(output_size=1)

            #Augmentations and crop
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
                T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.GaussianBlur(5)], p = 0.25),
                T.RandomHorizontalFlip(),
            ])

            #Standarization
            self.standarization = T.Compose([
                T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) #Imagenet mean and std
            ])
            
            # Self-Attention Layer
            self._self_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
            self._norm1 = nn.LayerNorm(512)

            # MLP for classification
            self._fc = FCLayers(512*2, args.num_classes)


        def preprocess(self,x):
            x = self.normalize(x)
            if self.training:
                x = self.augment(x)
            x = self.standarize(x)
            x = x.permute(0, 2, 1, 3, 4)
            return x
        
        def preprocess_flow(self,x):
            x = self.normalize(x)

            x = x.permute(0, 2, 1, 3, 4) #B, 2, T, H, W
            return x
        
        def forward(self, x,flow):
            x=self.preprocess(x)
            flow=self.preprocess_flow(flow)

            rgb_feat = self.rgb_model(x)  # Extract features from RGB
            flow_feat = self.flow_model(flow)  # Extract features from Flow

            fusion_feat = torch.cat((rgb_feat, flow_feat), dim=1)

            #MLP
            im_feat = self._fc(fusion_feat) #B, num_classe

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

            dummy_input = torch.randn(4, 50, 3, 224, 398).to(next(self.parameters()).device)
            dummy_input_flow = torch.randn(4, 50, 2, 224, 398).to(next(self.parameters()).device)
            self.eval()
            flops = FlopCountAnalysis(self, dummy_input,dummy_input_flow)

            print(f"FLOPs: {flops.total():,}")
            print(parameter_count_table(self))

            

    def __init__(self, args=None):
        self.device = "cpu"
        if torch.cuda.is_available() and ("device" in args) and (args.device == "cuda"):
            self.device = "cuda"

        # self._model = Model.Impl(args=args)
        # self._model=Model.r2plus1d_18(args=args)
        #self._model=Model.two_stream(args=args)
        self._model = Model.X3DClassifier(args)
        self._model.print_stats()
        self._args = args

        self._model.to(self.device)
        self._num_classes = args.num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):
        inference = optimizer is None
        self._model.eval() if inference else self._model.train()
        epoch_loss = 0.
        
        with torch.no_grad() if inference else nullcontext():
            for batch in tqdm(loader):
                frame = batch['frame'].to(self.device, non_blocking=True).float()
                label = batch['label'].to(self.device, non_blocking=True).float()
                
                with torch.cuda.amp.autocast():
                    pred = self._model(frame)
                    loss = F.binary_cross_entropy_with_logits(pred, label)

                if optimizer:
                    # Clear gradients
                    optimizer.zero_grad()
                    
                    if scaler:
                        # Use gradient scaling for mixed precision
                        scaler.scale(loss).backward()
                        
                        # Add gradient clipping to prevent exploding gradients
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                        
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Standard backprop without mixed precision
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                        optimizer.step()
                    
                    if lr_scheduler:
                        lr_scheduler.step()
                    
                    free_memory()

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)
        
    def predict(self, seq):
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        seq = seq.unsqueeze(0) if len(seq.shape) == 4 else seq
        seq = seq.to(self.device).float()
        
        self._model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            pred = self._model(seq)
        
        return torch.sigmoid(pred).cpu().numpy()
#    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):
#
#        if optimizer is None:
#            inference = True
#            self._model.eval()
#        else:
#            inference = False
#            optimizer.zero_grad()
#            self._model.train()
#
#        epoch_loss = 0.
#        with torch.no_grad() if optimizer is None else nullcontext():
#            for batch_idx, batch in enumerate(tqdm(loader)):
#                frame = batch['frame'].to(self.device).float()
#                flow = batch['flow'].to(self.device).float()
#                label = batch['label']
#                label = label.to(self.device).float()
#
#                with torch.cuda.amp.autocast():
#                    pred = self._model(frame,flow)
#                    loss = F.binary_cross_entropy_with_logits(
#                            pred, label)
#
#                if optimizer is not None:
#                    step(optimizer, scaler, loss,
#                        lr_scheduler=lr_scheduler)
#
#                epoch_loss += loss.detach().item()
#
#        return epoch_loss / len(loader)     # Avg loss
#
#    def predict(self, seq,flow):
#
#        if not isinstance(seq, torch.Tensor):
#            seq = torch.FloatTensor(seq)
#            flow=torch.FloatTensor(flow)
#        if len(seq.shape) == 4: # (L, C, H, W)
#            seq = seq.unsqueeze(0)
#            flow=flow.unsqueeze(0)
#        if seq.device != self.device:
#            seq = seq.to(self.device)
#            flow=flow.to(self.device)
#        seq = seq.float()
#        flow=flow.float()
#
#        self._model.eval()
#        with torch.no_grad():
#            with torch.cuda.amp.autocast():
#                pred = self._model(seq,flow)
#
#            # apply sigmoid
#            pred = torch.sigmoid(pred)
#            
#            return pred.cpu().numpy()