import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F

from model.modules import BaseRGBModel, FCLayers, step
from fvcore.nn import FlopCountAnalysis, parameter_count_table

torch.backends.cudnn.benchmark = True  # OptimizaciÃ³n para GPU

def free_memory():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

class Model(BaseRGBModel):

    class Impl(nn.Module):
        def __init__(self, args=None):
            super().__init__()
            self._feature_arch = args.feature_arch
            self._d = None

            # Mejor backbone (regnety_004 en lugar de regnety_002)
            model_map = {
                'rny002': 'regnety_002',
                'rny004': 'regnety_004',  # Mejor rendimiento sin aumentar mucho el cÃ³mputo
                'rny008': 'regnety_008'
            }
            features = timm.create_model(model_map[args.feature_arch], pretrained=True)
            self._d = features.head.fc.in_features
            features.head.fc = nn.Identity()
            self._features = features

            # Capa de clasificaciÃ³n mejorada con BatchNorm y Dropout
            self._fc = nn.Sequential(
                nn.BatchNorm1d(self._d),
                nn.Dropout(0.3),  # RegularizaciÃ³n para evitar overfitting
                FCLayers(self._d, args.num_classes)
            )

            # Mejor combinaciÃ³n de augmentaciones
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue=0.1, brightness=0.8, contrast=0.8)], p=0.3),
                T.RandomHorizontalFlip(),
            ])

            # NormalizaciÃ³n
            self.standarization = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        def forward(self, x):
            x = x / 255.  # Normalizar a [0,1]
            batch_size, clip_len, channels, height, width = x.shape

            if self.training:
                x = torch.stack([self.augmentation(frame) for frame in x])

            x = torch.stack([self.standarization(frame) for frame in x])
            x = self._features(x.view(-1, channels, height, width)).reshape(batch_size, clip_len, self._d)
            im_feat = torch.max(x, dim=1)[0]  # Max pooling temporal

            return self._fc(im_feat)
        def print_stats(self):
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
    
            dummy_input = torch.randn(4, 50, 3, 224, 224).to(next(self.parameters()).device) 
    
            self.eval()
            try:
                flops = FlopCountAnalysis(self, dummy_input)
                print(f"FLOPs: {flops.total():,}")
                print(parameter_count_table(self))
            except Exception as e:
                print(f"Could not calculate FLOPs: {e}")

    class X3DClassifier(nn.Module):
      def __init__(self, args=None):
          super().__init__()
          self.model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
          
          # Replace final classification layer
          self.model.blocks[-1].proj = nn.Linear(2048, args.num_classes)
          
          for param in self.model.parameters():
            param.requires_grad = False  # Freeze all parameters
        
          # Unfreeze the last few layers
          for param in self.model.blocks[-1].parameters():  
              param.requires_grad = True
          
          # Mejor combinaciÃ³n de augmentaciones
          self.augmentation = T.Compose([
              T.RandomApply([T.ColorJitter(hue=0.1, brightness=0.8, contrast=0.8)], p=0.3),
              T.RandomHorizontalFlip(),
          ])

          # NormalizaciÃ³n
          self.standarization = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

          
      def forward(self, x):
            x = self.normalize(x) #Normalize to 0-1

            if self.training:
                x = self.augment(x) #augmentation per-batch

            x = self.standarize(x) #standarization imagenet stats
             #B, T, C, H, W to #B, C, T,H, W    
            x = x.permute(0, 2, 1, 3, 4)     
           
              
            return self.model(x)
            
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
        
            # Create a dummy input matching the model's expected input shape
            dummy_input = torch.randn(4, 50, 3, 224, 398).to(next(self.parameters()).device) 
        
            # Compute FLOPs and MACs
            self.eval()
            try:
                flops = FlopCountAnalysis(self, dummy_input)
                print(f"FLOPs: {flops.total():,}")
                print(parameter_count_table(self))
            except Exception as e:
                print(f"Could not calculate FLOPs: {e}")

    class SlowFast(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
            self.model.blocks[-1].proj = nn.Linear(2304, args.num_classes)
            
            # Standard SlowFast configuration
            self.alpha = 8  # Temporal downsampling factor for slow pathway
            self.beta = 2   # Fast pathway uses 2x lower resolution spatially
            
            # Get default pathway sizes from the model
            self.slow_pathway_size = 8
            self.fast_pathway_size = 32
            
            # First freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False
                
            # Unfreeze the last ResStage (block 4) for both pathways
            # This includes the last fusion stage and feature extraction blocks
            for param in self.model.blocks[4].parameters():
                param.requires_grad = True
                
            # Unfreeze the pooling pathway (block 5)
            for param in self.model.blocks[5].parameters():
                param.requires_grad = True
                
            # Unfreeze the classification head (block 6)
            for param in self.model.blocks[6].parameters():
                param.requires_grad = True
                
            # You can also gradually unfreeze more blocks if needed
            # For example, you could also unfreeze block 3 which has important fusion layers:
            for param in self.model.blocks[3].parameters():
                param.requires_grad = True
            
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue=0.1, brightness=0.8, contrast=0.8)], p=0.3),
                T.RandomHorizontalFlip(),
            ])
            
            self.standarization = T.Normalize(mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225))
            
        def forward(self, x):
            # Input shape: [B, T, C, H, W]
            x = self.normalize(x)
            if self.training:
                x = self.augment(x)
            x = self.standarize(x)
            
            # Prepare input according to SlowFast requirements
            B, T, C, H, W = x.shape
            
            # Convert to [B, C, T, H, W] format for easier time-dimension manipulation
            x = x.permute(0, 2, 1, 3, 4)
            
            # Create slow pathway with uniform sampling
            if T > self.slow_pathway_size:
                indices = torch.linspace(0, T-1, self.slow_pathway_size).long().to(x.device)
                slow_pathway = x[:, :, indices]
            else:
                slow_pathway = F.interpolate(
                    x, size=[self.slow_pathway_size, H, W], mode='trilinear', align_corners=False
                )
            
            # Create fast pathway with uniform sampling 
            if T > self.fast_pathway_size:
                indices = torch.linspace(0, T-1, self.fast_pathway_size).long().to(x.device)
                fast_pathway = x[:, :, indices]
            else:
                fast_pathway = F.interpolate(
                    x, size=[self.fast_pathway_size, H, W], mode='trilinear', align_corners=False
                )
            
            # IMPORTANT: Do NOT downsample the spatial dimensions of the fast pathway
            # The original PyTorchVideo SlowFast implementation handles this internally
            # Just make sure both pathways have the correct number of frames
            
            # Pass both pathways to the model
            return self.model([slow_pathway, fast_pathway])
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
                # Create dummy inputs with the correct shapes
                B, C, H, W = 4, 3, 224, 398
                
                # Create slow pathway with expected temporal dimension
                slow_pathway = torch.randn(B, C, self.slow_pathway_size, H, W).to(next(self.parameters()).device)
                
                # Create fast pathway with expected temporal dimension and reduced spatial resolution
                fast_pathway = torch.randn(
                    B, C, self.fast_pathway_size, H // self.beta, W // self.beta
                ).to(next(self.parameters()).device)
                
                # Use a wrapper class with a fixed forward method
                class ModelWrapper(torch.nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                        self.slow_size = model.slow_pathway_size
                        self.fast_size = model.fast_pathway_size
                        self.beta = model.beta
                    
                    def forward(self, x):
                        # Instead of unpacking, we'll reshape the input back to the two pathways
                        # Calculate sizes to split the flattened input back into two tensors
                        slow_size = B * C * self.slow_size * H * W
                        # We know where to split because we flattened them in a specific order
                        slow_flat, fast_flat = torch.split(x, [slow_size, x.size(1) - slow_size], dim=1)
                        
                        # Reshape back to original dimensions
                        slow_pathway = slow_flat.view(B, C, self.slow_size, H, W)
                        fast_pathway = fast_flat.view(B, C, self.fast_size, H // self.beta, W // self.beta)
                        
                        return self.model([slow_pathway, fast_pathway])
                
                self.eval()
                
                # Create the wrapper
                wrapper_model = ModelWrapper(self)
                
                # Create a combined input by flattening and concatenating both pathways
                slow_flat = slow_pathway.flatten(1)
                fast_flat = fast_pathway.flatten(1)
                combined_input = torch.cat([slow_flat, fast_flat], dim=1)
                
                # Calculate FLOPs using the wrapper
                flops = FlopCountAnalysis(wrapper_model, combined_input)
                
                print(f"FLOPs: {flops.total():,}")
                print(parameter_count_table(self))
            except Exception as e:
                print(f"Could not calculate FLOPs: {e}")
                print(f"Error details: {str(e)}")
                
    def __init__(self, args=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
        self._model = Model.SlowFast(args)
        self._args = args
        self._model.to(self.device)
        self._num_classes = args.num_classes
        
        # Print model stats if possible
        try:
            self._model.print_stats()
        except Exception as e:
            print(f"Warning: Could not print model stats: {e}")

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
                
    # def __init__(self, args=None):
    #     self.device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    #     self._model = Model.SlowFast(args)
    #     print(self._model)
    #     try:
    #         self._model.print_stats()
    #     except Exception as e:
    #         print(f"Warning: Could not print model stats: {e}")
    #     self._args = args
    #     self._model.to(self.device)
    #     self._num_classes = args.num_classes

    # def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):
    #     inference = optimizer is None
    #     self._model.eval() if inference else self._model.train()
    #     epoch_loss = 0.
        
    #     with torch.no_grad() if inference else nullcontext():
    #         for batch in tqdm(loader):
    #             frame = batch['frame'].to(self.device, non_blocking=True).float()
    #             label = batch['label'].to(self.device, non_blocking=True).float()
                
    #             with torch.cuda.amp.autocast():
    #                 pred = self._model(frame)
    #                 loss = F.binary_cross_entropy_with_logits(pred, label)

    #             if optimizer:
    #                 step(optimizer, scaler, loss, lr_scheduler=lr_scheduler)
    #                 free_memory()

    #             epoch_loss += loss.detach().item()

    #     return epoch_loss / len(loader)

    def predict(self, seq):
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        seq = seq.unsqueeze(0) if len(seq.shape) == 4 else seq
        seq = seq.to(self.device).float()
        
        self._model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            pred = self._model(seq)
        
        return torch.sigmoid(pred).cpu().numpy()