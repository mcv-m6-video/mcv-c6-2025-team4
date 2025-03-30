"""
Optimized model to reduce memory usage.
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

# Enable optimized CUDA performance
torch.backends.cudnn.benchmark = True

def free_memory():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

class Model(BaseRGBModel):

    class Impl(nn.Module):
        def __init__(self, args=None):
            super().__init__()
            self._feature_arch = args.feature_arch
            self._d = None

            if self._feature_arch.startswith('resnet50'):
                features = timm.create_model('resnet50', pretrained=True)
                self._d = features.fc.in_features
                features.fc = nn.Identity()
            elif self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
                model_map = {
                    'rny002': 'regnety_002',
                    'rny004': 'regnety_004',
                    'rny008': 'regnety_008'
                }
                features = timm.create_model(model_map[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
                self._d = features.head.fc.in_features
                features.head.fc = nn.Identity()
            else:
                raise NotImplementedError(args.feature_arch)

            self._features = features
            self._fc = FCLayers(self._d, args.num_classes)

            # Data augmentations
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
                T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
                T.RandomHorizontalFlip(),
            ])

            # Standardization
            self.standarization = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        def forward(self, x):
            x = x / 255.  # Normalize
            batch_size, clip_len, channels, height, width = x.shape

            if self.training:
                x = torch.stack([self.augmentation(frame) for frame in x])

            x = torch.stack([self.standarization(frame) for frame in x])
            x = self._features(x.view(-1, channels, height, width)).reshape(batch_size, clip_len, self._d)
            im_feat = torch.max(x, dim=1)[0]
            return self._fc(im_feat)

    def __init__(self, args=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
        self._model = Model.Impl(args).to(self.device)
        self._args = args
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
                    step(optimizer, scaler, loss, lr_scheduler=lr_scheduler)
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
