from abc import ABC, abstractmethod
from Utilities.ImageProcessor import ImageProcessor
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

class ValidationHelperBase(ABC):    
    def __init__(self, scale: int = 4):
        self.scale = scale

    @abstractmethod
    def EvaluteForTesting(
        self,
        data_loader: DataLoader, 
        model: nn.Module, 
    ):
        pass
    
    def GetInference(self, batch, model):
        # Pad the iage to be divisible by 14 and 16
        valids = [112, 224, 336, 448, 560, 672, 784] # Add more valid multiples if required
        h, w = batch['rgb'].shape[-2], batch['rgb'].shape[-1]

        filtered = [i for i in valids if i >= max([h, w])]


        img, h, w = ImageProcessor.PadToSize(batch['rgb'], filtered[0])
        depth_norm, h, w = ImageProcessor.PadToSize(batch['gt_norm'], filtered[0])

        for k, v in batch.items():
            if k not in ['orig_h', 'orig_w']:
                batch[k] = v.cuda(non_blocking=True)

        lr = TF.resize(depth_norm, (filtered[0]//self.scale, filtered[0]//self.scale), interpolation=TF.InterpolationMode.BICUBIC)

        out = model(img.cuda(non_blocking=True), lr.cuda(non_blocking=True))

        return ImageProcessor.CropFromTop(out, h, w), batch['gt_norm']
    
    