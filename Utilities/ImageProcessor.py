import torch.nn.functional as F

class ImageProcessor:
    @staticmethod
    def Upsample(x, scale = 2):
        return F.interpolate(x, scale_factor=scale, mode="nearest")
        
    @staticmethod
    def PadToMultiple(x, multiple=14):
        if x.dim() == 4:
            _, _, h, w = x.shape
        elif x.dim() == 3:
            _, h, w = x.shape
        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}")

        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple

        # F.pad format: (left, right, top, bottom)
        return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0), h, w
    
    @staticmethod
    def PadToSize(x, patch_size=280):
        if x.dim() == 4:
            _, _, h, w = x.shape
        elif x.dim() == 3:
            _, h, w = x.shape
        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}")

        pad_h = patch_size - h
        pad_w = patch_size - w

        # F.pad format: (left, right, top, bottom)
        return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0), h, w
    
    @staticmethod
    def CropFromTop(x, orig_h, orig_w):
        if x.dim() == 4:
            return x[:, :, :orig_h, :orig_w]
        elif x.dim() == 3:
            return x[:, :orig_h, :orig_w]
        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}")