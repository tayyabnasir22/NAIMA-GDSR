import torch

class Evaluation:
    @staticmethod
    def DepthRMSE(a, b, mx, mn, shave_pixels: bool = False):
        if shave_pixels == True: # This is to be done for validation and testing not training
            # Crop 6 pixels from all sides to remove boundary effects.
            a = a[:, :, 6:-6, 6:-6]
            b = b[:, :, 6:-6, 6:-6]
        
        # it is a*(max-min) + min
        a = a*(mx-mn) + mn
        b = b*(mx-mn) + mn
        a = a * 100
        b = b * 100
        
        return torch.sqrt(torch.mean(torch.pow(a-b,2))).item()

    @staticmethod    
    def DepthRMSEBenchmark(a, b, shave_pixels: bool = False):
        if shave_pixels == True: # This is to be done for validation and testing not training
            # Crop 6 pixels from all sides to remove boundary effects.
            a = a[:, :, 6:-6, 6:-6]
            b = b[:, :, 6:-6, 6:-6]
        
        # it is a*(max-min) + min
        a = a * 255.0
        b = b * 255.0
        
        return torch.sqrt(torch.mean(torch.pow(a-b,2))).item()
