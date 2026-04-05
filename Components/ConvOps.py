import torch.nn as nn
import torch.nn.init as init

class ConvOps:
    @staticmethod
    def DefaultConv(in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), bias=bias)

    @staticmethod
    def ProjectionConv(in_channels, out_channels, scale, up=True):
        kernel_size, stride, padding = {
            2: (6, 2, 2),
            4: (8, 4, 2),
            8: (12, 8, 2),
            16: (20, 16, 2)
        }[scale]
        if up:
            conv_f = nn.ConvTranspose2d
        else:
            conv_f = nn.Conv2d

        return conv_f(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )
    
    @staticmethod
    def StableInitWeights(net_l, scale=1):
        if not isinstance(net_l, list):
            net_l = [net_l]
        for net in net_l:
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                    m.weight.data *= scale  # for residual block
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                    m.weight.data *= scale
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias.data, 0.0)

    @staticmethod
    def XavierInitWeights(net_l, scale=1):
        if not isinstance(net_l, list):
            net_l = [net_l]
        for net in net_l:
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    init.xavier_normal_(m.weight)
                    m.weight.data *= scale  # for residual block
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    init.xavier_normal_(m.weight)
                    m.weight.data *= scale
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias.data, 0.0)

    @staticmethod
    def ChannelMean(F):
        spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
        return spatial_sum / (F.size(2) * F.size(3))

    @staticmethod
    def ChannelSTD(F):
        assert(F.dim() == 4)
        F_mean = ConvOps.ChannelMean(F)
        F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
        return F_variance.pow(0.5)