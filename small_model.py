# script for 3d unet, exactly as 3dunet paper
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class smallmodel(nn.Module):
    def __init__(self, out_channels, kernel_size=3, interpolate=None,
                 norm_type=None, init_channels=64, scale_factor=(1, 2, 2)):
        super(smallmodel, self).__init__()

        # define the number of groups in the group normalization
        num_groups = init_channels // 8

        # define the decoder pathway
        self.decoders = nn.ModuleList([
                small_model_DecoderModule(init_channels, init_channels//2, kernel_size, norm_type, num_groups, interpolate, scale_factor)])

        self.final = nn.ModuleList([
            nn.Conv3d(in_channels=init_channels//2, out_channels=out_channels, kernel_size=1),
            nn.Softmax(dim=1)])

    def forward(self, x):

        # forward in the decoder pathway
        for decoder in self.decoders:
            x = decoder(x)
        
        # forward in the final layer
        for module in self.final:
            x = module(x)

        # return the final results
        return x

class small_model_DecoderModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm_type, num_groups, interpolate, scale_factor):
        super(small_model_DecoderModule, self).__init__()

        self.scale_factor = scale_factor
        self.interpolate = interpolate

        # define the interpolation module
        if self.interpolate is False:
            self.upsample = nn.ConvTranspose3d(in_channels=out_channels * 2, out_channels=out_channels * 2,
                                               kernel_size=kernel_size, stride=scale_factor, padding=(1, 1, 1),
                                               output_padding=(0, 1, 1))

        else:
            self.upsample = self.interpolate

        # define the module in the decoder -------------------------------------- maybe independent function
        if norm_type is False: # module without any normalization
            self.conv_module = nn.ModuleList([
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
                nn.ReLU(inplace=True)])

        elif norm_type == 'GroupNorm':  # module with group normalization
            self.conv_module = nn.ModuleList([
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
                nn.ReLU(inplace=True),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
                nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
                nn.ReLU(inplace=True),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)])

    def forward(self, x):
        # forward to upsample module
        if self.upsample == self.interpolate:
            x = F.interpolate(input=x, scale_factor=self.scale_factor, mode=self.interpolate)
        else:
            x = self.upsample(x)

        # forward to convolution module
        for component in self.conv_module:
            x = component(x)

        return x
