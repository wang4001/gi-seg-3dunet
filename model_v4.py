# script for 3d unet, exactly as 3dunet paper
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class Unet3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, interpolate=None, concatenate=True,
                 norm_type=None, init_channels=32, scale_factor=(2, 2, 2)):
        super(Unet3d, self).__init__()

        # define the number of groups in the group normalization
        num_groups = init_channels // 8

        # define the encoder pathway
        self.encoders = nn.ModuleList([
            EncoderModule(in_channels, init_channels, kernel_size, norm_type, num_groups, max_pool=True),
            EncoderModule(init_channels, init_channels * 2, kernel_size, norm_type, num_groups, max_pool=True),
            EncoderModule(init_channels * 2, init_channels * 4, kernel_size, norm_type, num_groups, max_pool=True),
            EncoderModule(init_channels * 4, init_channels * 8, kernel_size, norm_type, num_groups, max_pool=False)])

        # define the decoder pathway
        if concatenate:  # concatenate is True
            self.decoders = nn.ModuleList([
                DecoderModule(init_channels * 8 + init_channels * 4, init_channels * 4, kernel_size, norm_type, num_groups, interpolate,
                              concatenate, scale_factor),
                DecoderModule(init_channels * 4 + init_channels * 2, init_channels * 2, kernel_size, norm_type, num_groups, interpolate,
                              concatenate, scale_factor),
                DecoderModule(init_channels * 2 + init_channels, init_channels, kernel_size, norm_type, num_groups, interpolate,
                              concatenate, scale_factor)])
        else:  # concatenate is False -------  need to be further modified: add one-by-one convolution to make addition possible
            self.decoders = nn.ModuleList([
                DecoderModule(init_channels * 4, init_channels * 4, kernel_size, norm_type, num_groups, interpolate,
                              concatenate, scale_factor),
                DecoderModule(init_channels * 2, init_channels * 2, kernel_size, norm_type, num_groups, interpolate,
                              concatenate, scale_factor),
                DecoderModule(init_channels, init_channels, kernel_size, norm_type, num_groups, interpolate,
                              concatenate, scale_factor)])

        #print('in the code -----------------------------------')
        #print(init_channels, out_channels)
        self.final = nn.ModuleList([
            nn.Conv3d(in_channels=init_channels, out_channels=out_channels, kernel_size=1),
            nn.Softmax(dim=1)])

    def forward(self, x):
        # forward in the encoder pathway
        #print(x.size())
        encoder_features = []
        for encoder in self.encoders:
            x, encoder_feature = encoder(x)
            encoder_features.insert(0, encoder_feature)
        # remove the final encoder features
        encoder_features = encoder_features[1:]

        # forward in the decoder pathway
        for encoder_feature, decoder in zip(encoder_features, self.decoders):
            x = decoder(x, encoder_feature)
        
        last_decoder = x
        # forward in the final layer
        for module in self.final:
            x = module(x)

        # return the final results
        return x, last_decoder


class EncoderModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm_type, num_groups, max_pool):
        super(EncoderModule, self).__init__()

        # save the flag for max pooling
        self.max_pool = max_pool
        
        # define the module in the encoder -------------------------------------- maybe independent function
        if norm_type is False:  # module without any normalization
            self.conv_module = nn.ModuleList([
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=kernel_size, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=kernel_size, padding=1),
                nn.ReLU(inplace=True)])

        elif norm_type == 'GroupNorm':  # module with group normalization
            self.conv_module = nn.ModuleList([
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=kernel_size, padding=1),
                nn.ReLU(inplace=True),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_channels // 2),
                nn.Conv3d(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=kernel_size, padding=1),
                nn.ReLU(inplace=True),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)])
        
        self.maxpool_module = nn.MaxPool3d(kernel_size=(2, 2, 2), padding=0)

    def forward(self, x):
        # forward to convolution module
        for component in self.conv_module:
            #print(x.size())
            x = component(x)
        # record the encoder features before maxpooling
        encoder_feature = x
        # forward to max pooling module
        if self.max_pool:
            x = self.maxpool_module(x)
        return x, encoder_feature


class DecoderModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm_type, num_groups, interpolate, concatenate,
                 scale_factor):
        super(DecoderModule, self).__init__()

        # save concatenate types
        self.concatenate = concatenate
        self.scale_factor = scale_factor
        self.interpolate = interpolate

        # define the interpolation module
        if self.interpolate is False:
            self.upsample = nn.ConvTranspose3d(in_channels=out_channels * 2, out_channels=out_channels * 2,
                                               kernel_size=kernel_size, stride=scale_factor, padding=(1, 1, 1),
                                               output_padding=(1, 1, 1))
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

    def forward(self, x, encoder_features):
        # forward to upsample module
        if self.upsample == self.interpolate:
            x = F.interpolate(input=x, scale_factor=self.scale_factor, mode=self.interpolate)
        else:
            x = self.upsample(x)

        # forward to combine encoder and decoder features
        if self.concatenate:  # concatenation
            x = torch.cat((encoder_features, x), dim=1)
        else:  # addition
            x = encoder_features + x

        # forward to convolution module
        for component in self.conv_module:
            #print(x.size())
            x = component(x)

        return x
