# script for losses functions
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

torch.cuda.manual_seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

# function to change the input and target dimension from (N,C,D,H,W) to (C,NxDxHxW)
def flatten(data):
    # change the dimension from (N,C,D,H,W) to (C,N,D,H,W)
    new_axis_order = (1, 0, 2, 3, 4)
    transposed = data.permute(new_axis_order)
    # change the dimension from (C,N,D,H,W) to (C,NxDxHxW)
    flattened = transposed.contiguous().view(transposed.size(0), -1)
    return flattened


# calculate dice accuracy
class DiceAccuracy(nn.Module):
    def __init__(self, epsilon=1e-5, weight=None, generalized_dice_option=None):
        super(DiceAccuracy, self).__init__()
        self.epsilon = epsilon
        self.weight = weight
        self.generalized_dice_option = generalized_dice_option

    def forward(self, output, target):
        # use one-hot encoding to expand the target (N,C,D,H,W) => C from 1 to the number of classes
        shape = list(target.size())
        shape[1] = output.size(1)
        target = target.type(torch.cuda.LongTensor)
        target = torch.zeros(shape).to(target.device).scatter_(1, target, 1)
        # change the input and target dimension from (N,C,D,H,W) to (C,NxDxHxW)
        output = flatten(output)
        target = flatten(target)
        target = target.type(torch.cuda.FloatTensor)
        # compute dice per class/channel
        intersection = (output * target).sum(-1)
        union = output.sum(-1) + target.sum(-1)

        if self.generalized_dice_option:
            class_weights = Variable(1. / (target.sum(-1) * target.sum(-1)).clamp(min=epsilon), requires_grad=False)
            intersection = intersection * class_weights
            union = union * class_weights

        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            intersection = intersection * weight
        dice = 2. * intersection / union.clamp(min=self.epsilon)

        # compute averaged dice across all classes/channels
        averaged_dice = torch.mean(dice)

        return dice, averaged_dice
       

# the type of losses supported in this script
class DiceLoss(nn.Module):
    # dice loss is defined as 1 - dice accuracy
    def __init__(self, epsilon=1e-5, weight=None):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.weight = weight

    def forward(self, output, target):
        # compute dice score
        dice = DiceAccuracy(generalized_dice_option=False)
        _, averaged_dice = dice(output, target)
        # compute dice loss
        return 1 - averaged_dice


class GeneralizedDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5, weight=None):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.weight = weight

    def forward(self, output, target):
        # compute generalized dice score
        generalized_dice = DiceAccuracy(generalized_dice_option=True)
        # compute generalized dice loss
        return 1 - generalized_dice(output, target)


class NormalizedCC(nn.Module):
    def __init__(self, window=None, epsilon=1e-5):
        super(NormalizedCC, self).__init__()
        self.window = window
        self.epsilon = epsilon
   
    def forward(self, im1, im2):
        # dimension: N x C x D x H x W where N = 1 and C = 1
        ndims = 3
        
        # set window size
        if self.window is None:
            self.window = [9] * ndims

        # compute CC squares
        im1_2 = im1 * im1
        im2_2 = im2 * im2
        im1_im2 = im1 * im2

        # define filters
        filters = torch.ones([1, 1, *self.window])
        filters = filters.type(torch.cuda.FloatTensor)

        # compute local sums via convolution function
        im1_sum = F.conv3d(im1, filters)
        im2_sum = F.conv3d(im2, filters)
        im1_2_sum = F.conv3d(im1_2, filters)
        im2_2_sum = F.conv3d(im2_2, filters)
        im1_im2_sum = F.conv3d(im1_im2, filters)

        # compute cross correlations
        window_size = np.prod(self.window)
        im1_avg = im1_sum/window_size
        im2_avg = im2_sum/window_size
 
        cross = im1_im2_sum - im1_avg*im2_sum - im2_avg*im1_sum + window_size*im1_avg*im2_avg
        im1_var = im1_2_sum - 2*im1_sum*im1_avg + window_size*im1_avg*im1_avg
        im2_var = im2_2_sum - 2*im2_sum*im2_avg + window_size*im2_avg*im2_avg

        ncc = cross * cross/ (im1_var*im2_var + self.epsilon)
        return torch.mean(ncc)

class NCCLoss(nn.Module):
    # NCC loss is defined as 1 - NCC accuracy
    def __init__(self, window=None, epsilon=1e-5):
        super(NCCLoss, self).__init__()
        self.epsilon = epsilon
        self.window = window

    def forward(self, im1, im2):
        # compute NCC score
        ncc = NormalizedCC(self.window, self.epsilon)
        ncc_value = ncc(im1, im2)
        # compute NCC loss
        return 1 - ncc_value

class SmoothGradLoss(nn.Module):
    def __init__(self, penalty_principle='l1'):
        super(SmoothGradLoss, self).__init__()
        self.penalty = penalty_principle

    def forward(self, f):
        assert self.penalty in ['l1', 'l2']
        if self.penalty == 'l1':
            df = [torch.mean(torch.abs(df_i)) for df_i in self.diff(f)]
        else: 
            df = [torch.mean(torch.abs(df_i * df_i)) for df_i in self.diff(f)]

        df_mean = torch.mean(torch.stack(df))   

        return df_mean
        
    def diff(self, f):
        # the dis field dimension: N x C x D x H x W
        assert len(f.shape) == 5
        
        vol_shape = list(f.shape[2:])
        ndims = len(vol_shape)

        # initiate diffusion field
        df = [None] * ndims
        # update diffusion field
        for dim_id in range(ndims):
            # permute matrix to move the #id dimension to the first
            new_dim = [dim_id+2, *range(dim_id+2), *range(dim_id+3, ndims+2)]
            new_f = f.permute(new_dim)
            new_df = new_f[1:,:,:,:,:] - new_f[:-1,:,:,:,:]
            # permute matrix back to N x C x D x H x W
            dim = [*range(1, dim_id+3), 0, *range(dim_id+3, ndims+2)]
            df[dim_id] = new_df.permute(dim)
 
        return df
