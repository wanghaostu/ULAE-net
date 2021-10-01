import torch
import numpy as np
import torch.nn.functional as F
import math

def smoothloss(y_pred):
    dy = torch.abs(y_pred[:,:,1:, :, :] - y_pred[:,:, :-1, :, :])
    dx = torch.abs(y_pred[:,:,:, 1:, :] - y_pred[:,:, :, :-1, :])
    dz = torch.abs(y_pred[:,:,:, :, 1:] - y_pred[:,:, :, :, :-1])
    return (torch.mean(dx * dx)+torch.mean(dy*dy)+torch.mean(dz*dz))/3.0

def JacboianDet(y_pred, sample_grid):
    J = y_pred + sample_grid
    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:,:,:,:,0] * (dy[:,:,:,:,1] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,1])
    Jdet1 = dx[:,:,:,:,1] * (dy[:,:,:,:,0] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,0])
    Jdet2 = dx[:,:,:,:,2] * (dy[:,:,:,:,0] * dz[:,:,:,:,1] - dy[:,:,:,:,1] * dz[:,:,:,:,0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet

def mse(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


def msewithmask(y_true, y_pred, mask):
    return torch.mean(((y_true - y_pred)*mask) ** 2)


def neg_Jdet_loss(y_pred, sample_grid):
    neg_Jdet = -1.0 * JacboianDet(y_pred, sample_grid)
    selected_neg_Jdet = F.relu(neg_Jdet)

    return torch.mean(selected_neg_Jdet)
    
def dice(y_true, y_pred):
    ndims = len(list(y_pred.size())) - 2
    vol_axes = list(range(2, ndims+2))
    top = 2 * (y_true * y_pred).sum(dim=vol_axes)
    bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
    dice = torch.mean(top / bottom)
    return -dice

def magnitude_loss(flow_1, flow_2):
    num_ele = torch.numel(flow_1)
    flow_1_mag = torch.sum(torch.abs(flow_1))
    flow_2_mag = torch.sum(torch.abs(flow_2))

    diff = (torch.abs(flow_1_mag - flow_2_mag))/num_ele

    return diff

def ncc_loss(I, J, win=None, mind=False):

    if win is None:
        win = [9] * 3


    sum_filter = torch.ones(1, I.shape[1], *win).to("cuda")
    pad = math.floor(win[0] / 2)
    stride = (1, 1, 1)
    pading = (pad, pad, pad)

    I_var, J_var, cross = compute_local_sums(I, J, sum_filter, stride, pading, win)

    cc = cross * cross / (I_var * J_var + 1e-5)
    return -1 * torch.mean(cc)

def compute_local_sums(I, J, filter, stride, padding, win):

    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv3d(I, filter, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filter, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filter, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filter, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filter, stride=stride, padding=padding)

    win_size = np.prod(win) * I.shape[1]
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross


class NCC(torch.nn.Module):

    # local (over window) normalized cross correlation

    def __init__(self, win=5, eps=1e-5):
        super(NCC, self).__init__()
        self.win = win
        self.eps = eps
        self.w_temp = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.w_temp

        # set window size
        if self.win is None:
            self.win = [5] * ndims
        else:
            self.win = [self.w_temp] * ndims

        weight_win_size = self.w_temp
        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)

            
class multi_window_loss(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def  __init__(self, win=[11,9,7], eps=1e-5, gamma=0.5):
        super(multi_window_loss, self).__init__()
        self.num_scale = len(win)
        self.gamma = gamma
        self.similarity_metric = []
        
        for i in range(self.num_scale):
            self.similarity_metric.append(NCC(win=win[i]))

    def forward(self, I, J):
        total_NCC = []
        for i in range(self.num_scale):
            current_NCC = self.similarity_metric[i](I, J)
            total_NCC.append(current_NCC*(self.gamma**i))
        return total_NCC