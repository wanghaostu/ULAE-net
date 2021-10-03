import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from torch.distributions.normal import Normal


# from Transforms import *

class SpatialTransformer(nn.Module):
    def __init__(self, size, mode="bilinear"):
        super(SpatialTransformer, self).__init__()

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
            # -0.5, 0.5 * 2 -> -1.0, 1.0
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        # print(new_locs.shape)
        # B C D H W
        # N D H W
        new_locs = new_locs[..., [2, 1, 0]]
        # z y x -> x y z
        # print(new_locs)
        # print(new_locs.shape)
        # print(src.shape)
        """
        input (Tensor): input of shape :math:`(N, C, H_\text{in}, W_\text{in})` (4-D case)
                        or :math:`(N, C, D_\text{in}, H_\text{in}, W_\text{in})` (5-D case)
        grid (Tensor): flow-field of shape :math:`(N, H_\text{out}, W_\text{out}, 2)` (4-D case)
                       or :math:`(N, D_\text{out}, H_\text{out}, W_\text{out}, 3)` (5-D case)
        """
        return F.grid_sample(src, new_locs, mode=self.mode, align_corners=True)


class Uncoupled_Encoding_Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=4):
        super(Uncoupled_Encoding_Block, self).__init__()
        self.mean = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.conv = nn.Conv3d(in_ch, out_ch, 1, stride=stride)
        self.conv2 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.ab1 = nn.Conv3d(in_ch, out_ch, 1, stride=stride)
        self.ab6 = nn.Conv3d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, stride=stride)
        self.ab12 = nn.Conv3d(in_ch, out_ch, 3, padding=dilation * 2, dilation=dilation * 2, stride=stride)
        self.ab18 = nn.Conv3d(in_ch, out_ch, 3, padding=dilation * 3, dilation=dilation * 3, stride=stride)
        self.out = nn.Conv3d(out_ch * 7, out_ch, 1, 1)

    def forward(self, x, y):
        size = [int(i / 2) for i in x.shape[2:]]
        fea_e0 = self.mean(x)
        fea_e0 = self.conv(fea_e0)
        fea_e0 = F.upsample(fea_e0, size=size, mode='trilinear')

        e0 = self.conv2(x)
        e1 = self.ab1(x)
        e2 = self.ab6(x)
        e3 = self.ab12(x)
        e4 = self.ab18(x)

        y = self.out(torch.cat([fea_e0, e0, e1, e2, e3, e4, y], dim=1))

        return y


class VecInt(nn.Module):
    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, "nsteps should be >= 0, found: %d" % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):

    def __init__(self, vel_resize):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'trilinear'

    def forward(self, x):
        if self.factor < 1:
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x
        elif self.factor > 1:
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        return x


class Accumulative_Enhancement_1(nn.Module):
    def conv_block(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        return layer

    def mdown_conv(self, in_ch, out_ch, dilation, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                # nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        else:
            layer = nn.Sequential(
                # nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                aspp3d(in_ch, out_ch, stride=2, dilation=dilation),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        return layer

    def fdown_conv(self, in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        return layer
        """
        def outputs(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                    bias=False, batchnorm=False):
            if batchnorm:
                layer = nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                    nn.BatchNorm3d(out_ch),
                    nn.Tanh())
            else:
                layer = nn.Sequential(
                    nn.Conv3d(in_ch, int(in_ch / 2), kernel_size, stride=stride, padding=padding, bias=bias),
                    nn.LeakyReLU(0.2),
                    nn.Conv3d(int(in_ch / 2), out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                )
            return layer
        """

    def __init__(self, in_ch, out_ch, encoder, basic_channel=16, is_train=True, imgshape=(160, 192, 160)):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.bs_ch = basic_channel
        self.imgshape = imgshape
        bias_opt = True

        # self.range_flow = range_flow
        super(Accumulative_Enhancement_1, self).__init__()
        self.is_train = is_train

        #dilation = 4

        #self.mec1 = aspp3d(1, self.bs_ch, stride=2, dilation=dilation)
        #self.mec2 = aspp3d(self.bs_ch, self.bs_ch * 2, stride=2, dilation=dilation - 1)
        #self.mec3 = aspp3d(self.bs_ch * 2, self.bs_ch * 2, stride=2, dilation=dilation - 2)
        #self.mec4 = aspp3d(self.bs_ch * 2, self.bs_ch * 2, stride=2, dilation=dilation - 3)

        self.mec1, self.mec2, self.mec3, self.mec4, self.fec1, self.fec2,self.fec3, self.fec4, self.fusion1, self.fusion2, self.fusion3, self.fusion4  = encoder

        self.dc0 = self.conv_block(self.bs_ch * 2, self.bs_ch * 2)

        self.dc1 = self.conv_block(self.bs_ch * 2 + self.bs_ch * 2, self.bs_ch * 2)
        # self.dc2 = self.conv_block(self.bs_ch * 2 + self.bs_ch * 2, self.bs_ch * 2)
        # self.dc3 = self.conv_block(self.bs_ch * 2 + self.bs_ch, self.bs_ch * 2)

        self.dc4 = self.conv_block(self.bs_ch * 2, self.bs_ch * 2)
        self.dc5 = self.conv_block(self.bs_ch * 2, self.bs_ch)
        self.dc6 = self.conv_block(self.bs_ch + self.in_ch, self.bs_ch)

        self.output = nn.Conv3d(self.bs_ch, 3, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear')
        # self.grid = generate_grid_unit(self.imgshape)
        # self.grid = torch.from_numpy(np.reshape(self.grid, (1,) + self.grid.shape)).cuda().float()
        nd = Normal(0, 1e-5)
        self.output.weight = nn.Parameter(nd.sample(self.output.weight.shape))

        self.output.bias = nn.Parameter(torch.zeros(self.output.bias.shape))

        self.transform = SpatialTransformer(size=self.imgshape, mode='bilinear')
        self.resize = ResizeTransform(vel_resize=1 / 4)

    # self.diff_transform = DiffeomorphicTransform_unit(time_step=7).cuda()
    # self.transform = SpatialTransform_unit().cuda()

    def forward(self, x, y):
        x_in = torch.cat((x, y), 1)

        down_x_in = F.avg_pool3d(x_in, kernel_size=2, stride=2)
        down_down_x_in = F.avg_pool3d(down_x_in, kernel_size=2, stride=2)

        fe1 = self.fec1(y)
        fe2 = self.fec2(fe1)
        fe3 = self.fec3(fe2)
        fe4 = self.fec4(fe3)

        me1 = F.leaky_relu(self.mec1(x, fe1), negative_slope=0.2)
        me2 = F.leaky_relu(self.mec2(me1, fe2), negative_slope=0.2)
        me3 = F.leaky_relu(self.mec3(me2, fe3), negative_slope=0.2)
        me4 = F.leaky_relu(self.mec4(me3, fe4), negative_slope=0.2)

        e4 = self.fusion4(torch.cat([me4, fe4], dim=1))
        e3 = self.fusion3(torch.cat([me3, fe3], dim=1))
        e2 = self.fusion2(torch.cat([me2, fe2], dim=1))
        e1 = self.fusion1(torch.cat([me1, fe1], dim=1))

        d0 = self.dc0(e4)

        d1 = self.dc1(torch.cat((self.up(d0), e3), 1))
        # d2 = self.dc2(torch.cat((self.up(d1), e2), 1))
        # d3 = self.dc3(torch.cat((self.up(d2), e1), 1))

        d2 = self.dc4(d1)
        d3 = self.dc5(d2)
        d6 = self.dc6(torch.cat((self.up(d3), down_down_x_in), 1))

        out_v = self.output(d6)
        # out_flow = self.diff_transform(out_v, self.grid)
        out_v = self.resize(out_v)
        out_flow = out_v
        warp = self.transform(x, out_flow)

        if self.is_train:
            return out_flow, warp, out_flow
        else:
            return out_flow


class Accumulative_Enhancement_2(nn.Module):
    def conv_block(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        return layer

    def mdown_conv(self, in_ch, out_ch, dilation, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                # nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        else:
            layer = nn.Sequential(
                # nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                aspp3d(in_ch, out_ch, stride=2, dilation=dilation),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        return layer

    def fdown_conv(self, in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        return layer
        """
        def outputs(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                    bias=False, batchnorm=False):
            if batchnorm:
                layer = nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                    nn.BatchNorm3d(out_ch),
                    nn.Tanh())
            else:
                layer = nn.Sequential(
                    nn.Conv3d(in_ch, int(in_ch / 2), kernel_size, stride=stride, padding=padding, bias=bias),
                    nn.LeakyReLU(0.2),
                    nn.Conv3d(int(in_ch / 2), out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                )
            return layer
        """

    def __init__(self, in_ch, out_ch, encoder, basic_channel=16, is_train=True, imgshape=(160, 192, 160)):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.bs_ch = basic_channel
        self.imgshape = imgshape
        bias_opt = True

        # self.range_flow = range_flow
        super(Accumulative_Enhancement_2, self).__init__()
        self.is_train = is_train

        #dilation = 4

        #self.mec1 = aspp3d(1, self.bs_ch, stride=2, dilation=dilation)
        #self.mec2 = aspp3d(self.bs_ch, self.bs_ch * 2, stride=2, dilation=dilation - 1)
        #self.mec3 = aspp3d(self.bs_ch * 2, self.bs_ch * 2, stride=2, dilation=dilation - 2)
        #self.mec4 = aspp3d(self.bs_ch * 2, self.bs_ch * 2, stride=2, dilation=dilation - 3)


        self.mec1, self.mec2, self.mec3, self.mec4, self.fec1, self.fec2,self.fec3, self.fec4, self.fusion1, self.fusion2, self.fusion3, self.fusion4  = encoder

        self.dc0 = self.conv_block(self.bs_ch * 2, self.bs_ch * 2)

        self.dc1 = self.conv_block(self.bs_ch * 2 + self.bs_ch * 2, self.bs_ch * 2)
        self.dc2 = self.conv_block(self.bs_ch * 2 + self.bs_ch * 2, self.bs_ch * 2)
        # self.dc3 = self.conv_block(self.bs_ch * 2 + self.bs_ch, self.bs_ch * 2)

        self.dc4 = self.conv_block(self.bs_ch * 2, self.bs_ch * 2)
        self.dc5 = self.conv_block(self.bs_ch * 2, self.bs_ch)
        self.dc6 = self.conv_block(self.bs_ch + self.in_ch, self.bs_ch)

        self.output = nn.Conv3d(self.bs_ch, 3, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear')
        # self.grid = generate_grid_unit(self.imgshape)
        # self.grid = torch.from_numpy(np.reshape(self.grid, (1,) + self.grid.shape)).cuda().float()
        nd = Normal(0, 1e-5)
        self.output.weight = nn.Parameter(nd.sample(self.output.weight.shape))

        self.output.bias = nn.Parameter(torch.zeros(self.output.bias.shape))

        self.transform = SpatialTransformer(size=self.imgshape, mode='bilinear')
        self.resize = ResizeTransform(vel_resize=1 / 2)

    # self.diff_transform = DiffeomorphicTransform_unit(time_step=7).cuda()
    # self.transform = SpatialTransform_unit().cuda()

    def forward(self, x, y):
        x_in = torch.cat((x, y), 1)

        down_x_in = F.avg_pool3d(x_in, kernel_size=2, stride=2)
        # down_down_x_in = F.avg_pool3d(down_x_in, kernel_size=2, stride=2)

        fe1 = self.fec1(y)
        fe2 = self.fec2(fe1)
        fe3 = self.fec3(fe2)
        fe4 = self.fec4(fe3)

        me1 = F.leaky_relu(self.mec1(x, fe1), negative_slope=0.2)
        me2 = F.leaky_relu(self.mec2(me1, fe2), negative_slope=0.2)
        me3 = F.leaky_relu(self.mec3(me2, fe3), negative_slope=0.2)
        me4 = F.leaky_relu(self.mec4(me3, fe4), negative_slope=0.2)

        e4 = self.fusion4(torch.cat([me4, fe4], dim=1))
        e3 = self.fusion3(torch.cat([me3, fe3], dim=1))
        e2 = self.fusion2(torch.cat([me2, fe2], dim=1))
        e1 = self.fusion1(torch.cat([me1, fe1], dim=1))

        d0 = self.dc0(e4)

        d1 = self.dc1(torch.cat((self.up(d0), e3), 1))
        d2 = self.dc2(torch.cat((self.up(d1), e2), 1))
        # d3 = self.dc3(torch.cat((self.up(d2), e1), 1))

        d3 = self.dc4(d2)
        d4 = self.dc5(d3)
        d6 = self.dc6(torch.cat((self.up(d4), down_x_in), 1))

        out_v = self.output(d6)
        # out_flow = self.diff_transform(out_v, self.grid)
        out_v = self.resize(out_v)
        out_flow = out_v
        warp = self.transform(x, out_flow)

        if self.is_train:
            return out_flow, warp, out_flow
        else:
            return out_flow


class Accumulative_Enhancement_3(nn.Module):
    def conv_block(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        return layer

    def mdown_conv(self, in_ch, out_ch, dilation, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                # nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        else:
            layer = nn.Sequential(
                # nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                aspp3d(in_ch, out_ch, stride=2, dilation=dilation),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        return layer

    def fdown_conv(self, in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        return layer
        """
        def outputs(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                    bias=False, batchnorm=False):
            if batchnorm:
                layer = nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                    nn.BatchNorm3d(out_ch),
                    nn.Tanh())
            else:
                layer = nn.Sequential(
                    nn.Conv3d(in_ch, int(in_ch / 2), kernel_size, stride=stride, padding=padding, bias=bias),
                    nn.LeakyReLU(0.2),
                    nn.Conv3d(int(in_ch / 2), out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                )
            return layer
        """

    def __init__(self, in_ch, out_ch, encoder, basic_channel=16, is_train=True, imgshape=(160, 192, 160)):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.bs_ch = basic_channel
        self.imgshape = imgshape
        bias_opt = True

        # self.range_flow = range_flow
        super(Accumulative_Enhancement_3, self).__init__()
        self.is_train = is_train

        #dilation = 4

        #self.mec1 = aspp3d(1, self.bs_ch, stride=2, dilation=dilation)
        #self.mec2 = aspp3d(self.bs_ch, self.bs_ch * 2, stride=2, dilation=dilation - 1)
        #self.mec3 = aspp3d(self.bs_ch * 2, self.bs_ch * 2, stride=2, dilation=dilation - 2)
        #self.mec4 = aspp3d(self.bs_ch * 2, self.bs_ch * 2, stride=2, dilation=dilation - 3)

        self.mec1, self.mec2, self.mec3, self.mec4, self.fec1, self.fec2,self.fec3, self.fec4, self.fusion1, self.fusion2, self.fusion3, self.fusion4  = encoder


        self.dc0 = self.conv_block(self.bs_ch * 2, self.bs_ch * 2)

        self.dc1 = self.conv_block(self.bs_ch * 2 + self.bs_ch * 2, self.bs_ch * 2)
        self.dc2 = self.conv_block(self.bs_ch * 2 + self.bs_ch * 2, self.bs_ch * 2)
        self.dc3 = self.conv_block(self.bs_ch * 2 + self.bs_ch, self.bs_ch * 2)

        self.dc4 = self.conv_block(self.bs_ch * 2, self.bs_ch * 2)
        self.dc5 = self.conv_block(self.bs_ch * 2, self.bs_ch)
        self.dc6 = self.conv_block(self.bs_ch + self.in_ch, self.bs_ch)

        self.output = nn.Conv3d(self.bs_ch, 3, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear')
        # self.grid = generate_grid_unit(self.imgshape)
        # self.grid = torch.from_numpy(np.reshape(self.grid, (1,) + self.grid.shape)).cuda().float()
        nd = Normal(0, 1e-5)
        self.output.weight = nn.Parameter(nd.sample(self.output.weight.shape))

        self.output.bias = nn.Parameter(torch.zeros(self.output.bias.shape))

        self.transform = SpatialTransformer(size=self.imgshape, mode='bilinear')
        self.resize = ResizeTransform(vel_resize=1)

    # self.diff_transform = DiffeomorphicTransform_unit(time_step=7).cuda()
    # self.transform = SpatialTransform_unit().cuda()

    def forward(self, x, y):
        x_in = torch.cat((x, y), 1)

        # down_x_in = F.avg_pool3d(x_in, kernel_size=2, stride=2)
        # down_down_x_in = F.avg_pool3d(down_x_in, kernel_size=2, stride=2)

        fe1 = self.fec1(y)
        fe2 = self.fec2(fe1)
        fe3 = self.fec3(fe2)
        fe4 = self.fec4(fe3)

        me1 = F.leaky_relu(self.mec1(x, fe1), negative_slope=0.2)
        me2 = F.leaky_relu(self.mec2(me1, fe2), negative_slope=0.2)
        me3 = F.leaky_relu(self.mec3(me2, fe3), negative_slope=0.2)
        me4 = F.leaky_relu(self.mec4(me3, fe4), negative_slope=0.2)

        e4 = self.fusion4(torch.cat([me4, fe4], dim=1))
        e3 = self.fusion3(torch.cat([me3, fe3], dim=1))
        e2 = self.fusion2(torch.cat([me2, fe2], dim=1))
        e1 = self.fusion1(torch.cat([me1, fe1], dim=1))

        d0 = self.dc0(e4)

        d1 = self.dc1(torch.cat((self.up(d0), e3), 1))
        d2 = self.dc2(torch.cat((self.up(d1), e2), 1))
        d3 = self.dc3(torch.cat((self.up(d2), e1), 1))

        d4 = self.dc4(d3)
        d5 = self.dc5(d4)
        d6 = self.dc6(torch.cat((self.up(d5), x_in), 1))

        out_v = self.output(d6)
        # out_flow = self.diff_transform(out_v, self.grid)
        out_v = self.resize(out_v)
        out_flow = out_v
        warp = self.transform(x, out_flow)

        if self.is_train:
            return out_flow, warp, out_flow
        else:
            return out_flow


class ULAE(nn.Module):

    def conv_block(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        return layer

    def mdown_conv(self, in_ch, out_ch, dilation, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                # nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        else:
            layer = nn.Sequential(
                # nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                aspp3d(in_ch, out_ch, stride=2, dilation=dilation),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        return layer

    def fdown_conv(self, in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.2)
            )
        return layer
    def __init__(self, in_ch, out_ch, basic_channel=16, is_train=True, imgshape=(160, 192, 160)):
        super(ULAE, self).__init__()
        
        dilation = 4
        mec1 = Uncoupled_Encoding_Block(1, basic_channel, stride=2, dilation=dilation)
        mec2 = Uncoupled_Encoding_Block(basic_channel, basic_channel * 2, stride=2, dilation=dilation - 1)
        mec3 = Uncoupled_Encoding_Block(basic_channel * 2, basic_channel * 2, stride=2, dilation=dilation - 2)
        mec4 = Uncoupled_Encoding_Block(basic_channel * 2, basic_channel * 2, stride=2, dilation=dilation - 3)
        
        fec1 = self.fdown_conv(1, basic_channel)
        fec2 = self.fdown_conv(basic_channel, basic_channel * 2)
        fec3 = self.fdown_conv(basic_channel * 2, basic_channel * 2)
        fec4 = self.fdown_conv(basic_channel * 2, basic_channel * 2)

        fusion1 = self.conv_block(basic_channel * 2, basic_channel)
        fusion2 = self.conv_block(basic_channel * 4, basic_channel * 2)
        fusion3 = self.conv_block(basic_channel * 4, basic_channel * 2)
        fusion4 = self.conv_block(basic_channel * 4, basic_channel * 2)
        
        self.share_weight_encoder = [mec1, mec2, mec3, mec4, fec1, fec2, fec3, fec4, fusion1, fusion2, fusion3, fusion4]
        
        self.acc1 = Accumulative_Enhancement_1(in_ch, out_ch, self.share_weight_encoder, basic_channel=basic_channel, is_train=is_train, imgshape=imgshape)
        self.acc2 = Accumulative_Enhancement_2(in_ch, out_ch, self.share_weight_encoder, basic_channel=basic_channel, is_train=is_train, imgshape=imgshape)
        self.acc3 = Accumulative_Enhancement_3(in_ch, out_ch, self.share_weight_encoder, basic_channel=basic_channel, is_train=is_train, imgshape=imgshape)

        self.transformer = SpatialTransformer(size=imgshape)


    def forward(self, x, y):
        flow1, warp1, _= self.acc1(x, y)
        flow2, warp2, _ = self.acc2(warp1, y)
        flow3, warp3, _ = self.acc3(warp2, y)

        flowa = flow1
        flowb = self.transformer(flowa, flow2) + flow2
        flowc = self.transformer(flowb, flow3) + flow3

        return [warp1, warp2, warp3], [flowa, flowb, flowc], [flow1, flow2, flow3]

if __name__=="__main__":
    model = ULAE(2, 3, 16, True, [160,192,160]).cuda()
    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)
