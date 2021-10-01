import os, glob, sys
from argparse import ArgumentParser
import numpy as np
import torch

# from Transforms import *
from datagen import *
from Loss import *
from ULAE import ULAE, SpatialTransformer
parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=160001,
                    help="number of total iterations")
parser.add_argument("--local_ori", type=float,
                    dest="local_ori", default=0,
                    help="Local Orientation Consistency loss: suggested range 1 to 1000")
parser.add_argument("--smooth", type=float,
                    dest="smooth", default=1.0,
                    help="Gradient smooth loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=4000,
                    help="frequency of saving models")
parser.add_argument("--bs_ch", type=int,
                    dest="bs_ch", default=16,
                    help="number of basic channels")
parser.add_argument("--modelname", type=str,
                    dest="model_name",
                    default='ULAE',
                    help="Name for saving")
parser.add_argument("--gpu", type=str,
                    dest="gpu",
                    default='1',
                    help="gpus")
parser.add_argument("--gamma", type=float,
                    dest="gamma",
                    default='0.5',
                    help="hype-param for mutiwindow loss")               
opt = parser.parse_args()

lr = opt.lr
bs_ch = opt.bs_ch
local_ori = opt.local_ori
n_checkpoint = opt.checkpoint
smooth = opt.smooth
model_name = opt.model_name
iteration = opt.iteration
gamma = opt.gamma
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

train_vol_names = glob.glob(os.path.join("./data/train/", "*vol.npy"))
train_seg_names = glob.glob(os.path.join("./data/train/", "*seg.npy"))
train_vol_names.sort()
train_seg_names.sort()
dg = scan_to_scan(train_vol_names, train_seg_names, batch_size=1)

imgshape = next(dg)[0][0].shape[2:]





def train_ULAE():

    model = ULAE(2, 3, bs_ch, True, imgshape).cuda()


    sim_loss_fn = multi_window_loss(win=[11,9,7], gamma=gamma)
    smo_loss_fn = smoothloss
    jac_loss_fn = neg_Jdet_loss

    transform = SpatialTransformer(size=imgshape).cuda()
    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model_dir = './Model/'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    loss_all = np.zeros((5, iteration + 1))

    step = 0

    load_model = False

    if load_model is True:
        model_path = 'path for your model.pth'
        step = 'step'
        model.load_state_dict(torch.load(model_path))
        loss_load = np.load("path for your loss.npy")
        loss_all[:, :step] = loss_load[:, :step]

    while step <= iteration:
        for Data, Seg in dg:
            X, Y = Data

            X = torch.from_numpy(X).cuda().float()
            Y = torch.from_numpy(Y).cuda().float()

            warps, flows, _ = model(X, Y)

            sim_loss_1, sim_loss_2, sim_loss_3 = sim_loss_fn(warps[2], Y)
            smo_loss = smo_loss_fn(flows[-1])

            loss = sim_loss_1 + sim_loss_2 + sim_loss_3 + smooth * smo_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_all[:, step] = np.array(
                [loss.item(), sim_loss_1.item(), sim_loss_2.item(), sim_loss_3.item(), smo_loss.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_muti_window_NCC "{2:4f} {3:4f} {4:4f}" - smo "{5:.4f}"'.format(
                    step, loss.item(), sim_loss_1.item(), sim_loss_2.item(), sim_loss_3.item(), smo_loss.item()))
            sys.stdout.flush()

            if (step % n_checkpoint == 0):
                modelname = model_dir + '/' + model_name + str(bs_ch) + str(smooth).replace('.', '_') + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss' + model_name + str(bs_ch) + str(smooth).replace('.', '_') + str(step) + '.npy', loss_all)


            step += 1

            if step > iteration:
                break
        np.save(model_dir + '/loss' + model_name + str(bs_ch) + str(smooth).replace('.', '_') +  str(step) + '.npy', loss_all)


if __name__ == '__main__':
    train_ULAE()
