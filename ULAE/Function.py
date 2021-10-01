import nibabel as nib
from torch.utils import data
import itertools
import numpy as np


def dice(vol1, vol2, nargout=1):
    labels = [1002, 1003, 1005, 1006, 1007, 1008, 1009, 1011,
              1012, 1013, 1014, 1015, 1016, 1017, 1018, 1021, 1022,
              1024, 1025, 1028, 1029, 1030, 1031, 1034, 1035, 2002,
              2003, 2005, 2006, 2007, 2008, 2009, 2011, 2012, 2013,
              2014, 2015, 2016, 2017, 2018, 2021, 2022, 2024, 2025,
              2028, 2029, 2030, 2031, 2034, 2035]
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background
        labels = np.delete(labels, np.where(labels == 181))
        labels = np.delete(labels, np.where(labels == 182))
    dicem = np.zeros(len(labels))

    for idx, lab in enumerate(labels):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return (dicem, labels)

def save_img(I_img,savename):
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    nib.save(new_img, savename)


def save_flow(I_img,savename):
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    nib.save(new_img, savename)


class lbpadataset(data.Dataset):
    def __init__(self, path_to_file="lbpa40.npy", mode="train"):
        self.dataset = np.load(path_to_file, allow_pickle=True)
        self.dataset = self.dataset
        self.train_pair = [j for j in itertools.permutations([i for i in range(0, 30)], 2)]
        self.test_pair = [j for j in itertools.permutations([i for i in range(30, 40)], 2)]
        self.mode = mode
        self.vol_shape = [160, 192, 160]
    def __getitem__(self, idx):
        if self.mode == "train":
            fixed, moving = self.dataset[self.train_pair[idx][0]], self.dataset[self.train_pair[idx][1]]
        else:
            fixed, moving = self.dataset[self.test_pair[idx][0]], self.dataset[self.test_pair[idx][1]]
        return fixed, moving
    def __len__(self):
        if self.mode == "train":
            return len(self.train_pair)
        else:
            return len(self.test_pair)




class mindboggledataset(data.Dataset):

    def __init__(self, mode="train"):
        self.wrongs = ["Extra-18_volumes.npy", "MMRR-21_volumes.npy"]
        self.train_sets = ["NKI-RS-22_volumes.npy", "NKI-TRT-20_volumes.npy"]
        self.test_sets = ["./OASIS-TRT-20_volumes.npy"]
        if mode == "train":
            dataset = [np.load(self.train_sets[i], allow_pickle=True) for i in range(len(self.train_sets))]
        else:
            dataset = [np.load(self.test_sets[i], allow_pickle=True) for i in range(len(self.test_sets))]
        self.dataset = []
        for i in dataset:
            for j in i:
                self.dataset.append(j)
        self.pair = [j for j in itertools.permutations(range(0, len(self.dataset)), 2)]
        self.vol_shape = [160, 192, 160]

    def __getitem__(self, idx):
        fixed, moving = self.dataset[self.pair[idx][0]], self.dataset[self.pair[idx][1]]
        return fixed, moving

    def __len__(self):
        return len(self.pair)
