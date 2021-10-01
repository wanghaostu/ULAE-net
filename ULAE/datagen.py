
import os
import sys
import glob
import numpy as np



def volgen(
        vol_names,
        seg_names,
        batch_size=1,
        return_segs=False,
    ):
    """
    Base generator for random volume loading. Volumes can be passed as a path to
    the parent directory, a glob pattern or a list of file paths. Corresponding
    segmentations are additionally loaded if return_segs is set to True. If
    loading segmentations, npz files with variable names 'vol' and 'seg' are
    expected.
    Parameters:
        vol_names: Path, glob pattern or list of volume files to load.
        batch_size: Batch size. Default is 1.
        return_segs: Loads corresponding segmentations. Default is False.
        np_var: Name of the volume variable if loading npz files. Default is 'vol'.
        pad_shape: Zero-pads loaded volumes to a given shape. Default is None.
        resize_factor: Volume resize factor. Default is 1.
        add_feat_axis: Load volume arrays with added feature axis. Default is True.
    """

    # convert glob path to filenames
    if isinstance(vol_names, str):
        if os.path.isdir(vol_names):
            vol_names = os.path.join(vol_names, '*')
        vol_names = glob.glob(vol_names)

    while True:

        indices = np.random.randint(len(vol_names), size=batch_size)

        imgs = [np.load(vol_names[i]) for i in indices]
        vols = [np.concatenate(imgs, axis=0)]

        if return_segs:

            segs = [np.load(seg_names[i]) for i in indices]
            vols.append(np.concatenate(segs, axis=0))

        yield tuple(vols)

def scan_to_scan(vol_names, seg_names, batch_size=1, return_segs=False):
    """
    Generator for scan-to-scan registration.
    Parameters:
        vol_names: List of volume files to load.
        bidir: Yield input image as output for bidirectional models. Default is False.
        batch_size: Batch size. Default is 1.
        prob_same: Induced probability that source and target inputs are the same. Default is 0.
        no_warp: Excludes null warp in output list if set to True (for affine training). Default if False.
        kwargs: Forwarded to the internal volgen generator.
    """

    gen = volgen(vol_names, seg_names, batch_size=batch_size, return_segs=return_segs)
    while True:
        scan1 = next(gen)[0]
        scan2 = next(gen)[0]

        invols  = [scan1, scan2]
        outvols = [scan2]

        yield (invols, outvols)


def volgen_seg(
        vol_names,
        seg_names,
        batch_size=1,
    ):
    """
    Base generator for random volume loading. Volumes can be passed as a path to
    the parent directory, a glob pattern or a list of file paths. Corresponding
    segmentations are additionally loaded if return_segs is set to True. If
    loading segmentations, npz files with variable names 'vol' and 'seg' are
    expected.
    Parameters:
        vol_names: Path, glob pattern or list of volume files to load.
        batch_size: Batch size. Default is 1.
        return_segs: Loads corresponding segmentations. Default is False.
        np_var: Name of the volume variable if loading npz files. Default is 'vol'.
        pad_shape: Zero-pads loaded volumes to a given shape. Default is None.
        resize_factor: Volume resize factor. Default is 1.
        add_feat_axis: Load volume arrays with added feature axis. Default is True.
    """

    # convert glob path to filenames
    if isinstance(vol_names, str):
        if os.path.isdir(vol_names):
            vol_names = os.path.join(vol_names, '*')
        vol_names = glob.glob(vol_names)

    while True:
    
        indices = np.random.randint(len(vol_names), size=batch_size)

        imgs = [np.load(vol_names[i]) for i in indices]
        vols = [np.concatenate(imgs, axis=0)]
        seg = [np.load(seg_names[i]) for i in indices]
        segs = [np.concatenate(seg, axis=0)] 

        yield vols, segs

def scan_to_scan_return_seg(vol_names, seg_names, batch_size=1, return_segs=True):
    """
    Generator for scan-to-scan registration.
    Parameters:
        vol_names: List of volume files to load.
        bidir: Yield input image as output for bidirectional models. Default is False.
        batch_size: Batch size. Default is 1.
        prob_same: Induced probability that source and target inputs are the same. Default is 0.
        no_warp: Excludes null warp in output list if set to True (for affine training). Default if False.
        kwargs: Forwarded to the internal volgen generator.
    """

    gen = volgen_seg(vol_names, seg_names, batch_size=batch_size)
    while True:
        scan1 = next(gen)
        scan2 = next(gen)

        yield scan1, scan2

if __name__ == "__main__":
    train_vol_names = glob.glob(os.path.join("./data/train/", "*vol.npy"))
    train_seg_names = glob.glob(os.path.join("./data/train/", "*seg.npy"))
    train_vol_names.sort()
    train_seg_names.sort()
    dg = scan_to_scan_return_seg(train_vol_names, train_seg_names, batch_size=1)
    
    print(next(dg)[0][1][0].max())
    print(next(dg)[1][1][0].max())

def semisupervised(vol_names, seg_names, labels, batch_size=1):
    """
    Generator for semi-supervised registration training using ground truth segmentations.
    Scan-to-atlas training can be enabled by providing the atlas_file argument. It's
    expected that vol_names and atlas_file are npz files with both 'vol' and 'seg' arrays.
    Parameters:
        vol_names: List of volume npz files to load.
        labels: Array of discrete label values to use in training.
        atlas_file: Atlas npz file for scan-to-atlas training. Default is None.
        downsize: Downsize factor for segmentations. Default is 2.
    """
    # configure base generator
    gen = volgen(vol_names, seg_names, batch_size=batch_size, return_segs=True)

    # internal utility to generate downsampled prob seg from discrete seg

    def split_seg(seg):
        prob_seg = np.zeros((seg.shape[0], len(labels), seg.shape[2], seg.shape[3], seg.shape[4]))
        for i, label in enumerate(labels):
            prob_seg[:, i, ...] = seg[:, 0, ...] == label
        return prob_seg

    while True:
        # load source vol and seg


        src_vol, src_seg = next(gen)
        src_seg = split_seg(src_seg)
        trg_vol, trg_seg = next(gen)
        trg_seg = split_seg(trg_seg)

        invols  = [src_vol, trg_vol]
        outvols = [trg_vol, src_seg, trg_seg]
        yield (invols, outvols)
