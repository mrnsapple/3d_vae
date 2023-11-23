from os import listdir
from os.path import isfile, join

import numpy as np


def get_data(path):
    files = [
        join(path, f)
        for f in listdir(path)
        if isfile(join(path, f)) and f.endswith("npy")
    ]
    data = []
    for file in files:
        data.append([np.load(file)])
    data = np.concatenate([data], axis=0)
    return data


# get_data("/home/oriol/tools/3d_vae/data/raw")


def data_loader(fname, input_shape):
    xc = get_data(fname)
    # xc = np.zeros((reader.length(),) + input_shape, dtype=np.float32)
    # reader.reopen()
    # for ix, (x, _name) in enumerate(reader):
    #    xc[ix] = x.astype(np.float32)
    return 3.0 * xc - 1.0
