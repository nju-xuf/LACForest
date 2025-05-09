import h5py
import numpy as np
from scipy.sparse import csc_matrix as spmtx
from scipy.io import loadmat


def read_from_hdf(path, keys):
    f = h5py.File(path, 'r')
    result = {}
    for key in keys:
        data_np = None
        if key not in f.keys():
            print('{} does not exists.'.format(key))
        else:
            data = f[key]
            if isinstance(data, h5py.Dataset):
                data_np = np.empty(data.shape, data.dtype)
                data.read_direct(data_np)
            elif isinstance(data, h5py.Group):
                data_np = spmtx((data['data'], data['ir'], data['jc'])).toarray()
            else:
                print('Unkown Type: {}'.format(type(data)))
        result[key] = data_np
    f.close()

    return result


def read_from_mat(path, keys):
    f = loadmat(path)
    result = {}
    for key in keys:
        data_np = None
        if key not in f.keys():
            print('{} does not exists.'.format(key))
        else:
            data = f[key]
            if isinstance(data, spmtx):
                data_np = data.toarray()
            elif isinstance(data, np.ndarray):
                data_np = data
            else:
                print('Unkown Type: {}'.format(type(data)))
        result[key] = data_np
    return result

