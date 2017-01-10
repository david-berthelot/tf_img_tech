"""DataSet classes for training"""

import h5py
from tqdm import tqdm

import os
from core import *

__author__ = 'David Berthelot'


class DataSet:
    def sample(self):
        return self


class DataSetImageHomoSeries(DataSet):
    """Generate dataset of homogeneous series of images."""

    def __init__(self, filenames, xvalidate, ds, in_memory=False):
        assert isinstance(filenames, dict)
        assert isinstance(xvalidate, dict)
        self.filenames = filenames
        self.xvalidate = xvalidate
        self.ds = ds if not in_memory else {x: ds[x][:] for x in ds.keys()}
        self.len = sum(len(x) for x in self.ds.values())

    def __len__(self):
        return self.len

    @classmethod
    def create(cls, path, filenames, xvalidate, prefix=''):
        filepath = os.path.join(path, 'train_%s%s_f%d' % (prefix, cls.__name__.lower(), len(filenames)))
        try:
            os.mkdir(filepath)
        except FileExistsError:
            pass
        print('Analyzing samples')
        target_size = None
        all_filenames = []
        for x in filenames.values():
            all_filenames.extend(x)
        for f in tqdm(all_filenames):
            g = Img.load(f)
            if target_size is None:
                target_size = g.shape[:-1]
            assert g.shape[-1] == 3
            assert target_size == g.shape[:-1]

        args = dict(filenames=filenames, xvalidate=xvalidate)
        save_py((cls.__name__, args), os.path.join(filepath, 'params.p'))

        print('%s Building dataset  Target size = %.1fMB' % (cls.__name__, (target_size[0] * target_size[1] * len(all_filenames)) / (1 << 20)))
        with h5py.File(os.path.join(filepath, 'data.hdf5'), 'w') as ds:
            for group, filelist in filenames.items():
                an = ds.create_dataset(group, (len(filelist), *target_size), dtype='uint8')
                for x, f in enumerate(tqdm(filelist, desc=group)):
                    an[x] = N.round(128 + Img.load(f).to_ycbcr().m[:, :, 0]).astype('uint8')
        return cls.load(filepath)

    def __getitem__(self, s):
        return self.ds[s]

    @classmethod
    def load(cls, filepath, in_memory=False):
        clsname, args = load_py(os.path.join(filepath, 'params.p'))
        assert clsname == cls.__name__
        ds = h5py.File(os.path.join(filepath, 'data.hdf5'), 'r')
        return cls(ds=ds, in_memory=in_memory, **args)


class DataSetImageHomoSeriesRGB(DataSetImageHomoSeries):
    """Generate RGB dataset of homogeneous series of images."""

    @classmethod
    def create(cls, path, filenames, xvalidate, prefix=''):
        filepath = os.path.join(path, 'train_%s%s_f%d' % (prefix, cls.__name__.lower(), len(filenames)))
        try:
            os.mkdir(filepath)
        except FileExistsError:
            pass
        print('Analyzing samples')
        target_size = None
        all_filenames = []
        for x in filenames.values():
            all_filenames.extend(x)
        for f in tqdm(all_filenames):
            g = Img.load(f)
            if target_size is None:
                target_size = g.shape
            assert g.shape[-1] == 3
            assert target_size == g.shape

        args = dict(filenames=filenames, xvalidate=xvalidate)
        save_py((cls.__name__, args), os.path.join(filepath, 'params.p'))

        print('%s Building dataset  Target size = %.1fMB' % (cls.__name__, (target_size[0] * target_size[1] * target_size[2] * len(all_filenames)) / (1 << 20)))
        with h5py.File(os.path.join(filepath, 'data.hdf5'), 'w') as ds:
            for group, filelist in filenames.items():
                an = ds.create_dataset(group, (len(filelist), *target_size), dtype='uint8')
                for x, f in enumerate(tqdm(filelist, desc=group)):
                    an[x] = N.round(Img.load(f).m).astype('uint8')
        return cls.load(filepath)


def patch_xvalidate_path(ds, source, target):
    ds.xvalidate = tuple(x.replace(source, target) for x in ds.xvalidate)


# DataSetImageHomoSeries
if __name__ == '__main__' and OPTS.celeba(int, 0):
    from glob import glob

    files = sorted(glob('Train/celeba/img_align_celeba/*.jpg'))
    xvalidate = set()
    with open('Train/celeba/list_eval_partition.txt', 'r') as f:
        for l in f:
            fn, iseval = l.strip().split()
            if iseval == '2':
                xvalidate.add(fn)
    xvalidate_new = []
    for x, fn in enumerate(files):
        if os.path.basename(fn) in xvalidate:
            xvalidate_new.append(x)
    xvalidate = N.array(xvalidate_new, 'int')
    print('Full size %d  Validation size %d' % (len(files), len(xvalidate)))
    filenames = dict(faces=files)
    xvalidate = dict(faces=xvalidate)
    ds = DataSetImageHomoSeriesRGB.create('Train/celeba', filenames, xvalidate, 'celeba_')
