"Contains core function for image handling"
import bz2
import pickle
import sys
import threading
from getopt import getopt
from math import log

import PIL.Image as I
import matplotlib.pyplot as plt

import numpy as N
import numpy.linalg as LA

__author__ = 'David Berthelot'


class ClassDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def save_py(obj, fn, zipped=False):
    print('Saving "' + fn + '" ...', end='')
    sys.stdout.flush()
    if zipped:
        fn += 'z'
    if zipped:
        pickle.dump(obj, bz2.BZ2File(fn, 'w'), pickle.HIGHEST_PROTOCOL)
    else:
        pickle.dump(obj, open(fn, 'wb'), pickle.HIGHEST_PROTOCOL)
    print('done.')


def load_py(fn, zipped=False):
    if zipped:
        fn += 'z'
    if zipped:
        f = bz2.BZ2File(fn, 'r')
    else:
        f = open(fn, 'rb')
    print('Loading "' + fn + '" ...', end='')
    sys.stdout.flush()
    x = pickle.load(f)
    print('done.')
    return x


def pm(f, p=1):
    t = "%." + str(p) + "f"
    for y in range(f.shape[0]):
        for x in range(f.shape[1]):
            print("%6s" % (t % (f[y, x],),), end='')
        print()


def MAE(a, b):
    if isinstance(a, Img):
        a = a.m
    if isinstance(b, Img):
        b = b.m
    return N.abs(a - b).mean()


def RMSE(a, b):
    if isinstance(a, Img):
        a = a.m
    if isinstance(b, Img):
        b = b.m
    return N.sqrt(N.square(a - b).mean())


def rmse_str(rmse):
    return ('%.4f' % rmse).rjust(7)


def PSNR(a, b):
    rmse = RMSE(a.to_ycbcr()[:, :, 0], b.to_ycbcr()[:, :, 0])
    psnr = 20 * log(255. / rmse) / log(10)
    return psnr


def safe_inv(x):
    nx = x.copy()
    nx[x == 0] = 1
    return 1. / nx


class thread_function_launcher(threading.Thread):
    def __init__(self, f, *args):
        threading.Thread.__init__(self)
        self.f = f
        self.args = args

    def run(self):
        self.f(*self.args)


class Img:
    """Img class represents images. (y, x, colors): supports 1, 3 or 4 color channels."""
    _YCbCr = N.array(((0.299, 0.587, 0.114),
                      (-0.168736, -0.331264, 0.5),
                      (0.5, -0.418688, -0.081312)), 'f').T.copy()
    _YCbCri = LA.inv(_YCbCr).astype('f')

    def __init__(self, m):
        self.m = m if m.ndim == 3 else m.reshape(m.shape + (1,))

    @classmethod
    def from_img(cls, i):
        if i.mode == 'L':
            i = i.convert('RGB')
        return cls(N.array(N.asarray(i), 'f'))

    def to_img(self):
        if self.m.shape[2] >= 3:
            i = I.frombytes('RGB' if self.m.shape[2] == 3 else 'RGBA',
                            (self.m.shape[1], self.m.shape[0]),
                            N.array(N.round(self.m).clip(0, 255), 'uint8').tostring())
        else:
            i = I.fromarray(N.array(N.round(self.m.reshape(self.m.shape[:2])).clip(0, 255), 'uint8'))
        return i

    @classmethod
    def load(cls, fn):
        return cls.from_img(I.open(fn))

    def save(self, fn='test.bmp', q=None):
        i = self.to_img()
        if q is None:
            i.save(fn)
        else:
            i.save(fn, quality=q)

    def show(self):
        plt.imshow(N.clip(self.m / 255., 0, 1), interpolation='nearest')
        plt.ion()
        plt.show()

    def box(self, scale):
        f = self.mod_crop(scale).m
        f = f.reshape((f.shape[0] // scale, scale, f.shape[1] // scale, scale, -1))
        f = f.mean(3).mean(1)
        return Img(f)

    def box2(self):
        return self.box(2)

    def box3(self):
        return self.box(3)

    def unbox(self, n=2):
        shape = list(self.m.shape)
        s = N.zeros([shape[0], n, shape[1], n] + shape[2:], self.m.dtype)
        s += self.m[:, None, :, None]
        return Img(s.reshape((shape[0] * n, shape[1] * n, -1)))

    def collar(self, w):
        mshape = N.array(self.m.shape, 'uint32')
        mshape[:2] += 2 * w
        m = N.zeros(mshape, self.m.dtype)
        m[w:-w, w:-w] = self.m
        for x in range(w):
            m[w:-w, x] = m[w:-w, w]
            m[w:-w, -x - 1] = m[w:-w, -w - 1]
            m[x, w:-w] = m[w, w:-w]
            m[-x - 1, w:-w] = m[-w - 1, w:-w]
        m[:w, :w] = m[w, w]
        m[:w, -w:] = m[w, -w - 1]
        m[-w:, :w] = m[-w - 1, w]
        m[-w:, -w:] = m[-w - 1, -w - 1]
        return Img(m)

    def mod_crop(self, mod=2):
        return Img(self.m[:self.shape[0] - (self.shape[0] % mod), :self.shape[1] - (self.shape[1] % mod)])

    def mod_pad(self, mod=2):
        s = list(self.shape)
        if self.shape[0] % mod:
            s[0] = self.shape[0] + mod - (self.shape[0] % mod)
        if self.shape[1] % mod:
            s[1] = self.shape[1] + mod - (self.shape[1] % mod)
        m = N.zeros(s, self.m.dtype)
        s = list(self.shape)
        m[:s[0], :s[1]] = self.m
        return Img(m)

    def bw(self):
        """Black and white"""
        return Img(N.sqrt(N.square(self.m).mean(2)))

    def to_ycbcr(self):
        assert self.m.ndim == 3
        m = N.dot(self.m - 128, self._YCbCr)
        return Img(m)

    def to_rgb(self):
        assert self.m.ndim == 3
        return Img(N.dot(self.m, self._YCbCri) + 128)

    @classmethod
    def concat(cls, *images, axis=1):
        l = [i.m if isinstance(i, cls) else i for i in images]
        return cls(N.concatenate(l, axis=axis))

    @property
    def shape(self):
        return self.m.shape

    @property
    def size(self):
        return self.m.size

    def __getitem__(self, *args):
        return self.m.__getitem__(*args)


class Opts:
    def __init__(self):
        self._opts = {}
        for (opt, value) in getopt(sys.argv[1:], '', ['opts='])[0]:
            if opt == '--opts':
                self._opts = {x: y for x, y in [x.split('=') for x in value.split(',')]}

    def __getattr__(self, item):
        def f(cast, default):
            cast = cast or (lambda x: x)
            return cast(self._opts[item]) if item in self._opts else default

        return f


OPTS = Opts()
