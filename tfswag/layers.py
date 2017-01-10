import numpy as N
import numpy.linalg as LA
import tensorflow as tf

__author__ = 'David Berthelot'

_YCBCR = N.array(((0.299, 0.587, 0.114),
                  (-0.168736, -0.331264, 0.5),
                  (0.5, -0.418688, -0.081312)), 'f').T.copy()
_YCBCRI = N.linalg.inv(_YCBCR).astype('f')


def leaky_relu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def to_ycbcr(x):
    """4D tensor to YCBCR, assumes input in [0, 1]."""
    s = tf.shape(x)
    return tf.reshape(tf.matmul(tf.reshape(x - 128 / 255, [-1, s[3]]), _YCBCR), s)


def to_rgb(x):
    """4D tensor to RGB, assumes ycbcr input in [-0.5, 0.5]."""
    s = tf.shape(x)
    return tf.reshape(tf.matmul(tf.reshape(x, [-1, s[3]]), _YCBCRI), s) + 128 / 255


def loss_var(v, dims):
    """Regularize v to unit variance w.r.t. batch"""
    v = tf.reshape(v, [-1, dims])
    v_mean = tf.reduce_mean(v, 0, keep_dims=True)
    v_var = tf.reduce_mean(tf.square(v - v_mean), 0)
    return v_var


def zca(v):
    """Zero Component Analysis https://github.com/mwv/zca/blob/master/zca/zca.py"""
    v = v.reshape((v.shape[0], -1))
    m = v.mean(0)
    vm = N.ascontiguousarray((v - m).T)
    # print(vm.shape)
    # cov = N.zeros((vm.shape[0], vm.shape[0]), 'f')
    # for x in range(vm.shape[0]):
    #     for y in range(x, vm.shape[0]):
    #         cov[x, y] = cov[y, x] = vm[x].dot(vm[y])
    # cov /= v.shape[0] - 1
    cov = vm.dot(vm.T) / (v.shape[0] - 1)
    u, s = LA.svd(cov, full_matrices=0)[:2]
    w = (u * (1 / N.sqrt(s.clip(1e-6)))).dot(u.T)
    # dw = (u * (1 / N.sqrt(s))).dot(u.T)  # Dewithen
    return m, w


def loss_bound_weights(model, max_weight=1):
    """Regularize model vars so that values outside of max_weight are penalized."""
    reg_weight = 0
    for x in model.g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        x_ = tf.clip_by_value(x, -max_weight, max_weight)
        reg_weight += tf.reduce_sum(tf.square(x - x_))
    return reg_weight / model.size


def loss_l2(model):
    """Regularize model vars son L2."""
    reg_weight = 0
    for x in model.g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        reg_weight += tf.reduce_sum(tf.square(x))
    return reg_weight / model.size


def loss_orthogonal(e, dims, cross_correlate_only=False):
    """Compute the orthogonal loss."""
    ii = 1 - N.identity(dims, 'f')
    cross_correlation = tf.matmul(tf.transpose(e), e) * ii / tf.to_float(tf.shape(e)[0])
    if cross_correlate_only:
        return cross_correlation
    return tf.reduce_mean(tf.abs(cross_correlation))


def unboxn(vin, n):
    """vin = (batch, h, w, depth), returns vout = (batch, n*h, n*w, depth), each pixel is duplicated."""
    s = tf.shape(vin)
    vout = tf.concat(0, [vin] * (n ** 2))  # Poor man's replacement for tf.tile (required for Adversarial Training support).
    vout = tf.reshape(vout, [s[0] * (n ** 2), s[1], s[2], s[3]])
    vout = tf.batch_to_space(vout, [[0, 0], [0, 0]], n)
    return vout


def boxn(vin, n):
    """vin = (batch, h, w, depth), returns vout = (batch, h//n, w//n, depth), each pixel is averaged."""
    if n == 1:
        return vin
    s = tf.shape(vin)
    vout = tf.reshape(vin, [s[0], s[1] // n, n, s[2] // n, n, s[3]])
    vout = tf.reduce_mean(vout, [2, 4])
    return vout


class LayerBase:
    pass


class LayerWx(LayerBase):
    def __init__(self, name, n, nl=lambda x, y: x + y, w=None):
        """n = (n_in, n_out)"""
        self.nl = nl
        with tf.name_scope(name):
            if w is None:
                w = tf.Variable(tf.truncated_normal(n, stddev=0.01), name='w')
            self.w = w

    def __call__(self, vin):
        return self.nl(tf.matmul(vin, self.w), 0)


class LayerWxB(LayerBase):
    def __init__(self, name, n, nl=lambda x, y: x + y, w=None, use_bias=True):
        """n = (n_in, n_out)"""
        self.nl = nl
        with tf.name_scope(name):
            if w is None:
                w = tf.Variable(tf.truncated_normal(n, stddev=0.01), name='w')
            self.w = w
            self.b = tf.Variable(tf.zeros([n[1]]), name='b') if use_bias else 0

    def __call__(self, vin):
        return self.nl(tf.matmul(vin, self.w), self.b)


class LayerConv(LayerBase):
    def __init__(self, name, w, n, nl=lambda x, y: x + y, strides=(1, 1, 1, 1), padding='SAME', conv=None, use_bias=True):
        """w = (wy, wx), n = (n_in, n_out)"""
        self.nl = nl
        self.strides = list(strides)
        self.padding = padding
        with tf.name_scope(name):
            if conv is None:
                conv = tf.Variable(tf.truncated_normal([*w, *n], stddev=0.01), name='conv')
            self.conv = conv
            self.bias = tf.Variable(tf.zeros([n[1]]), name='bias') if use_bias else 0

    def __call__(self, vin):
        return self.nl(tf.nn.conv2d(vin, self.conv, strides=self.strides, padding=self.padding), self.bias)


class LayerConvDilated(LayerConv):
    def __init__(self, name, w, n, dilation, nl=lambda x, y: x + y, strides=(1, 1, 1, 1), padding='SAME', conv=None, use_bias=True):
        """w = (wy, wx), n = (n_in, n_out)"""
        LayerConv.__init__(self, name, w, n, nl, strides, padding, conv, use_bias)
        self.dilation = dilation

    def __call__(self, vin):
        # TODO: replace with atrous_2d
        vout = tf.space_to_batch(vin, [[0, 0], [0, 0]], self.dilation)
        vout = LayerConv.__call__(self, vout)
        vout = tf.batch_to_space(vout, [[0, 0], [0, 0]], self.dilation)
        return vout


class LowRankMatrix(LayerBase):
    def __init__(self, name, rank, dims):
        self.dims = dims
        tsize = 1
        for x in dims:
            tsize *= x
        rows = cols = int(N.sqrt(tsize))
        while rows * cols != tsize:
            rows -= 1
            cols = tsize // rows
        with tf.name_scope(name):
            self.rows = tf.Variable(tf.truncated_normal([rows, 1, rank], stddev=0.01), name='rows')
            self.cols = tf.Variable(tf.truncated_normal([1, cols, rank], stddev=0.01), name='cols')

    def __call__(self):
        return tf.reshape(tf.reduce_sum(self.rows * self.cols, 2), self.dims)


class LayerEncodeConvGrowLinear(LayerBase):
    def __init__(self, name, n, width, colors, depth, scales, nl=lambda x, y: x + y):
        with tf.name_scope(name):
            encode = []
            nn = n
            for x in range(scales):
                cl = []
                for y in range(depth - 1):
                    cl.append(LayerConv('conv_%d_%d' % (x, y), [width, width], [nn, nn], nl))
                cl.append(LayerConv('conv_%d_%d' % (x, depth - 1), [width, width], [nn, nn + n], nl, strides=[1, 2, 2, 1]))
                encode.append(cl)
                nn += n
            self.encode = [LayerConv('conv_pre', [width, width], [colors, n], nl), encode]

    def __call__(self, vin, carry=0, train=True):
        vout = self.encode[0](vin)
        for convs in self.encode[1]:
            for conv in convs[:-1]:
                vtmp = conv(vout)
                vout = carry * vout + (1 - carry) * vtmp
            vout = convs[-1](vout)
        return vout


class LayerEncodeConvGrowExp2(LayerEncodeConvGrowLinear):
    def __init__(self, name, n, width, colors, depth, scales, nl=lambda x, y: x + y):
        with tf.name_scope(name):
            encode = []
            nn = n
            for x in range(scales):
                cl = []
                for y in range(depth - 1):
                    cl.append(LayerConv('conv_%d_%d' % (x, y), [width, width], [nn, nn], nl))
                cl.append(LayerConv('conv_%d_%d' % (x, depth - 1), [width, width], [nn, 2 * nn], nl, strides=[1, 2, 2, 1]))
                encode.append(cl)
                nn *= 2
            self.encode = [LayerConv('conv_pre', [width, width], [colors, n], nl), encode]


class LayerDecodeConvBlend(LayerBase):
    def __init__(self, name, n, width, colors, depth, scales, nl=lambda x, y: x + y):
        with tf.name_scope(name):
            decode = []
            for x in range(scales):
                cl = []
                n2 = 2 * n if x else n
                cl.append(LayerConv('conv_%d_%d' % (x, 0), [width, width], [n2, n], nl))
                for y in range(1, depth):
                    cl.append(LayerConv('conv_%d_%d' % (x, y), [width, width], [n, n], nl))
                decode.append(cl)
            self.decode = [decode, LayerConv('conv_post', [width, width], [n, colors])]

    def __call__(self, data, carry, return_hidden=False, train=True):
        vout = data
        layers = []
        for x, convs in enumerate(self.decode[0]):
            vout = tf.concat(3, [vout, data]) if x else vout
            vout = unboxn(convs[0](vout), 2)
            data = unboxn(data, 2)
            for conv in convs[1:]:
                vtmp = conv(vout)
                vout = carry * vout + (1 - carry) * vtmp
            layers.append(vout)
        if return_hidden:
            return self.decode[1](vout), layers
        return self.decode[1](vout)
