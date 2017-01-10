import bz2
import pickle
import sys
from math import log
from threading import Thread
from time import sleep, time

from tqdm import tqdm

import numpy as N
import os.path
import tensorflow as tf
from tfswag.logus import Log

__author__ = 'David Berthelot'


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


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def model_vars(g, prefix, group=tf.GraphKeys.TRAINABLE_VARIABLES):
    if not prefix:
        return g.get_collection(group), []
    match, no_match = [], []
    for x in g.get_collection(group):
        if x.name.startswith(prefix):
            match.append(x)
        else:
            no_match.append(x)
    return match, no_match


class TrainError:
    def __init__(self):
        self.value = 0
        self.count = 0

    def __call__(self, value):
        self.value += value
        self.count += 1

    def __float__(self):
        return self.value / max(self.count, 1)

    def __lt__(self, other):
        return not self.isnan() and (float(self) + 1e-4) < float(other)

    def __str__(self):
        return '%.4f' % float(self)

    @classmethod
    def max(cls):
        this = cls()
        this.value = 1e100
        return this

    def isnan(self):
        return N.isnan(self.value).any()


def train_errors(names, min_improvement=1e-4):
    class TrainErrors:
        def __init__(self):
            self.names = names
            self.values = N.zeros(len(names), 'f')
            self.value = 0
            self.count = 0

        def __call__(self, value, *values):
            self.values += values
            self.value += value
            self.count += 1

        def __float__(self):
            return self.value / max(self.count, 1)

        def __lt__(self, other):
            return not self.isnan() and (float(self) + min_improvement) < float(other)

        def __str__(self):
            txt = ' '.join(['%s %.4f' % (name, v / max(0, self.count + 1)) for name, v in zip(self.names, self.values)])
            return '%.4f' % float(self) + ' (' + txt + ')'

        @classmethod
        def max(cls):
            this = cls()
            this.value = 1e100
            return this

        def isnan(self):
            return N.isnan(self.value).any()

    return TrainErrors


class LearningRate:
    def __float__(self):
        raise NotImplementedError

    def __call__(self, improved):
        """Returns whether the model state should be restored."""
        raise NotImplementedError

    @property
    def stop(self):
        raise NotImplementedError


class LearningRateBasic(LearningRate):
    def __init__(self, lr, decr=1 / 2, incr=1.1, max_degrade_in_a_row=1, limit=1e-4, limit_max=1.0):
        self.lr = lr
        self.decr = decr
        self.incr = incr
        self.degrade_in_a_row = 0
        self.max_degrade_in_a_row = max_degrade_in_a_row
        self.limit = limit
        self.limit_max = float(limit_max)

    def __float__(self):
        return self.lr

    def __call__(self, improved):
        """Returns whether the model state should be restored."""
        if not improved:
            self.degrade_in_a_row += 1
        else:
            self.lr = min(self.limit_max, self.lr * self.incr)
            self.degrade_in_a_row = 0
            return False
        if self.degrade_in_a_row < self.max_degrade_in_a_row:
            return False
        self.degrade_in_a_row = 0
        self.lr *= self.decr
        return True

    @property
    def stop(self):
        return self.lr <= self.limit


class LearningRateDecay(LearningRate):
    def __init__(self, lr, decay=1e-4, limit=1e-4):
        self.lr = lr
        self.limit = limit
        self.decay = decay

    def __float__(self):
        return self.lr

    def __call__(self, improved):
        """Returns whether the model state should be restored."""
        self.lr -= self.decay
        return False

    @property
    def stop(self):
        return self.lr < self.limit


class LearningRateGeoDecay(LearningRate):
    def __init__(self, lr, iterations=50, limit=1e-4):
        self.lr = lr
        self.limit = limit
        self.iterartions = iterations
        self.decay = N.exp((log(limit) - log(lr)) / iterations)

    def __float__(self):
        return self.lr

    def __call__(self, improved):
        """Returns whether the model state should be restored."""
        self.lr *= self.decay
        return False

    @property
    def stop(self):
        return self.lr < self.limit


class TFModel:
    TMP_PATH = '/shared/TEMP'
    TrainError = TrainError

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.log = Log()
        self.g = tf.Graph()
        self.epoch_ = 0
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.allow_soft_placement = True
        # config.log_device_placement = True
        with self.g.as_default():
            self.session = tf.Session(config=config)
        self.debug = False
        self.train_nodes = None
        self.eval = None
        self.is_initialized = False
        self.args = dict(batch_size=batch_size)
        self.title = ''

    @property
    def size(self):
        return sum(x.get_shape().num_elements() for x in self.g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    @property
    def summary(self):
        s = ', '.join('%s=%s' % (name, value) for name, value in sorted(self.args.items()))
        if self.debug:
            self.debug_vars()
        return s

    @classmethod
    def load(cls, filepath):
        name, args = load_py(os.path.join(filepath, 'args.pdb'))
        assert cls.__name__ == name
        self = cls(**args)
        with self.g.as_default():
            tf.train.Saver().restore(self.session, os.path.join(filepath, 'vars.ckpt'))
        self.is_initialized = True
        return self

    def save(self, filepath, overwrite=False):
        filepath = filepath or self.title
        assert overwrite or not os.path.exists(filepath), 'Filename already exists'
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        with self.g.as_default():
            tf.train.Saver().save(self.session, os.path.join(filepath, 'vars.ckpt'))
        save_py((self.__class__.__name__, self.args), os.path.join(filepath, 'args.pdb'))

    def debug_vars(self, verbose=True):
        l = model_vars(self.g, '', group=tf.GraphKeys.VARIABLES)[0]
        with self.session.as_default():
            l = [(x.name, x.eval()) for x in l]
        if verbose:
            for name, val in l:
                print('%-60s % .6f % .6f' % (name, val.min(), val.max()))
        return l

    def sample(self, ds):
        raise NotImplementedError

    def validate(self, ds, epoch=0):
        raise NotImplementedError

    def init(self):
        if not self.is_initialized:
            print('Initializing...')
            uninitialized_vars = []
            for var in tf.all_variables():
                try:
                    self.session.run(var)
                except tf.errors.FailedPreconditionError:
                    uninitialized_vars.append(var)
            tf.initialize_variables(uninitialized_vars).run()
            self.is_initialized = True

    def train(self, ds, lr, epoch=1 << 23, max_it=10000, autosave=None):
        assert isinstance(lr, LearningRate)
        self.log = Log(self.title[:-4])
        self.log('-' * 80)
        self.log('%s[%d] batch=%d samples=%d  %s', self.__class__.__name__, self.size, self.batch_size, len(ds), self.summary)
        self.log('-' * 80)
        self.epoch_, last_err = 0, self.TrainError.max()

        _queue, _reader = [], [False]

        def reader(ds):
            _reader[0] = True
            while _reader[0]:
                while len(_queue) < 10:
                    _queue.append(self.sample(ds))
                sleep(0.001)

        try:
            thread = Thread(target=reader, args=(ds,))
            thread.start()
            while self.epoch_ < max_it:
                self.epoch_ += 1
                err_t = self.TrainError()
                t0 = time()
                with self.g.as_default():
                    self.init()
                    for batch in tqdm(range(0, epoch, self.batch_size), leave=False):
                        while not _queue:
                            sleep(0.001)
                        train_data = _queue.pop()
                        feed_dict = {self.lr: lr, **train_data}
                        err_t(*(self.session.run(self.train_nodes, feed_dict=feed_dict)[1:]))
                        if err_t.isnan():
                            raise ValueError('NaN during optimization', (train_data, float(lr)))
                self.log('Epoch %d  Error %s  [%.2f seconds, lr=%.4f]', self.epoch_, str(err_t), time() - t0, float(lr))
                lr(err_t < last_err)
                last_err = err_t
                if autosave is not None:
                    self.save(autosave, self.epoch_ > 1)
                if lr.stop:
                    break
                if ((self.epoch_ - 1) % 5) == 0:
                    self.log('%s[%d] batch=%d samples=%d  %s', self.__class__.__name__, self.size, self.batch_size, len(ds), self.summary)
                    self.validate(ds, self.epoch_)
        finally:
            _reader[0] = False
            thread.join()
        self.log('%s[%d] batch=%d samples=%d  %s', self.__class__.__name__, self.size, self.batch_size, len(ds), self.summary)
        self.validate(ds, self.epoch_)
        if autosave is not None:
            self.save(autosave, self.epoch_ > 1)
