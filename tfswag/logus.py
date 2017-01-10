import datetime

import os
import os.path

__author__ = 'David Berthelot'


class Log:
    def __init__(self, prefix=None, path='logs', flush_every=10):
        self.save_to_file = prefix is not None
        self.flush_every = flush_every
        self.lines = 0
        if self.save_to_file:
            now = datetime.datetime.now().strftime('%Y-%m-%d_%H.%m.%S')
            self.filename = os.path.join(path, prefix.split('/')[-1] + '_' + now + '.txt')
            if not os.path.exists(path):
                os.mkdir(path)
            self.file = open(self.filename, 'w')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.save_to_file:
            self.file.close()

    def __call__(self, text, *args):
        to_print = text % args
        print(to_print)
        if self.save_to_file:
            last_lines = self.lines
            self.lines += 1 + to_print.count('\n')
            self.file.write(to_print + '\n')
            if self.lines // self.flush_every != last_lines // self.flush_every:
                self.file.flush()
