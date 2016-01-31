import glob
import os
import subprocess
from argparse import ArgumentParser
import code
import sys
try:
    from subprocess import DEVNULL  # py3k
except ImportError:
    DEVNULL = open(os.devnull, 'wb')


def chunk(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def main(dirn, name_cell):
    objs = glob.glob(os.path.join(dirn, '*.obj'))
    d = 0
    while len(objs) > 1:
        c = 0
        for subobjs in chunk(objs, 450):
            objl = ' '.join(subobjs)
            out = os.path.join(dirn, 'out-%i-%i.obj' % (d, c))
            subprocess.call('meshlabserver -i ' + objl + ' -s flatten.mlx -o ' + out, shell=True)
            c += 1
            for obj in subobjs:
                os.remove(obj)
        d += 1
        objs = glob.glob(os.path.join(dirn, '*.obj'))
    os.rename(objs[0], os.path.join('convertedOBJs/' + os.path.basename(name_cell) + '.obj'))


if __name__ == '__main__':
    'FLAT'
    d = sys.argv[1]
    name = sys.argv[2]
    main(d, name)
