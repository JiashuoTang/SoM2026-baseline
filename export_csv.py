"""Flatten H_train.mat (complex CSI) to long-format CSV: one row per (sample, time, antenna, subcarrier)."""
import numpy as np, scipy.io as sio, h5py, sys
from pathlib import Path

DATA = Path('dataset/Task1')
OUT = Path('export'); OUT.mkdir(exist_ok=True)


def load_mat(path, key):
    """Read one variable from a .mat file, v7 or v7.3."""
    try:
        return sio.loadmat(path)[key]
    except NotImplementedError:
        with h5py.File(path, 'r') as f:
            a = f[key][...]
            if a.dtype.names == ('real', 'imag'):
                a = a['real'] + 1j * a['imag']
            return a.transpose(range(a.ndim)[::-1])


H = load_mat(DATA / 'H_train.mat', 'H')
L = load_mat(DATA / 'L_train.mat', 'label').ravel()
N, T, A, K = H.shape
assert len(L) == N, f'label count {len(L)} != sample count {N}'

s, t, a, k = np.meshgrid(np.arange(N), np.arange(T), np.arange(A), np.arange(K), indexing='ij')
rows = np.column_stack([
    s.ravel(), t.ravel(), a.ravel(), k.ravel(), L[s.ravel()],
    H.real.ravel(), H.imag.ravel(), np.abs(H).ravel(), np.angle(H).ravel(),
])

path = OUT / 'H_train.csv'
np.savetxt(path, rows, delimiter=',', comments='',
           header='sample,time,antenna,subcarrier,label,real,imag,magnitude,phase',
           fmt=['%d'] * 5 + ['%.8g'] * 4)
print(f'{path}  {rows.shape[0]} rows  {path.stat().st_size / 1e6:.1f} MB')
