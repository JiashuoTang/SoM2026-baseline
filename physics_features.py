"""Classical LoS/NLoS features from CSI.

A direct path leaves three signatures the frozen WiFo2 encoder does not surface
on 10 training samples: energy concentrated in the first delay tap, a short RMS
delay spread, and a channel that stays correlated over time. These are the
textbook Rician-vs-Rayleigh discriminators.

Measured on dataset/Task1 (10 samples, seed-averaged 5-fold CV, binary F1 with
LoS positive -- the challenge metric):

    physics + WiFo2 mean-pool, k=3   0.846
    physics alone, k=1               0.771
    all-LoS (trivial positive)       0.571
    WiFo2 mean-pool alone, k=3       0.679
    WiFo2 linear probe (baseline)    0.169
    majority class (all NLoS)        0.000

The Doppler, angular and per-antenna-K-spread features below measured USELESS on
this dataset (max t-stat 1.58 vs 3.86 for rms_delay; SelectKBest never picks them,
and scores are identical with and without). Kept because they cost nothing at
inference and the private set behaves very differently from these 10 samples --
see notes/results.md for why they fail here.
"""
import numpy as np
from scipy import stats

# Selected in 10/10 leave-one-out folds by SelectKBest(f_classif). Everything
# else is available but less stable at this sample count.
STABLE = ('rms_delay', 'first_tap_frac', 'temporal_corr')


def sample_features(h):
    """h: (T, A, K) complex CSI for one sample -> dict of scalar features."""
    f = {}
    mag = np.abs(h)

    # Rician K-factor: direct-path power over scattered power, estimated across
    # time for each (antenna, subcarrier). High K => LoS.
    mu = h.mean(0)
    var = (np.abs(h - mu) ** 2).mean(0)
    K = (np.abs(mu) ** 2) / (var + 1e-12)
    f['K_mean'], f['K_med'], f['K_max'] = K.mean(), np.median(K), K.max()
    f['K_db'] = 10 * np.log10(K.mean() + 1e-12)

    # moment-based Rice fit, 2nd and 4th moments of |h|
    m2, m4 = (mag ** 2).mean(), (mag ** 4).mean()
    r = np.sqrt(max(2 * m2 ** 2 - m4, 1e-12))
    f['K_moment'] = r / (m2 - r + 1e-12)

    # amplitude distribution shape. Rayleigh (NLoS) has CV = 0.523
    m = mag.ravel()
    f['cv'] = m.std() / (m.mean() + 1e-12)
    f['kurt'] = stats.kurtosis(m)
    f['skew'] = stats.skew(m)
    f['p90_p50'] = np.percentile(m, 90) / (np.percentile(m, 50) + 1e-12)

    # power delay profile: IFFT across subcarriers
    pdp = (np.abs(np.fft.ifft(h, axis=-1)) ** 2).mean((0, 1))
    tau = np.arange(len(pdp))
    p = pdp / pdp.sum()
    mean_tau = (tau * p).sum()
    f['rms_delay'] = np.sqrt((((tau - mean_tau) ** 2) * p).sum())
    f['pdp_peak_ratio'] = pdp.max() / (pdp.mean() + 1e-12)
    f['pdp_entropy'] = -(p * np.log(p + 1e-12)).sum()
    f['first_tap_frac'] = pdp[0] / (pdp.sum() + 1e-12)

    # coherence bandwidth proxy: how far correlation survives across subcarriers
    hf = h.reshape(-1, h.shape[-1])
    c = np.abs([_corr(hf[:, 0], hf[:, k]) for k in range(h.shape[-1])])
    f['coh_bw'] = (c > 0.5).sum() / len(c)
    f['freq_corr_mean'] = c.mean()

    # spatial correlation between antenna pairs
    ha = h.transpose(1, 0, 2).reshape(h.shape[1], -1)
    f['spatial_corr'] = np.mean([_corr(ha[i], ha[j])
                                 for i in range(len(ha)) for j in range(i + 1, len(ha))])

    # temporal stability against the first slot
    ht = h.reshape(h.shape[0], -1)
    f['temporal_corr'] = np.mean([_corr(ht[0], ht[t]) for t in range(h.shape[0])])

    # --- per-antenna K-factor spread. A direct path reaches every element of the
    # array, so LoS should give a similar K everywhere; NLoS varies element to
    # element as the local scattering differs.
    Ka = K.mean(-1)                                   # K per antenna, averaged over subcarriers
    f['K_ant_std'] = Ka.std()
    f['K_ant_range'] = Ka.max() - Ka.min()
    f['K_ant_cv'] = Ka.std() / (Ka.mean() + 1e-12)

    # --- Doppler: FFT along time. A static direct path concentrates power at DC;
    # motion and rich scattering spread it.
    dop = (np.abs(np.fft.fft(h, axis=0)) ** 2).mean((1, 2))
    dop = np.fft.fftshift(dop)
    pd_ = dop / dop.sum()
    nu = np.arange(len(pd_)) - len(pd_) // 2
    f['doppler_dc_frac'] = pd_[len(pd_) // 2]
    f['doppler_spread'] = np.sqrt(((nu ** 2) * pd_).sum() - ((nu * pd_).sum()) ** 2)
    f['doppler_entropy'] = -(pd_ * np.log(pd_ + 1e-12)).sum()
    f['doppler_peak_ratio'] = dop.max() / (dop.mean() + 1e-12)

    # --- angular spread: FFT along the antenna axis into beamspace. LoS arrives
    # from one direction, so power sits in a single angular bin. Zero-padded to 64
    # because 8 elements alone give very coarse resolution.
    ang = (np.abs(np.fft.fft(h, n=64, axis=1)) ** 2).mean((0, 2))
    ang = np.fft.fftshift(ang)
    pa = ang / ang.sum()
    th = np.arange(len(pa)) - len(pa) // 2
    f['angular_spread'] = np.sqrt(((th ** 2) * pa).sum() - ((th * pa).sum()) ** 2)
    f['angular_entropy'] = -(pa * np.log(pa + 1e-12)).sum()
    f['angular_peak_ratio'] = ang.max() / (ang.mean() + 1e-12)
    f['angular_top_frac'] = np.sort(pa)[-4:].sum()     # power in the 4 strongest beams
    return f


def _corr(a, b):
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def extract(H, only=None):
    """H: (N, T, A, K) complex -> (X, names). `only` selects a subset of features."""
    rows = [sample_features(H[i]) for i in range(len(H))]
    names = [n for n in rows[0] if only is None or n in only]
    return np.array([[r[n] for n in names] for r in rows]), names


def _demo():
    """Synthetic LoS and NLoS channels must separate on the stable features."""
    rng = np.random.default_rng(0)
    T, A, K, taps = 24, 8, 128, 16

    def make(k_factor):
        # one direct tap plus rayleigh-faded scatterers, then FFT to frequency
        out = np.empty((T, A, K), complex)
        for t in range(T):
            for a in range(A):
                imp = (rng.normal(size=taps) + 1j * rng.normal(size=taps)) * np.exp(-np.arange(taps) / 4)
                imp[0] += k_factor
                out[t, a] = np.fft.fft(imp, K)
        return out

    los, nlos = sample_features(make(20.0)), sample_features(make(0.0))
    assert los['first_tap_frac'] > nlos['first_tap_frac'], 'LoS must concentrate energy in tap 0'
    assert los['rms_delay'] < nlos['rms_delay'], 'LoS must have shorter delay spread'
    assert los['K_max'] > nlos['K_max'], 'LoS must show a higher Rician K'
    assert los['doppler_dc_frac'] > nlos['doppler_dc_frac'], 'LoS must hold more power at DC'
    assert los['angular_entropy'] < nlos['angular_entropy'], 'LoS must occupy fewer beams'
    print(f'demo ok: {len(los)} features')
    for n in STABLE + ('K_max', 'K_ant_std', 'doppler_dc_frac', 'doppler_spread',
                       'angular_spread', 'angular_entropy'):
        print(f'  {n:20s} LoS {los[n]:10.4f}   NLoS {nlos[n]:10.4f}')


if __name__ == '__main__':
    _demo()
