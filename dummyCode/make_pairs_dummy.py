# make_pairs.py
import numpy as np
import soundfile as sf
from pathlib import Path

SRC = "data/sounds_guitarScale.wav"     # guitar DI
TGT = "data/sounds_pianoScale.wav"     # same performance, target timbre
OUT = Path("datasets"); OUT.mkdir(parents=True, exist_ok=True)

FRAME = 512        # in/out size
N_TRAIN = 40000
N_VAL   = 5000
ALIGN_RADIUS = 256 # +/- samples for local alignment
RNG = np.random.default_rng(22)

def to_mono(x):
    return x.mean(axis=1) if x.ndim == 2 else x

def norm(x):
    m = np.max(np.abs(x)) + 1e-9
    return np.clip(x / m, -1.0, 1.0)

def best_shift(x_frame, y_seg):
    # maximize dot of abs signals (robust vs polarity)
    L = len(x_frame)
    center = len(y_seg) // 2
    best_c, best_s = -1e9, 0
    for s in range(center - ALIGN_RADIUS, center + ALIGN_RADIUS + 1):
        ya = y_seg[s - L//2 : s + (L - L//2)]
        if len(ya) != L: 
            continue
        c = float(np.dot(np.abs(x_frame), np.abs(ya)))
        if c > best_c:
            best_c, best_s = c, s - center
    return best_s

def build_pairs(x, y, n):
    lo = FRAME + ALIGN_RADIUS
    hi = len(x) - FRAME - ALIGN_RADIUS - 1
    starts = RNG.integers(low=lo, high=hi, size=n)

    X = np.empty((n, FRAME), dtype=np.float32)
    Y = np.empty((n, FRAME), dtype=np.float32)
    for i, r in enumerate(starts):
        xin = x[r : r + FRAME]
        center = r
        seg = y[center - ALIGN_RADIUS : center + ALIGN_RADIUS + FRAME]
        s = best_shift(xin, seg)
        start_t = center + s
        yout = y[start_t : start_t + FRAME]
        if len(yout) != FRAME:
            yout = y[center : center + FRAME]  # fallback
        X[i] = xin
        Y[i] = yout
    return X, Y

if __name__ == "__main__":
    x, sr_x = sf.read(SRC, dtype="float32", always_2d=True)
    y, sr_y = sf.read(TGT, dtype="float32", always_2d=True)
    x, y = to_mono(x), to_mono(y)
    assert sr_x == sr_y, f"Sample rates differ: {sr_x} vs {sr_y}"
    x, y = norm(x), norm(y)

    Xtr, Ytr = build_pairs(x, y, N_TRAIN)
    Xva, Yva = build_pairs(x, y, N_VAL)

    np.savez_compressed(OUT / "pairs_train.npz", X=Xtr, Y=Ytr, sr=sr_x)
    np.savez_compressed(OUT / "pairs_val.npz",   X=Xva, Y=Yva, sr=sr_x)
    print("Wrote datasets.")
