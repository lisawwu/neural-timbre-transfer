#!/usr/bin/env python3
"""
Offline version of live_pedal_dummy.py
Processes an input WAV file using your trained TorchScript model
and saves the output to a new WAV file.
"""

import argparse
import numpy as np
import soundfile as sf
import torch

FRAME = 512
HOP = FRAME // 2  # 50% overlap

def frame_signal(x, frame, hop):
    n = len(x)
    if n < frame:
        x = np.pad(x, (0, frame - n))
    n_frames = 1 + (len(x) - frame) // hop
    return np.stack([x[i * hop : i * hop + frame] for i in range(n_frames)])

def overlap_add(frames, hop):
    frame = frames.shape[1]
    n = (frames.shape[0] - 1) * hop + frame
    y = np.zeros(n, dtype=np.float32)
    w = np.hanning(frame).astype(np.float32)
    norm = np.zeros(n, dtype=np.float32)
    for i in range(frames.shape[0]):
        start = i * hop
        y[start:start + frame] += frames[i] * w
        norm[start:start + frame] += w
    norm[norm == 0] = 1
    return y / norm

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to TorchScript model (.ts)")
    p.add_argument("--in", dest="inp", required=True, help="Input WAV")
    p.add_argument("--out", dest="out", required=True, help="Output WAV")
    args = p.parse_args()

    print(f"Loading model: {args.model}")
    model = torch.jit.load(args.model, map_location="cpu").eval()
    torch.set_num_threads(1)

    x, sr = sf.read(args.inp)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    x /= np.max(np.abs(x)) + 1e-8

    frames = frame_signal(x, FRAME, HOP)
    out_frames = np.zeros_like(frames, dtype=np.float32)

    with torch.no_grad():
        for i in range(frames.shape[0]):
            inp = torch.from_numpy(frames[i]).unsqueeze(0)  # (1, 512)
            y = model(inp).squeeze(0).cpu().numpy().astype(np.float32)
            out_frames[i] = y

    y = overlap_add(out_frames, HOP)
    y /= np.max(np.abs(y)) + 1e-8
    sf.write(args.out, y, sr, subtype="PCM_16")
    print(f"✅ Done! Wrote {args.out}")

if __name__ == "__main__":
    main()
