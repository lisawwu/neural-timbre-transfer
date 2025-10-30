# live_pedal.py
import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import soundfile as sf  # optional for quick test recording

MODEL_PATH = "models/mlp_512_best.ts"  # TorchScript model
SR = 48000
FRAME = 512         # model IO size
WIN = 1024          # synthesis window
HOP = 512           # 50% overlap
CHANNELS_IN = 1
CHANNELS_OUT = 2    # stereo out
INPUT_DEVICE = None   # set to your Focusrite input index or name
OUTPUT_DEVICE = None  # set to your Focusrite output index or name

# Load model
model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()
torch.set_num_threads(1)

# Hann window for overlap-add
win = np.hanning(WIN).astype(np.float32)

# Simple circular buffer for input frames
inbuf = np.zeros(WIN, dtype=np.float32)
outbuf = np.zeros(WIN, dtype=np.float32)
outpos = 0

def process_block(x_block):
    # x_block: shape (FRAME,) float32 in [-1,1]
    with torch.no_grad():
        inp = torch.from_numpy(x_block).view(1, -1)
        y = model(inp).squeeze(0).numpy().astype(np.float32)
    return y

def callback(indata, outdata, frames, time, status):
    global inbuf, outbuf, outpos
    if status:
        print(status)
    # mono input (avg L/R if stereo)
    mono = indata.mean(axis=1).astype(np.float32)

    # shift left by HOP, append new samples
    inbuf[:-HOP] = inbuf[HOP:]
    inbuf[-HOP:] = mono[:HOP] if len(mono) >= HOP else np.pad(mono, (0, HOP-len(mono)))

    # take center FRAME from the last WIN using a simple tap (or use a window to feed)
    start = WIN - FRAME
    xin = inbuf[start:start+FRAME].copy()

    y = process_block(xin)

    # overlap-add: place y into windowed buffer
    # here we window a 1024 output frame by embedding 512-sample y in the middle of WIN
    # (simplest: mirror y across WIN/2, but we’ll just zero-pad around the center)
    temp = np.zeros(WIN, dtype=np.float32)
    mid = WIN//2 - FRAME//2
    temp[mid:mid+FRAME] = y
    temp *= win

    # shift outbuf and add temp
    outbuf[:-HOP] = outbuf[HOP:]
    outbuf[-WIN:] += temp

    # take latest HOP samples to output
    chunk = outbuf[-HOP:]
    # limiter just in case
    chunk = np.clip(chunk, -0.98, 0.98)

    # write stereo
    outdata[:len(chunk), 0] = chunk
    if CHANNELS_OUT == 2:
        outdata[:len(chunk), 1] = chunk
    if len(chunk) < frames:
        outdata[len(chunk):, :] = 0.0

# open full-duplex stream
with sd.Stream(
    device=(INPUT_DEVICE, OUTPUT_DEVICE),
    samplerate=SR,
    blocksize=HOP,          # audio callback every HOP samples
    dtype="float32",
    channels=(CHANNELS_IN, CHANNELS_OUT),
    callback=callback
):
    print("Neural pedal running. Ctrl+C to stop.")
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        pass
