import numpy as np
import sounddevice as sd
import torch
import time

MODEL_PATH = "models/mlp_512_best.ts"

SR = 48000
HOP = 1024           # device callback size
FRAME = 512          # model expects this

# TODO: set these after running sd.query_devices()
INPUT_DEVICE = 1
OUTPUT_DEVICE = 1

CHANNELS_IN = 1
CHANNELS_OUT = 2

# load model
model = torch.jit.load(MODEL_PATH, map_location="cpu").eval()
torch.set_num_threads(1)

# warmup
with torch.no_grad():
    _ = model(torch.zeros(1, FRAME))

inbuf = np.zeros(FRAME, dtype=np.float32)
torch_in = torch.zeros(1, FRAME, dtype=torch.float32)

last_print = time.time()

def callback(indata, outdata, frames, timeinfo, status):
    global inbuf, last_print

    if status:
        print("audio status:", status)

    if frames != HOP:
        print("Warning: frames != HOP", frames, HOP)

    # mono input
    mono = indata[:, 0].astype(np.float32)

    # keep last FRAME samples
    if len(mono) >= FRAME:
        inbuf[:] = mono[-FRAME:]
    else:
        # shift old data left, append new
        inbuf = np.roll(inbuf, -len(mono))
        inbuf[-len(mono):] = mono

    # measure RMS to see if we’re getting signal
    rms = np.sqrt(np.mean(inbuf ** 2))

    # ====== GATE (looser) ======
    # if you play softly, this will still pass
    gate_thresh = 1e-5   # was 1e-4
    if rms < gate_thresh:
        # either no input or very quiet, still output tiny noise to prove it's alive
        y = np.zeros(frames, dtype=np.float32)
    else:
        # run model
        torch_in[0].copy_(torch.from_numpy(inbuf))
        with torch.no_grad():
            pred = model(torch_in)[0].cpu().numpy().astype(np.float32)
        # pred is 512, but we need 1024 -> pad
        y = np.zeros(frames, dtype=np.float32)
        y[:FRAME] = pred

    # safety clip
    y = np.clip(y, -0.98, 0.98)

    # write to output (stereo)
    outdata[:, 0] = y
    if CHANNELS_OUT == 2:
        outdata[:, 1] = y

    # occasional debug print
    now = time.time()
    if now - last_print > 1.0:
        print(f"callback ok | rms_in={rms:.6f}")
        last_print = now

# start stream
with sd.Stream(
    device=(INPUT_DEVICE, OUTPUT_DEVICE),
    samplerate=SR,
    blocksize=HOP,
    dtype="float32",
    channels=(CHANNELS_IN, CHANNELS_OUT),
    callback=callback,
):
    print("Running neural pedal...")
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        pass
