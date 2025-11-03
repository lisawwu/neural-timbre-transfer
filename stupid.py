import numpy as np
import sounddevice as sd

SR = 48000
BLOCKSIZE = 256
CHANNELS_IN = 1
CHANNELS_OUT = 2

# filter/drive params
DRIVE = 4.0         # increase for more distortion
LP_ALPHA = 0.85     # closer to 1.0 = darker

prev_y = 0.0  # for low-pass state


def find_scarlett():
    devices = sd.query_devices()
    input_dev = None
    output_dev = None
    for i, d in enumerate(devices):
        name = d['name'].lower()
        if "scarlett" in name or "focusrite" in name:
            # pick first match
            if d['max_input_channels'] > 0 and input_dev is None:
                input_dev = i
            if d['max_output_channels'] > 0 and output_dev is None:
                output_dev = i
    return input_dev, output_dev


def audio_callback(indata, outdata, frames, time, status):
    global prev_y

    if status:
        print("Status:", status)

    # mono in
    x = indata[:, 0].astype(np.float32)

    # 1) overdrive
    x = np.tanh(DRIVE * x)

    # 2) simple 1-pole low-pass
    y = np.empty_like(x)
    y_prev = prev_y
    for i, sample in enumerate(x):
        y_prev = LP_ALPHA * y_prev + (1 - LP_ALPHA) * sample
        y[i] = y_prev
    prev_y = y_prev

    # stereo out
    outdata[:] = np.repeat(y[:, None], CHANNELS_OUT, axis=1)


def main():
    in_dev, out_dev = find_scarlett()

    if in_dev is None or out_dev is None:
        print("⚠️ Could not auto-find a Scarlett/Focurite device.")
        print("Using system defaults. Run `sd.query_devices()` to get exact indexes.")
        device = None
    else:
        print(f"✅ Using input device {in_dev}, output device {out_dev}")
        device = (in_dev, out_dev)

    with sd.Stream(
        device=device,
        samplerate=SR,
        blocksize=BLOCKSIZE,
        dtype="float32",
        channels=(CHANNELS_IN, CHANNELS_OUT),
        callback=audio_callback,
    ):
        print("🎸 Real-time guitar filter running. Ctrl+C to stop.")
        while True:
            sd.sleep(1000)


if __name__ == "__main__":
    main()
