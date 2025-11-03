import importlib, sys
import sounddevice as sd
print(sd.query_devices())

modules = [
    "sounddevice",
    "soundfile",
    "numpy",
    "scipy",
    "tensorflow",
    "torch",
    "torchaudio",
    "matplotlib",
    "pandas",
    "keras"
]

for m in modules:
    try:
        mod = importlib.import_module(m)
        print(f"✅ {m} imported successfully – version:", getattr(mod, "__version__", "unknown"))
    except Exception as e:
        print(f"❌ {m} failed: {e.__class__.__name__} – {e}")

