# Neural Timbre Transfer 
This project uses a neural network (MLP) to transform the timbre of one instrument into another — for example, converting a guitar tone into a piano tone in real-time.

## Project Structure
- `dummyCode/` — DUMMY MODEL: its gonna chopped. contains model training and live audio code
- `datasets/` — contains input/output audio pairs (ignored in Git because the files are too big lol)
- `models/` — stores trained weights (ignored in Git, ignored again)
- `libraryTest.py` — used for quick testing

## Run Training
```bash
python dummyCode/train_mlp_dummy.py
