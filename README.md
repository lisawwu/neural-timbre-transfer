# Neural Timbre Transfer (Version 1, dummy model)
Hi guys, this is the prototype version I made to see if this concept and algorithm will work. This project uses a neural network (MLP) to transform the timbre of one instrument into another — for example, converting a guitar tone into a piano tone in real-time.

## Project Structure
- `dummyCode/` — DUMMY MODEL: its gonna chopped. contains model training and live audio code
- `datasets/` — contains input/output audio pairs (ignored in Git because the files are too big lol)
- `models/` — stores trained weights (ignored in Git again)
- `libraryTest.py` — used for validating libraries. i made this because my things got installed wrong and i had to fix it haha

## Run Training
```bash
python dummyCode/train_mlp_dummy.py
