# train_mlp.py
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

DATA_TR = "datasets/pairs_train.npz"
DATA_VA = "datasets/pairs_val.npz"
OUT = Path("models"); OUT.mkdir(parents=True, exist_ok=True)

HIDDEN_MUL = 3
LR = 1e-3
EPOCHS = 30
BATCH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MLP(nn.Module):
    def __init__(self, frame=512, hidden_mul=3):
        super().__init__()
        h = frame * hidden_mul
        self.net = nn.Sequential(
            nn.Linear(frame, h),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(h, h),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(h, h),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(h, frame),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

def load_npz(path):
    d = np.load(path)
    return d["X"].astype("float32"), d["Y"].astype("float32"), int(d["sr"])

if __name__ == "__main__":
    Xtr, Ytr, sr = load_npz(DATA_TR)
    Xva, Yva, _  = load_npz(DATA_VA)
    FRAME = Xtr.shape[1]

    tr_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr))
    va_ds = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva))
    tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=BATCH, shuffle=False, drop_last=False)

    model = MLP(frame=FRAME, hidden_mul=HIDDEN_MUL).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best = 1e9
    for epoch in range(1, EPOCHS+1):
        model.train()
        tl = 0.0
        for xb, yb in tqdm(tr_dl, desc=f"Epoch {epoch}/{EPOCHS}"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tl += loss.item() * xb.size(0)
        tl /= len(tr_ds)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vl += loss_fn(model(xb), yb).item() * xb.size(0)
        vl /= len(va_ds)
        print(f"train_loss={tl:.6f}  val_loss={vl:.6f}")

        if vl < best:
            best = vl
            torch.save(model.state_dict(), OUT / "mlp_512_best.pt")
            # TorchScript export
            example = torch.randn(1, FRAME, device=DEVICE)
            traced = torch.jit.trace(model, example)
            traced.save(str(OUT / "mlp_512_best.ts"))
            print("Saved best model (.pt and .ts)")

    print("Done. Best val_loss:", best)
