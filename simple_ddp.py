from accelerate import Accelerator
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def main():
    acc = Accelerator()
    torch.cuda.set_device(acc.local_process_index)
    # small, deterministic dataset
    x = torch.randn(2048, 32)
    y = torch.randint(0, 10, (2048,))
    dl = DataLoader(TensorDataset(x,y), batch_size=64, shuffle=True,
                    num_workers=0, pin_memory=False, persistent_workers=False, drop_last=True)
    model = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 10)).cuda()
    opt = optim.AdamW(model.parameters(), 1e-3)
    model, opt, dl = acc.prepare(model, opt, dl)
    for step,(bx,by) in enumerate(dl):
        opt.zero_grad(); loss = nn.functional.cross_entropy(model(bx), by)
        acc.backward(loss); opt.step()
        if acc.is_main_process and step % 20 == 0:
            print("step", step, "loss", loss.item())
if __name__ == "__main__":
    main()