"""Exercise the real 3D U-Net, Dice loss and Bhattacharyya uncertainty on CPU."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from UNet3D import Unet3D
from losses import DiceLoss, new_bhatta_tensor


def run_demo():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = Unet3D(init_ker=2, inp_chan=1, out_chan=3, n_scales=3)
    image = torch.randn(2, 1, 16, 16, 16)
    target = torch.randint(0, 3, (2, 16, 16, 16))
    one_hot = F.one_hot(target, num_classes=3).movedim(-1, 1).float()
    logits = model(image)
    loss = F.cross_entropy(logits, target) + DiceLoss(normalization="softmax")(logits, one_hot)
    loss.backward()
    with torch.no_grad():
        samples = torch.stack([model(image).softmax(dim=1) for _ in range(4)])
        uncertainty = new_bhatta_tensor(samples)
    assert torch.isfinite(loss)
    assert torch.isfinite(uncertainty).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    result = {"logits_shape": list(logits.shape), "loss": float(loss.detach()),
              "uncertainty_shape": list(uncertainty.shape),
              "purpose": "synthetic execution smoke test; not paper reproduction"}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_demo()
