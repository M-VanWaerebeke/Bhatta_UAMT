"""Predict a label volume from an explicitly supplied, trusted model checkpoint."""
import argparse
from pathlib import Path

import numpy as np
import torch


def predict(checkpoint, image, output, device="cpu"):
    import nibabel as nib
    from medicalDataLoader import normalize_intensity

    volume = nib.load(str(image))
    array = volume.get_fdata(dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("Input must be a prepared 3D single-channel volume or patch")
    # Historical training saves a complete nn.Module. Load only a checkpoint
    # you created or trust; its Python objects are required for deserialization.
    model = torch.load(checkpoint, map_location=device, weights_only=False)
    model.eval().to(device)
    tensor = torch.from_numpy(array.copy())[None, None]
    tensor = normalize_intensity(tensor, normalization="mean").to(device)
    with torch.no_grad():
        logits = model(tensor)
        if isinstance(logits, tuple):
            logits = logits[0]
        labels = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int16)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(labels, volume.affine), str(output))
    return labels


def runInference(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Trusted .pkl saved by main.py")
    parser.add_argument("image", help="Prepared NIfTI volume; dimensions compatible with model scales")
    parser.add_argument("output", help="Output NIfTI segmentation")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    predict(args.checkpoint, args.image, args.output, args.device)


if __name__ == "__main__":
    runInference()
