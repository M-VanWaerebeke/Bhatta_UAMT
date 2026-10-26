"""Verify the model and a complete tiny train/save/inference path without medical data."""
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from examples.cpu_demo import run_demo
from Inference import predict
from UNet3D import Unet3D


def test_core_cpu_demo():
    result = run_demo()
    assert result["logits_shape"] == [2, 3, 16, 16, 16]
    assert result["uncertainty_shape"] == [2, 1, 16, 16, 16]
    assert result["loss"] > 0


def test_one_epoch_with_prepared_nifti_data(tmp_path, monkeypatch):
    import main

    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rng = np.random.default_rng(123)
    rows = []
    for i in range(9):
        image = rng.normal(size=(16, 16, 16)).astype(np.float32)
        labels = rng.integers(0, 3, size=image.shape).astype(np.int16)
        for name, array in [(f"image{i}.nii.gz", image), (f"labels{i}.nii.gz", labels),
                            (f"mask{i}.nii.gz", np.ones_like(image))]:
            nib.save(nib.Nifti1Image(array, np.eye(4)), tmp_path / name)
        rows.append({"sub": f"subject{i // 3}", "ses": i // 3,
                     "T2_modality": f"image{i}.nii.gz",
                     "segmentation_2classes": f"labels{i}.nii.gz",
                     "brainmask_bet": f"mask{i}.nii.gz", "patches": "{}"})
    pd.DataFrame(rows).to_csv(tmp_path / "DHCP-patches-train-dataframe.csv", index=False)
    pd.DataFrame(rows[:3]).to_csv(tmp_path / "DHCP-patches-val-dataframe.csv", index=False)
    main.runTraining(root_dir=str(tmp_path), main_path=str(tmp_path / "stats"),
                     model_dir=str(tmp_path / "models"), epoch=1, labelled_samples=1,
                     val_samples=0, batch_size=2, batch_size_val=2,
                     batch_size_unlabelled=2, num_classes=3, x_patch=16,
                     y_patch=16, z_patch=16, T=2, N=1, seed=17, num_workers=0)
    checkpoints = list((tmp_path / "models").glob("*.pkl"))
    assert len(checkpoints) == 1
    stats = list((tmp_path / "stats").glob("*/train-Losses.npy"))
    assert len(stats) == 1
    assert np.isfinite(np.load(stats[0])).all()
    assert np.load(stats[0]).shape == (1,)
    prediction = predict(checkpoints[0], tmp_path / "image0.nii.gz", tmp_path / "prediction.nii.gz")
    assert prediction.shape == (16, 16, 16)
    assert set(np.unique(prediction)) <= {0, 1, 2}


def test_inference_preserves_volume_affine(tmp_path):
    model = Unet3D(init_ker=2, inp_chan=1, out_chan=3, n_scales=3)
    checkpoint = tmp_path / "model.pkl"
    torch.save(model, checkpoint)
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    image = np.random.default_rng(0).normal(size=(16, 16, 16)).astype(np.float32)
    nib.save(nib.Nifti1Image(image, affine), tmp_path / "image.nii.gz")
    predict(checkpoint, tmp_path / "image.nii.gz", tmp_path / "labels.nii.gz")
    np.testing.assert_array_equal(nib.load(tmp_path / "labels.nii.gz").affine, affine)


def test_single_image_labels_keep_batch_dimension():
    from utils import getTargetSegmentation
    labels = torch.zeros(1, 8, 8, 8)
    assert getTargetSegmentation(labels).shape == labels.shape
