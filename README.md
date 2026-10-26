# Bhattacharyya uncertainty for semi-supervised 3D segmentation

Research code accompanying **On the pitfalls of entropy-based uncertainty for multi-class semi-supervised segmentation** (2022), by Martin Van Waerebeke, Gregory Lodygensky and Jose Dolz.

[Paper](https://arxiv.org/abs/2203.03587) · [Data preparation contract](docs/data.md) · [Citation](CITATION.cff) · [Third-party notices](THIRD_PARTY.md)

The project compares uncertainty measures for multi-class segmentation in a mean-teacher setup. It includes a 3D U-Net, Dice and consistency losses, and a Bhattacharyya-based measure computed across perturbed predictions.

## Run a small CPU example

The synthetic example uses the actual model and loss code. It performs a forward pass, backpropagates cross-entropy plus Dice loss, and computes Bhattacharyya uncertainty from four stochastic predictions. It needs no medical data or GPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
python examples/cpu_demo.py
```

Expected output includes logits shaped `[2, 3, 16, 16, 16]`, uncertainty shaped `[2, 1, 16, 16, 16]` and a finite loss (approximately `1.6963` in the checked CPU environment). This verifies execution of the model, loss and uncertainty calculation; it is not a reproduction of the paper's results.

For the complete integration checks:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The tests also prepare tiny synthetic NIfTI volumes, run one epoch through the real dataset loader and training loop, save a checkpoint, and perform segmentation inference. The checked environment is Python 3.9.6, PyTorch 2.8.0 and NumPy 2.0.2, with the medical-image dependencies pinned in `requirements.txt`. It is a current maintenance environment, not a reconstruction of the original training environment.

## Train with prepared medical data

The research used the [developing Human Connectome Project release](http://www.developingconnectome.org/data-release/second-data-release/), which has its own access process. Medical data, split CSVs and trained checkpoints are not distributed here.

First read [the required volume and CSV layout](docs/data.md). In particular, the historical loader reads already prepared fixed-size volumes/patches: it does not apply the recorded crop coordinates. The original preprocessing pipeline and exact paper splits are not available in this repository, so a fresh run should not be described as an exact paper reproduction.

```bash
python main.py --root-dir /path/to/prepared-data \
  --main-path results/statistics --model-dir results/models \
  --epoch 100 --seed 42 --labelled-samples 3 --val-samples 1 \
  --batch-size 6 --patch-size 64
```

A GPU with approximately 16 GB or more VRAM is recommended for the historical configuration. The command writes per-epoch training/validation statistics and the best validation checkpoint. Additional research options are available through the documented `runTraining(...)` function in `main.py`.

Predict a label map from a prepared volume or patch compatible with the model's scales:

```bash
python Inference.py /path/to/trusted-checkpoint.pkl \
  /path/to/prepared-volume.nii.gz results/prediction.nii.gz --device cpu
```

The historical trainer saves the complete PyTorch model object, so inference should load a checkpoint you created or trust. Inference preserves the input affine; it does not implement whole-volume sliding-window stitching or raw-MRI preprocessing.

## Code map and maintenance scope

| Path | Purpose |
| --- | --- |
| `main.py` | Labelled/unlabelled training and validation |
| `UNet3D.py` | 3D segmentation model |
| `losses.py` | Dice, consistency and uncertainty calculations |
| `medicalDataLoader.py` | Prepared NIfTI data and split CSVs |
| `Inference.py` | Checkpoint-to-NIfTI label prediction |
| `examples/cpu_demo.py`, `tests/` | Offline CPU example and supported checks |
| `UNet.py`, `UNet_Base.py`, `utils/` | Retained historical model and utility variants |

The repository history records the main code import and subsequent updates under Martin Van Waerebeke's historical GitHub identity, `Mart1vw`. The implementation also contains adapted third-party utilities; their source notices remain in place. The existing [MIT license](LICENSE) is retained, and file-specific notices and exceptions are described in [THIRD_PARTY.md](THIRD_PARTY.md).

This maintenance update fixes output paths, configurable seeding, loss recording, CPU allocation and batch-size-one labels, and replaces the incomplete inference stub. Full medical-data experiments and unused legacy model/utility variants have not been rerun. Please cite the paper when building on the research.
