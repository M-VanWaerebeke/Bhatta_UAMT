# Prepared-data contract

The historical code uses dHCP T2 images and segmentation labels. Obtain the dataset through its own authorized access process. This repository does not contain the original preprocessing pipeline or exact train/validation/test split files.

`--root-dir` must contain prepared NIfTI inputs and these manifests:

- `DHCP-patches-train-dataframe.csv`
- `DHCP-patches-val-dataframe.csv`
- `DHCP-patches-test-dataframe.csv` only if calling the loader in test mode

Each CSV row describes one already prepared volume or fixed-size patch. Required columns for the default ten-class model:

```csv
sub,ses,T2_modality,segmentation_9classes,brainmask_bet,patches
subject001,1,images/patch001.nii.gz,labels/patch001.nii.gz,masks/patch001.nii.gz,{}
```

Paths are resolved relative to `root-dir`. For `num_classes=C`, the label column is named `segmentation_{C-1}classes`. Labels must be integer class indices from 0 to C−1 and align voxel-for-voxel with the image and mask. The input image is single-channel. All batch members must have the same spatial shape, matching the configured patch dimensions; the default four-scale network requires dimensions divisible by eight.

The current loader reads the entire named file. Although it parses the legacy `patches` column, it does not crop a larger volume with those coordinates. Use `{}` for already extracted patches; a recorded crop dictionary by itself does not extract them. The mask is loaded but the current normalization uses nonzero image voxels. Images, labels and masks must already share geometry.

`ses` identifies training sessions; `labelled_samples` selects this many sessions and uses the remaining sessions as unlabelled training data. Keep enough training sessions for both subsets. `val_samples` instead selects validation subjects via `sub`; setting it to zero uses all rows in the validation manifest. Keep subject assignments disjoint between training and validation when preparing the data. The historical API uses `seed` for the training-session split.

The included tests construct a synthetic example of this file contract. They establish that loading, a short train/save cycle and inference work; they do not validate clinical data quality, reproduce the original split, or establish the paper's segmentation scores.
