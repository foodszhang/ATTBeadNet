# Traditional bead-detection baselines

`scripts/traditional_baselines.py` reproduces the two non-deep-learning comparison methods used for the two-size bead experiment.

## Methods

- `imagej_like`: percentile normalization, white top-hat background suppression, Gaussian smoothing, Otsu/intensity thresholding, morphology, connected-component area/circularity filtering, and supplementary Laplacian-of-Gaussian blob detection. Component area/equivalent diameter and LoG scale separate `1.0` from `2.8` beads.
- `svm_patch`: white top-hat local-maximum candidates followed by a standardized RBF-SVM patch classifier. The three labels are background, `1.0`, and `2.8`; positive samples come from annotated centers and negative samples are candidate peaks more than six pixels from every annotation.

Both methods use the same class-aware Hungarian center matching and emit per-image metrics, aggregate metrics, confusion matrices, predicted centers, and optional overlays.

## Run

```bash
python scripts/traditional_baselines.py \
  --raw-root datasets/20260512 \
  --processed-root datasets/processed/20260512_tiles_96 \
  --out-dir runs/traditional_baselines \
  --methods imagej_like svm_patch \
  --eval-split internal_val \
  --match-radius 3 \
  --min-distance 2 \
  --save-overlays
```

The processed dataset must contain a `manifest.csv` with `split` and `source_image_id` columns. Use `--eval-split all` to evaluate every TIFF in `RF/` without selecting an internal split.

## Historical reference results

The original internal validation used images `4`, `5`, and `18` with a 3-pixel matching radius:

| Method | Class | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| ImageJ-like | 1.0 | 0.7198 | 0.7773 | 0.7475 |
| ImageJ-like | 2.8 | 0.5673 | 0.8725 | 0.6875 |
| SVM patch | 1.0 | 0.6919 | 0.6727 | 0.6822 |
| SVM patch | 2.8 | 0.5393 | 0.8411 | 0.6572 |

These values are recorded for provenance; rerun the script when data, preprocessing, or dependencies change.
