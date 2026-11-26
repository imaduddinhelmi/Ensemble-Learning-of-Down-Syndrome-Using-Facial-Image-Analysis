# Ensemble Learning Optimization for Early Screening of Down Syndrome Using Facial Image Analysis

Code, data split, per-image predictions, and training logs for the revised manuscript (BEEI, Paper ID 13183).

## Dataset

The images come from the public Kaggle **Down Syndrome Dataset**:
<https://www.kaggle.com/datasets/mamunhasan2cs/down-syndrome-dataset> (2,999 images; `downSyndrome/` and `healthy/`).
At the time of revision the dataset license was listed as "Unknown", so **no images are redistributed here**. Download them from Kaggle.
Trained model weights are available from the corresponding author on reasonable request.

## Contents

```
code/
  ds_pipeline.py                        pipeline: pHash near-duplicate grouping, seeded group-aware split,
                                        model building (5 backbones, concatenation ensembles), training,
                                        soft voting, metrics + CIs, McNemar/Holm, paired AUC bootstrap,
                                        Youden threshold, compute benchmark, Grad-CAM
  Retraining_DownSyndrome_Colab.ipynb   notebook that runs the full experiment (Google Colab / any GPU machine)
results/
  split_seed42.csv                      subset (train/val/test) and duplicate group of every image (seed 42)
  dedup_report.json                     near-duplicate screening summary
  ablation_augmentation.csv             augmentation ablation on DenseNet-201 (Table 2)
  summary_test.csv                      test metrics with 95% CIs, threshold 0.5 and validation-Youden (Table 3)
  paired_tests_vs_D_X__none.csv         exact McNemar (Holm) and paired AUC differences vs DenseNet-201+Xception
  all_predictions.csv                   per-image validation and test probabilities for all 35 configurations (10 trained models + 25 soft-voting combinations)
  compute.csv                           parameters, FP32/TFLite size, CPU/GPU latency (Table 5)
  gradcam_cases.csv                     images used for the Grad-CAM figure
  histories/                            per-epoch training logs
  meta/                                 run metadata (seed, learning rate, batch size, epochs, TF version)
  figures/                              ROC/PR curves and test confusion matrices
```

Probabilities are P(Down syndrome); in `y`, 1 = Down syndrome, 0 = healthy.
Configuration codes: `D` DenseNet-201, `X` Xception, `I3` InceptionV3, `IR` InceptionResNetV2, `CNT` ConvNeXt-Tiny;
`D_X` = feature concatenation of D and X; `SV(D+X)` = soft voting; suffix `__none/__original/__conservative` = augmentation policy.

## Reproducing

Requirements: Python 3.10+, TensorFlow 2.21 (Keras 3), numpy, pandas, scikit-learn, scipy, statsmodels, Pillow, ImageHash.

1. Download and unzip the Kaggle dataset so that `DownSyndrome/downSyndrome/` and `DownSyndrome/healthy/` exist.
2. Open `code/Retraining_DownSyndrome_Colab.ipynb`, set `BASE`/`DATA` to your paths, and run all cells.
   Training is resumable; the split is regenerated deterministically from seed 42 and should match `results/split_seed42.csv`.

## Ethics note

The images are of children. Any reuse requires independent verification of provenance, consent or another lawful basis,
permissions, and applicable ethics requirements. The models are research prototypes, not diagnostic devices.
