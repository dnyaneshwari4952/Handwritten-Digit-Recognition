# Artifacts Directory

This directory stores generated model checkpoints, evaluation metrics, visual plots, and sample prediction assets.

## Directory Structure

```text
artifacts/
├── models/
│   ├── mnist_cnn.keras            # Production CNN model checkpoint (421k params, ~99.2% acc)
│   └── mnist_baseline_mlp.keras   # Baseline MLP model checkpoint (109k params, ~97.6% acc)
├── metrics/
│   ├── evaluation_metrics.json    # Complete test set evaluation (accuracy, F1, confusion matrix)
│   ├── training_history.json      # Epoch-by-epoch loss and accuracy curves
│   └── baseline_metrics.json      # Comparative baseline performance metrics
├── plots/
│   ├── training_history.png       # Loss & accuracy convergence plots
│   ├── confusion_matrix.png       # 10x10 annotated heatmap
│   ├── sample_digits_grid.png     # MNIST dataset sample grid
│   └── misclassified_examples.png # Error analysis gallery
└── predictions/
    └── sample_digit_1.png         # Sample digit for CLI & API smoke testing
```

## Reproducibility

All artifacts in `plots/` and `metrics/` can be deterministically re-generated via the CLI:

```bash
# Re-evaluate model and generate metrics
python -m src.cli evaluate

# Re-generate all publication plots
python -m src.cli visualize
```
