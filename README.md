# SKAB Sensor Anomaly Detection

### Isolation Forest + OC-SVM + LOF + AutoEncoder Ensemble

Sensor anomaly detection on the Skoltech Anomaly Benchmark (SKAB) dataset. Uses an ensemble approach combining Isolation Forest, One-Class SVM, Local Outlier Factor, and AutoEncoder with a LightGBM meta-learner for robust anomaly identification in industrial sensor time-series data.

---

## Architecture

**Multi-model ensemble with LightGBM stacking meta-learner**

| Component | Role |
|-----------|------|
| Isolation Forest | Unsupervised outlier detection via random partitioning |
| One-Class SVM | Kernel-based boundary estimation for normal data |
| Local Outlier Factor | Density-based local anomaly scoring |
| AutoEncoder (AE) | Reconstruction-error anomaly detection via neural network |
| LightGBM Meta-Learner | Stacks all model outputs for final anomaly classification |

---

## Features

- **Rolling statistics** — sliding window mean, std, min, max across sensor channels
- **FFT features** — frequency-domain energy and dominant frequency extraction
- **CUSUM** — cumulative sum control chart for shift detection
- **Cross-correlations** — inter-sensor lag correlations for multivariate anomaly patterns
- **Feature engineering** — time-series-specific transformations for improved signal quality

---

## Dataset

**SKAB — Skoltech Anomaly Benchmark**

Industrial sensor time-series data collected from a water circulation testbed at Skoltech. Contains labeled anomalies across multiple sensor channels (accelerometers, pressure, flow rate, temperature), making it suitable for benchmarking multivariate anomaly detection algorithms.

- Source: [SKAB GitHub](https://github.com/waico/SKAB)
- Format: CSV time-series with anomaly labels

---

## Kaggle Notebook

Live notebook: [skab-sensor-anomaly-transformer-ae](https://www.kaggle.com/code/akobirmusaev/skab-sensor-anomaly-transformer-ae)

> **Note:** Model is currently running on Kaggle — results will be updated upon completion.

---

## Quick Start

```python
# Install dependencies
pip install scikit-learn lightgbm torch pandas numpy scipy

# Run the notebook
jupyter notebook skab-sensor-anomaly-transformer-ae.ipynb
```

---

## Results

| Metric | Score |
|--------|-------|
| F1 Score | *TBD — model running* |
| ROC-AUC | *TBD — model running* |
| Precision | *TBD — model running* |
| Recall | *TBD — model running* |

Results will be updated once the Kaggle run completes.

---

## License

MIT
