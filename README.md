<p align="center">
  <img src="ensemble_logo.jpg" alt="Logo" width="400">
  <br /> <br / >
</p>

# Predictive Maintenance on a NASA dataset with NdLinear- Arshad Shaik

This repository presents a complete predictive maintenance pipeline for fatigue monitoring in composite materials, using the parameter-efficient **NdLinear** layer to build a high-performance model with reduced resource requirements.

## Project Summary

Industrial components such as aircraft engine parts and composite structures undergo cyclic loading that leads to fatigue damage over time. Early detection of damage can prevent costly failures and downtime. In this project, I have:

1. Ingested multivariate sensor data (strain gauges, Lamb-wave signals) from fatigue tests conducted on composite coupons.
2. Implemented a PyTorch `Dataset` to load and window raw time-series data.
3. Built a baseline 1D convolutional neural network (CNN) that predicts fatigue cycle count from sensor signals.
4. Replace standard dense layers with `NdLinear` to reduce parameter count while maintaining or improving accuracy.
5. Train, validate, and profile the model, comparing metrics such as loss, parameter count, and inference latency.
6. Containerize the training pipeline with Docker and prepare for edge deployment.

## Data Description

Data originates from the Stanford Structures and Composites Laboratory and NASA Ames PCoE. Each coupon undergoes tension–tension fatigue testing under controlled load cycles. The dataset includes:

- **Strain data** from gauge sensors and MTS machine logs.
- **Lamb-wave signals** recorded by PZT actuators and sensors at multiple frequencies.
- **X-ray imagery** captured at intervals for ground-truth damage assessment.
- **Cycle metadata** indicating the number of loading cycles at each measurement.

Data is organized into three layup configurations (`layup1`, `layup2`, `layup3`), each containing multiple coupon experiments.

## Repository Structure

```plain
NdLinear-PM/
├── data/               # Raw input data folders per layup
├── src/
│   ├── data_loader.py  # Custom Dataset for loading .mat, CSV, and Excel files
│   ├── models.py       # Model definitions (baseline and NdLinear-enabled)
│   ├── train.py        # Training script with argument parsing
│   └── inference.py    # (Optional) Deployment and inference utilities
├── notebooks/          # Exploration and visualization notebooks
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container specification for training
└── README.md           # This document
```

## Setup Instructions

1. **Clone the repository** and install dependencies:
   ```bash
   clone the repo to your local
   cd NdLinear-PM
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Prepare data**: Place the `layup1`, `layup2`, and `layup3` folders under `data/`.
3. **Run training** with default settings:
   ```bash
   python -m src.train --data-dir data --layup layup1 --window-size 1024 \
       --batch-size 32 --epochs 10 --lr 1e-3 --output-path model.pth
   ```
4. **Profile results**: Review training loss, parameter counts, and inference timing printed in the console.
5. **Explore notebooks** under `notebooks/` for signal visualization and performance analysis.

## Expected Outcomes

- A trained CNN model that predicts fatigue cycle count from raw sensor data.
- A comparative analysis showing model size reduction and potential accuracy gains after integrating `NdLinear`.
- A Docker container able to reproduce training on any compatible environment.

## Next Steps

- Extend the pipeline to include real-time anomaly detection on edge devices.
- Integrate telemetry dashboards to visualize anomaly scores and prediction trends.
- Evaluate performance on additional layup configurations or new datasets.

---

