# Spiking Neural Network for Energy-Efficient Anomaly Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![SpikingJelly](https://img.shields.io/badge/SpikingJelly-0.0.0.0.14-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Abstract

Conventional deep learning-based anomaly detectors rely on continuous-valued activations, incurring substantial multiply-accumulate (MAC) operations that are prohibitive for edge deployment. This project investigates whether **Spiking Neural Networks (SNNs)** which communicate via discrete binary spike events rather than continuous activations can approximate the detection performance of standard autoencoder-based anomaly detectors while delivering a meaningful reduction in computational cost.

We conduct a systematic **energy-accuracy tradeoff analysis** comparing an SNN against a standard deep autoencoder across three benchmark anomaly detection datasets, measuring inference FLOPs, latency, and F1-score under equivalent training conditions. Findings are formalised as a Pareto optimisation problem and discussed in the accompanying white paper.

---

## Research Question

> *Can Spiking Neural Networks approximate the anomaly detection capability of conventional autoencoders while substantially reducing multiply-accumulate operations, and what is the nature of the energy-accuracy tradeoff?*

---

## Architecture Overview

```
Input Signal
     │
     ▼
┌─────────────────────────────┐
│   Leaky Integrate-and-Fire  │  ← Neuromorphic neuron model
│   (LIF) Encoder Layers      │     spike threshold: ϑ = 1.0
└────────────┬────────────────┘
             │  Binary spikes {0,1}
             ▼
┌─────────────────────────────┐
│   Latent Spike Representation│  ← Sparse temporal encoding
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   LIF Decoder Layers        │  ← Reconstruction via spike rates
└────────────┬────────────────┘
             │
             ▼
     Reconstruction Error
     (Anomaly Score Threshold)
```

**Baseline comparison:** Standard Autoencoder (ReLU activations, MSE reconstruction loss)

---

## Datasets

| Dataset | Type | Samples | Anomaly Ratio |
|---|---|---|---|
| [KDD Cup 1999](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) | Network intrusion | 494,021 | 19.7% |
| [MNIST-AD](http://yann.lecun.com/exdb/mnist/) | Image (one-class) | 70,000 | 10% |
| [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) | Industrial inspection | 5,354 | variable |

---

## Planned Experiments

- [ ] Train SNN encoder-decoder on all 3 datasets
- [ ] Train equivalent standard AE on same datasets
- [ ] Measure inference FLOPs via `thop` (PyTorch profiler)
- [ ] Measure wall-clock latency (CPU and simulated neuromorphic)
- [ ] Compare F1, AUROC, and precision-recall across models
- [ ] Plot Pareto frontier: FLOPs reduction vs F1 retention
- [ ] Ablation: timestep count T ∈ {4, 8, 16, 32} vs performance

---

## Installation

```bash
git clone https://github.com/samiurk70/snn-anomaly-detection.git
cd snn-anomaly-detection
pip install -r requirements.txt
```

**Core dependencies:**
```
torch>=2.0.0
spikingjelly>=0.0.0.0.14
numpy>=1.24
scikit-learn>=1.2
matplotlib>=3.7
thop  # FLOPs profiling
```

---

## Usage

```python
# Train SNN model
python train_snn.py --dataset kdd --timesteps 16 --epochs 50

# Train baseline autoencoder
python train_ae.py --dataset kdd --epochs 50

# Run comparative benchmark
python benchmark.py --model both --dataset kdd --metrics flops latency f1
```

---

## Repository Structure

```
snn-anomaly-detection/
├── models/
│   ├── snn_autoencoder.py       # LIF-based encoder-decoder
│   └── ae_baseline.py           # Standard autoencoder baseline
├── data/
│   └── loaders.py               # Dataset loaders
├── experiments/
│   ├── train_snn.py
│   ├── train_ae.py
│   └── benchmark.py
├── analysis/
│   └── pareto_plot.py           # Energy-accuracy Pareto frontier
├── whitepaper/
│   └── snn_anomaly_whitepaper.pdf
├── requirements.txt
└── README.md
```

---

## Paper Submission Status

Sumbitted to ACM conference ICONS2026
**Title:** *T=8 or Not T=8: Pareto-Optimal Timestep Selection for Energy-Efficient SNN Anomaly Detection*

---

## References

1. Mahowald, M., & Douglas, R. (1991). A silicon neuron. *Nature*, 354(6354), 515–518.
2. Fang, W., et al. (2023). SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence. *Science Advances*.
3. Peng, F., et al. (2023). Spiking neural networks for anomaly detection: A survey. *arXiv preprint*.
4. Ruff, L., et al. (2021). A unifying review of deep and shallow anomaly detection. *Proceedings of the IEEE*.

---

## Author

**Samiur Rahman Khan**

MSc Data Science (Distinction) — Middlesex University London

MSc Computer Network & Architecture (Summa Cum Laude) - American International University Bangladesh

[scholar.google.co.uk/citations?user=ddQa7D4AAAAJ](https://scholar.google.co.uk/citations?user=ddQa7D4AAAAJ&hl)

[linkedin.com/in/samiurk70](https://linkedin.com/in/samiurk70)

---

*This research is conducted as part of independent post-MSc research. Code and full results will be released upon completion of experiments.*
