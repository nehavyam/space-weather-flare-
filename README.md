#  Solar Flare Prediction with a Small CNN


A lightweight deep learning project that predicts **strong solar flares** from **solar magnetic field images**, with an interactive Streamlit web app for running predictions.



---

## Overview

Strong solar flares can disrupt satellites, GPS, radio communication, and power grids. Flares often originate in regions of the Sun with complex magnetic fields, which makes magnetic imagery a useful signal for prediction.

This project applies a small convolutional neural network (CNN), implemented in PyTorch, to classify magnetic images by flare likelihood. A preprocessing step smooths each image before it reaches the model, and a Streamlit app makes the trained model easy to try.

## Features

- Flare prediction from solar magnetic images
- Image preprocessing (blur) to reduce noise
- Compact PyTorch CNN, quick to train and run
- Reproducible data generation and training scripts
- Interactive Streamlit web interface

## Pipeline

```
Magnetic image → Blur → CNN → Flare prediction → Streamlit app
```

## Project Structure

```
space-weather-flare-cnn/
├── app/
│   └── app.py                # Streamlit web app
├── src/
│   ├── image_processing.py   # Image blurring / preprocessing
│   ├── generate_data.py      # Training data generation
│   ├── model.py              # CNN architecture
│   └── train.py              # Training script
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/nehavyam/space-weather-flare-.git
cd space-weather-flare-cnn

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install torch streamlit numpy pillow
```

## Usage

### Run the app

```bash
streamlit run app/app.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`).

### Train the model

Generate the training data, then train:

```bash
python src/generate_data.py
python src/train.py
```

## Tech Stack

- **Python**
- **PyTorch** for the CNN
- **Streamlit** for the web interface

## Limitations

- This is a hackathon prototype and is **not** intended for real space weather forecasting.
- The model is intentionally small, favoring speed and simplicity over accuracy.
- Training images come from `generate_data.py`, so performance on real solar data may differ.

## Future Work

- Train and evaluate on real solar magnetogram data
- Report accuracy, precision, and recall in the app
- Experiment with deeper architectures
- Add visual explanations of what the model focuses on

## Acknowledgements

Some AI assistance was used for the PyTorch implementation.
