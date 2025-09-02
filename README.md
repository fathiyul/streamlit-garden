# Streamlit Garden 🌱

A collection of interactive Streamlit applications I made for fun.

## 🛠️ Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

1. Clone the repository:
   ```bash
   git clone git@github.com:fathiyul/streamlit-garden.git
   cd streamlit-garden
   ```

2. Initialize and sync dependencies:
   ```bash
   uv init
   uv sync
   ```

## 🚀 Usage

Run any application using the following pattern:

```bash
uv run streamlit run <path_to_app>
```

### Examples

```bash
# Image recolorizer
uv run streamlit run apps/image_processing/app_recolorizer.py

# Bayesian inference demo
uv run streamlit run apps/math_demo/bayes_inference.py
```

## 📋 Applications

### Image Processing
- **[Two-Color Recolorizer](apps/image_processing/image_recolorizer.py)** - Convert images to grayscale and apply custom two-color gradients with adjustable parameters

### Mathematical Demos
- **[Bayesian Inference](apps/math_demo/bayes_inference.py)** - Interactive Bayesian probability inference with Beta priors and posterior visualization