# DDPM - Denoising Diffusion Probabilistic Model

A PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) for image generation, trained on CIFAR-10 dataset.

## Overview

This project implements a diffusion model based on the DDPM paper, featuring a U-Net architecture with residual blocks, attention mechanisms, and cosine noise scheduling. The model learns to generate images by gradually denoising from pure Gaussian noise.

## Features

- **Cosine Beta Schedule**: Improved noise scheduling for better sample quality
- **U-Net Architecture**: Deep convolutional network with skip connections
- **Attention Mechanisms**: Self-attention blocks at multiple resolution levels
- **Mixed Precision Training**: Faster training with automatic mixed precision (AMP)
- **Gradient Clipping**: Stable training with gradient norm clipping
- **Model Compilation**: Optimized performance with `torch.compile()`

## Architecture Details

### U-Net Components
- **Residual Blocks**: ConvNet blocks with GroupNorm and SiLU activation
- **Attention Blocks**: Self-attention for capturing long-range dependencies
- **Time Embeddings**: Sinusoidal position embeddings for timestep conditioning
- **Skip Connections**: Feature concatenation between encoder and decoder

### Model Configuration
- Base channels: 128
- Channel multipliers: (1, 2, 4, 8)
- Timesteps: 1000
- Image size: 32×32 (CIFAR-10)
- Attention levels: Applied at 4× and 8× channel multipliers

## Requirements

```bash
torch>=2.0.0
torchvision
numpy
matplotlib
Pillow
tqdm
```

## Installation

```bash
git clone https://github.com/yourusername/DDPM-Denoising-Diffusion-Probabilistic-Model.git
cd DDPM-Denoising-Diffusion-Probabilistic-Model
pip install -r requirements.txt
```

## Usage

### Training

```python
# The notebook automatically downloads CIFAR-10 and starts training
# Simply run all cells in diffusion_model.ipynb
```

### Key Training Parameters

```python
image_size = 32          # Image resolution
timesteps = 1000         # Number of diffusion steps
batch_size = 256         # Training batch size
num_epochs = 500         # Total training epochs
lr = 2e-5               # Learning rate
```

### Generating Samples

```python
# Generate 8 samples
samples = generate_samples(model, num_samples=8, device=device)

# Samples are automatically generated every 25 epochs during training
```

## Training Process

The model uses a multi-stage training approach:

1. **Forward Diffusion**: Gradually adds Gaussian noise to images over 1000 timesteps
2. **Noise Prediction**: U-Net learns to predict the noise added at each step
3. **Reverse Diffusion**: Generates images by iteratively denoising from pure noise

### Loss Function

```python
loss = MSE(predicted_noise, actual_noise)
```

## Optimizations

- **Mixed Precision Training**: Uses FP16 for faster computation
- **Gradient Scaling**: Prevents underflow in mixed precision
- **TF32 Support**: Enabled for NVIDIA Ampere+ GPUs
- **Model Compilation**: JIT compilation for optimized execution
- **Cosine Learning Rate Schedule**: Smooth learning rate decay

## Results

The model generates samples every 25 epochs, showing progressive improvement in image quality. Checkpoints are saved with:
- Model weights
- Optimizer state
- Training loss
- Epoch number

## File Structure

```
├── diffusion_model.ipynb    # Main training notebook
├── data/                     # CIFAR-10 dataset (auto-downloaded)
├── *.pth                     # Model checkpoints
└── README.md
```

## Model Architecture

```
UNet(
  Time Embedding → MLP(128 → 512 → 128)
  
  Encoder:
    Level 1: ResBlock(3→128) → Attention → Downsample
    Level 2: ResBlock(128→256) → Downsample
    Level 3: ResBlock(256→512) → Attention → Downsample
    Level 4: ResBlock(512→1024) → Attention
  
  Bottleneck:
    ResBlock(1024) → Attention → ResBlock(1024)
  
  Decoder:
    Level 4: Upsample → ResBlock(1024+1024→1024) → Attention
    Level 3: Upsample → ResBlock(1024+512→512) → Attention
    Level 2: Upsample → ResBlock(512+256→256)
    Level 1: ResBlock(256+128→128)
  
  Output: Conv(128→3)
)
```

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (Tesla T4 or better recommended)
- **VRAM**: Minimum 16GB for batch size 256
- **RAM**: 16GB+ recommended

## Performance

- Training time: ~4-5 seconds per batch (on Tesla T4)
- Sample generation: ~30 seconds for 8 images (1000 denoising steps)
- Checkpoint size: ~500MB

## References

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [Improved Denoising Diffusion Probabilistic Models (Nichol & Dhariwal, 2021)](https://arxiv.org/abs/2102.09672)

## License

MIT License

## Acknowledgments

- CIFAR-10 dataset from the Canadian Institute For Advanced Research
- PyTorch team for the deep learning framework
- Original DDPM paper authors

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ddpm-implementation,
  author = {Your Name},
  title = {DDPM - Denoising Diffusion Probabilistic Model Implementation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/DDPM-Denoising-Diffusion-Probabilistic-Model}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub or contact [shrey7shrey@gmail.com]
