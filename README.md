# PINN C2P Solver

Physics-Informed Neural Networks for Conservative-to-Primitive Variable Conversion in Relativistic Hydrodynamics.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pinn-c2p-solver.git
cd pinn-c2p-solver

# Install in development mode
pip install -e .

# Or install with all extras
pip install -e ".[dev,viz]"
```

### Basic Training

```bash
# Train with default configuration
python scripts/train_pinn.py

# Train with custom configuration
python scripts/train_pinn.py --config configs/large_pinn.yaml

# Resume from checkpoint
python scripts/train_pinn.py --checkpoint experiments/checkpoints/best_model.pth
```

### Project Structure

```
├── src/                    # Core source code
│   ├── physics/           # Physics modules (metric, EOS, utils)
│   ├── models/            # Neural network architectures
│   ├── data/              # Data generation and datasets
│   ├── training/          # Training utilities and losses
│   ├── solvers/           # C2P solver implementations
│   └── visualization/     # Plotting and analysis tools
├── scripts/               # Executable training scripts
├── configs/               # Configuration files
├── experiments/           # Results and checkpoints
└── notebooks/            # Jupyter notebooks
```

## 🎯 Features

- **Fast Training**: Optimized for multi-GPU training with mixed precision support
- **Modular Design**: Easy to swap models, loss functions, and data generators
- **Physics-Informed**: Incorporates physical constraints directly into the loss
- **Comprehensive Metrics**: Detailed error analysis and performance benchmarking
- **Flexible Configuration**: YAML-based configuration system
- **Checkpointing**: Automatic model saving and resumable training

## 📊 Configuration

Edit `configs/default.yaml` to customize:

```yaml
# Model architecture
model:
  type: "PhysicsGuided_TinyPINNvKen"
  correction_scale: 0.2
  hidden_dims: [64, 32]

# Training parameters
training:
  num_epochs: 200
  batch_size: 256
  learning_rate: 0.005
  physics_weight: 0.1

# Data generation
data:
  n_train: 1000
  n_val: 200
  lrho_min: -8
  lrho_max: -2.7
  W_min: 1.0
  W_max: 1.5
```

## 🔧 Advanced Usage

### Multi-GPU Training

The script automatically detects and uses all available GPUs:

```python
# Automatic multi-GPU detection
if torch.cuda.device_count() > 1:
    model = DataParallel(model)
```

### Custom Models

Add new models in `src/models/`:

```python
class MyCustomPINN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # Define your architecture
    
    def forward(self, x):
        # Forward pass
        return output
    
    def physics_loss(self, C, Z_pred, eos):
        # Optional: Custom physics loss
        return loss
```

### Custom Loss Functions

Add new losses in `src/training/losses.py`:

```python
def my_custom_loss(pred, target):
    # Your loss implementation
    return loss
```

## 📈 Performance

On AMD MI300A GPUs:
- Training speed: ~10k samples/second
- Inference speed: ~100k samples/second
- Memory usage: <2GB for typical configurations

## 🧪 Testing

Run unit tests:

```bash
pytest tests/
```

Run benchmarks:

```bash
python scripts/benchmark.py
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper,
  title={Physics-Informed Neural Networks for C2P Conversion},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 💡 Tips for Best Performance

1. **Data Generation**: Use `meshgrid_cold` for structured sampling, `random` for diverse coverage
2. **Batch Size**: Larger batches (512-1024) work well with multi-GPU setups
3. **Learning Rate**: Start with 5e-3 and use scheduler for automatic adjustment
4. **Physics Weight**: 0.1 is a good starting point, adjust based on validation metrics
5. **Early Stopping**: Set patience to 20-30 epochs to avoid overfitting

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce batch size in config
- Enable mixed precision training (`use_amp: true`)
- Use gradient accumulation for effective larger batches

### Slow Training
- Enable GPU acceleration
- Increase number of data loader workers
- Use persistent_workers: true
- Check for CPU bottlenecks with profiler

### Poor Convergence
- Adjust learning rate (try 1e-3 to 1e-2 range)
- Increase physics_weight gradually
- Use more training data
- Try different model architectures

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [your.email@example.com]