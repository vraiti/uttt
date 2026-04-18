# Quick Start Guide

## ✅ Setup Complete!

Your AlphaZero training system is ready to use. All tests passed.

## System Info
- **GPU**: RTX 4060 (CUDA available)
- **Game Engine**: Rust-based UTTT (fast C-speed execution)
- **ML Framework**: PyTorch with CUDA acceleration
- **Training**: Self-play + MCTS + ResNet

## Quick Commands

### 1. Verify Installation
```bash
python test_setup.py
```

### 2. Start Training (Quick Test)
```bash
# Small test run (5-10 minutes)
python train.py \
  --iterations 5 \
  --games-per-iter 20 \
  --simulations 100 \
  --num-channels 128 \
  --num-res-blocks 5
```

### 3. Full Training (Recommended)
```bash
# Longer training for better results (2-3 hours)
python train.py \
  --iterations 100 \
  --games-per-iter 50 \
  --simulations 400 \
  --num-channels 256 \
  --num-res-blocks 10 \
  --batch-size 256 \
  --buffer-size 50000
```

### 4. Resume Training
```bash
python train.py \
  --resume checkpoints/model_iter_50.pt \
  --iterations 100
```

### 5. Play Against AI
```bash
# After training, play against your model
python play.py checkpoints/model_iter_100.pt --simulations 400
```

## Training Tips

### GPU Memory
The RTX 4060 (8GB) can handle:
- Network: 256 channels, 10 res blocks
- Batch size: 256-512
- Simulations: 400-800 per move

If you get OOM errors, reduce:
```bash
--num-channels 128 --batch-size 128
```

### Speed vs Quality
- **Faster**: Fewer simulations (100-200), smaller network (128 channels, 5 blocks)
- **Better**: More simulations (600-800), larger network (256+ channels, 10+ blocks)

### Monitoring
Checkpoints saved every 5 iterations to `checkpoints/`:
- `model_iter_N.pt` - Model weights
- `buffer_iter_N.pkl` - Replay buffer

Watch for:
- Policy loss decreasing
- Value loss stabilizing around 0.5-1.0
- Replay buffer growing (means diverse positions)

## Next Steps

1. **Run test training**: Start with 5 iterations to verify GPU usage
2. **Check GPU utilization**: `nvidia-smi` should show high usage during training
3. **Let it train**: 100+ iterations for competent play
4. **Evaluate**: Play against it to see improvement
5. **Iterate**: Tune hyperparameters based on results

## Hyperparameter Guide

| Parameter | Effect | RTX 4060 Sweet Spot |
|-----------|--------|---------------------|
| `--simulations` | MCTS depth/strength | 400-600 |
| `--num-channels` | Network capacity | 256 |
| `--num-res-blocks` | Network depth | 10 |
| `--games-per-iter` | Data diversity | 50-100 |
| `--batch-size` | Training stability | 256 |
| `--lr` | Learning speed | 0.001-0.0005 |
| `--buffer-size` | Memory diversity | 50000 |

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce network size
--num-channels 128 --batch-size 128

# Or reduce simulations
--simulations 200
```

### Training Too Slow
```bash
# Reduce simulations
--simulations 200 --games-per-iter 25
```

### Model Not Improving
- Increase replay buffer: `--buffer-size 100000`
- More iterations: `--iterations 200`
- Lower learning rate: `--lr 0.0005`

## File Structure
```
uttt/
├── checkpoints/          # Saved models
├── alphazero/           # Training code
├── train.py             # Main training script
├── play.py              # Play vs AI
├── test_setup.py        # Verify installation
└── README.md            # Full documentation
```

Happy training! 🚀
