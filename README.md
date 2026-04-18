# AlphaZero for Ultimate Tic Tac Toe

This project implements an AlphaZero-style reinforcement learning agent for Ultimate Tic Tac Toe, using:
- Fast Rust game engine (from `ultimattt/`)
- PyO3 Python bindings
- PyTorch neural networks
- Monte Carlo Tree Search (MCTS)
- Self-play training

## Architecture

### Neural Network
- ResNet-style architecture with configurable depth
- **Input**: 7-channel 9×9 board representation
  - Current player pieces
  - Opponent pieces
  - Legal moves mask
  - Board constraints
  - Won boards (current/opponent)
  - Drawn boards
- **Output**: 
  - Policy head: 81-dim probability distribution over all positions
  - Value head: Scalar win probability in [-1, 1]

### MCTS
- Uses neural network for leaf evaluation
- UCB1-based tree traversal
- Temperature-controlled action selection
- ~800 simulations per move (configurable)

### Training Loop
1. **Self-play**: Generate games using current model + MCTS
2. **Experience replay**: Store (state, policy, outcome) tuples
3. **Training**: Optimize network on replay buffer
4. **Iteration**: Repeat to improve

## Installation

### Prerequisites
- Rust nightly (for the game engine)
- Python 3.8+
- CUDA toolkit (for GPU acceleration)

### Build

```bash
# Install Rust (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default nightly

# Install Python dependencies and build bindings
pip install maturin torch numpy tqdm

# Build the Rust bindings
maturin develop --release
```

## Usage

### Training

Start training from scratch:
```bash
python train.py --iterations 100 --games-per-iter 100 --simulations 800
```

Key arguments:
- `--iterations`: Number of training iterations (default: 100)
- `--games-per-iter`: Self-play games per iteration (default: 100)
- `--simulations`: MCTS simulations per move (default: 800)
- `--epochs`: Training epochs per iteration (default: 10)
- `--batch-size`: Training batch size (default: 256)
- `--lr`: Learning rate (default: 0.001)
- `--num-channels`: Network width (default: 256)
- `--num-res-blocks`: Network depth (default: 10)
- `--device`: cuda or cpu (default: cuda)

Resume from checkpoint:
```bash
python train.py --resume checkpoints/model_iter_50.pt
```

### Playing Against the AI

```bash
python play.py checkpoints/model_iter_100.pt --simulations 800
```

Options:
- `--ai-first`: Let AI play first (you play as O)
- `--simulations`: MCTS simulations for AI moves
- `--device`: cuda or cpu

### Move Format
Moves are specified as `global,local` where:
- `global` (0-8): Which of the 9 boards (row-major order)
- `local` (0-8): Position within that board (row-major order)

```
Board layout:
0 1 2
3 4 5
6 7 8
```

## Performance Notes

### GPU Acceleration
- The RTX 4060 will accelerate neural network inference and training
- Batch inference during MCTS can provide significant speedup
- Expected training time: ~1-2 hours per 100 iterations (depends on settings)

### Hyperparameter Tuning
- **More simulations** → stronger play, slower training
- **Larger network** → more capacity, slower training
- **More games** → better exploration, slower iterations
- **Larger buffer** → more diverse training data, more memory

### Recommended Settings for RTX 4060
```bash
python train.py \
  --iterations 200 \
  --games-per-iter 50 \
  --simulations 400 \
  --num-channels 256 \
  --num-res-blocks 10 \
  --batch-size 256
```

This balances training speed with model quality.

## Project Structure

```
uttt/
├── alphazero/
│   ├── __init__.py
│   ├── network.py       # Neural network architecture
│   ├── mcts.py          # MCTS implementation
│   └── trainer.py       # Training loop
├── src/
│   └── lib.rs           # PyO3 bindings
├── ultimattt/           # Rust game engine (submodule)
├── train.py             # Training script
├── play.py              # Play against AI
├── Cargo.toml           # Rust build config
└── pyproject.toml       # Python build config
```

## References

- [AlphaZero paper](https://arxiv.org/abs/1712.01815)
- [Ultimate Tic Tac Toe rules](https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe)
- [Original UTTT engine](https://github.com/nelhage/ultimattt)

## License

MIT License (see individual component licenses)
