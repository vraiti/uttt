#!/usr/bin/env python3
"""Test that all components work before training."""
import torch
from uttt_engine import UTTTGame
from alphazero import UTTTNet, MCTS

print("Testing AlphaZero setup...")
print("=" * 60)

# Test 1: Game engine
print("\n1. Testing game engine...")
game = UTTTGame()
print(f"   ✓ Created game")
print(f"   ✓ Legal moves: {len(game.legal_moves())}")
game2 = game.make_move(4, 4)  # Play center
print(f"   ✓ Made move, new legal moves: {len(game2.legal_moves())}")

# Test 2: Neural network
print("\n2. Testing neural network...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"   Device: {device}")
model = UTTTNet(num_channels=64, num_res_blocks=2).to(device)
print(f"   ✓ Created model")

# Test forward pass
from alphazero.network import encode_state
state = encode_state(game).unsqueeze(0).to(device)
print(f"   State shape: {state.shape}")
log_policy, value = model(state)
print(f"   ✓ Forward pass: policy shape {log_policy.shape}, value shape {value.shape}")
print(f"   Value: {value.item():.4f}")

# Test 3: MCTS
print("\n3. Testing MCTS...")
mcts = MCTS(model, device=device, num_simulations=10)  # Low sims for testing
print(f"   ✓ Created MCTS with 10 simulations")
policy, move = mcts.search(game)
print(f"   ✓ Search completed")
print(f"   Best move: ({move[0]}, {move[1]})")
print(f"   Policy sum: {policy.sum():.4f}")

# Test 4: Self-play game
print("\n4. Testing self-play...")
from alphazero import AlphaZeroTrainer
trainer = AlphaZeroTrainer(
    model=model,
    device=device,
    num_simulations=10,
    num_selfplay_games=1,
    num_epochs=1,
    batch_size=32,
)
print(f"   ✓ Created trainer")
game_data = trainer.self_play_game(temperature_threshold=5)
print(f"   ✓ Played one game: {len(game_data)} positions")

print("\n" + "=" * 60)
print("✅ All tests passed! Ready to train.")
print("=" * 60)
print("\nTo start training:")
print("  python train.py --iterations 10 --games-per-iter 10 --simulations 100")
