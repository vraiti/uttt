import os
import pickle
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .network import UTTTNet, encode_state
from .optimized_mcts import BatchedMCTS


class ReplayDataset(Dataset):
    """Optimized dataset for replay buffer."""
    def __init__(self, data):
        # Pre-convert to numpy arrays
        states = []
        policies = []
        values = []

        for state, policy, value in data:
            if isinstance(state, np.ndarray):
                states.append(state)
            else:
                states.append(state.numpy())
            policies.append(policy)
            values.append(value)

        self.states = np.stack(states)
        self.policies = np.array(policies)
        self.values = np.array(values)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.policies[idx]),
            torch.tensor(self.values[idx], dtype=torch.float32)
        )


def self_play_worker_optimized(gpu_id, model_state, model_params, num_games, num_simulations, c_puct, temperature_threshold, batch_size, return_queue):
    """
    Optimized worker function with batched MCTS.
    """
    try:
        import torch
        from uttt_engine import UTTTGame

        torch.cuda.set_device(gpu_id)
        device = f'cuda:{gpu_id}'

        # Load model with correct parameters
        num_channels, num_res_blocks = model_params
        model = UTTTNet(num_channels=num_channels, num_res_blocks=num_res_blocks).to(device)
        model.load_state_dict(model_state)
        model.eval()

        # Create batched MCTS
        mcts = BatchedMCTS(
            model,
            device=device,
            num_simulations=num_simulations,
            c_puct=c_puct,
            batch_size=batch_size
        )

        all_game_data = []

        for game_idx in range(num_games):
            game = UTTTGame()
            game_data = []
            move_count = 0

            while not game.is_terminal():
                temperature = 1.0 if move_count < temperature_threshold else 0.1
                mcts.temperature = temperature
                policy, move = mcts.search(game)

                state = encode_state(game)
                game_data.append((state.cpu().numpy(), policy, game.current_player()))

                game = game.make_move(move[0], move[1])
                move_count += 1

            result = game.get_result()
            if result is None:
                result = 0

            for state, policy, player in game_data:
                value = result * player
                all_game_data.append((state, policy, value))

        return_queue.put((gpu_id, all_game_data))

    except Exception as e:
        import traceback
        return_queue.put((gpu_id, []))
        print(f"GPU {gpu_id} worker error: {e}")
        traceback.print_exc()


class OptimizedAlphaZeroTrainer:
    def __init__(
        self,
        model=None,
        device='cuda:0',
        num_gpus=None,
        num_simulations=800,
        c_puct=1.0,
        num_selfplay_games=100,
        num_epochs=10,
        batch_size=1024,
        learning_rate=0.001,
        replay_buffer_size=500000,
        checkpoint_dir='checkpoints',
        mcts_batch_size=64,
        num_workers=4,
        num_channels=256,
        num_res_blocks=10,
    ):
        """
        Optimized AlphaZero trainer with batched MCTS and mixed precision.

        Args:
            model: Neural network (if None, creates new UTTTNet)
            device: Primary device for training
            num_gpus: Number of GPUs to use
            num_simulations: MCTS simulations per move
            c_puct: MCTS exploration constant
            num_selfplay_games: Number of self-play games per iteration
            num_epochs: Training epochs per iteration
            batch_size: Training batch size
            learning_rate: Optimizer learning rate
            replay_buffer_size: Maximum size of experience replay buffer
            checkpoint_dir: Directory to save checkpoints
            mcts_batch_size: Batch size for MCTS evaluation
            num_workers: DataLoader worker threads
        """
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.num_selfplay_games = num_selfplay_games
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.mcts_batch_size = mcts_batch_size
        self.num_workers = num_workers
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks

        # Auto-detect GPUs
        if num_gpus is None:
            self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        else:
            self.num_gpus = num_gpus

        print(f"Optimized trainer initialized:")
        print(f"  GPUs: {self.num_gpus}")
        print(f"  MCTS batch size: {mcts_batch_size}")
        print(f"  Training batch size: {batch_size}")
        print(f"  Replay buffer: {replay_buffer_size:,}")

        # Initialize model
        if model is None:
            self.model = UTTTNet(num_channels=num_channels, num_res_blocks=num_res_blocks).to(device)
        else:
            self.model = model.to(device)

        # Compile model for optimization (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            print("  Compiling model with torch.compile...")
            self.model = torch.compile(self.model, mode='max-autotune')

        # Optimizer and mixed precision scaler
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scaler = torch.amp.GradScaler('cuda')

        # Replay buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        os.makedirs(checkpoint_dir, exist_ok=True)

    def generate_self_play_data(self, temperature_threshold=15):
        """Generate self-play games with optimized batched MCTS."""
        print(f"Generating {self.num_selfplay_games} self-play games (batched MCTS)...")

        # Get state dict from unwrapped model if compiled
        model_to_share = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        model_state = model_to_share.state_dict()
        model_params = (self.num_channels, self.num_res_blocks)

        # Distribute games
        games_per_gpu = [self.num_selfplay_games // self.num_gpus] * self.num_gpus
        for i in range(self.num_selfplay_games % self.num_gpus):
            games_per_gpu[i] += 1

        print(f"Games per GPU: {games_per_gpu}")

        # Multiprocessing setup
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass

        return_queue = mp.Queue()

        # Launch workers
        processes = []
        for gpu_id in range(self.num_gpus):
            p = mp.Process(
                target=self_play_worker_optimized,
                args=(
                    gpu_id,
                    model_state,
                    model_params,
                    games_per_gpu[gpu_id],
                    self.num_simulations,
                    self.c_puct,
                    temperature_threshold,
                    self.mcts_batch_size,
                    return_queue
                )
            )
            p.start()
            processes.append(p)

        # Collect results
        all_data = []
        completed_gpus = 0
        pbar = tqdm(total=self.num_gpus, desc="GPUs completed")

        while completed_gpus < self.num_gpus:
            gpu_id, game_data = return_queue.get()
            all_data.extend(game_data)
            completed_gpus += 1
            pbar.update(1)
            print(f"  GPU {gpu_id}: Generated {len(game_data)} positions")

        pbar.close()

        for p in processes:
            p.join()

        self.replay_buffer.extend(all_data)
        print(f"Replay buffer size: {len(self.replay_buffer):,}")

    def train_network(self):
        """Train network with mixed precision and optimized data loading."""
        if len(self.replay_buffer) < self.batch_size:
            print("Not enough data in replay buffer to train")
            return

        print(f"Training network for {self.num_epochs} epochs (mixed precision)...")
        self.model.train()

        # Create dataset and dataloader
        dataset = ReplayDataset(list(self.replay_buffer))
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

        for epoch in range(self.num_epochs):
            epoch_losses = []
            epoch_policy_losses = []
            epoch_value_losses = []

            for states, target_policies, target_values in dataloader:
                states = states.to(self.device, non_blocking=True)
                target_policies = target_policies.to(self.device, non_blocking=True)
                target_values = target_values.unsqueeze(1).to(self.device, non_blocking=True)

                # Mixed precision forward pass
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    log_policies, values = self.model(states)

                    # Policy loss: cross-entropy
                    policy_loss = -torch.sum(target_policies * log_policies) / self.batch_size

                    # Value loss: MSE
                    value_loss = torch.mean((values - target_values) ** 2)

                    # Total loss
                    loss = policy_loss + value_loss

                # Mixed precision backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_losses.append(loss.item())
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())

            avg_loss = np.mean(epoch_losses)
            avg_policy_loss = np.mean(epoch_policy_losses)
            avg_value_loss = np.mean(epoch_value_losses)

            print(f"Epoch {epoch + 1}/{self.num_epochs} - "
                  f"Loss: {avg_loss:.4f} (Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f})")

    def train(self, num_iterations=100):
        """Main training loop."""
        for iteration in range(num_iterations):
            print(f"\n{'=' * 60}")
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"{'=' * 60}")

            self.generate_self_play_data()
            self.train_network()

            if (iteration + 1) % 5 == 0:
                self.save_checkpoint(iteration + 1)

        print("\nTraining complete!")

    def save_checkpoint(self, iteration):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f'model_iter_{iteration}.pt')
        buffer_path = os.path.join(self.checkpoint_dir, f'buffer_iter_{iteration}.pkl')

        # Unwrap compiled model if needed
        model_to_save = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model

        torch.save({
            'iteration': iteration,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)

        with open(buffer_path, 'wb') as f:
            pickle.dump(list(self.replay_buffer)[-50000:], f)  # Save last 50k samples

        print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path, load_buffer=True):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Unwrap compiled model if needed
        model_to_load = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        iteration = checkpoint['iteration']
        print(f"Loaded checkpoint from iteration {iteration}")

        if load_buffer:
            buffer_path = checkpoint_path.replace('model_', 'buffer_').replace('.pt', '.pkl')
            if os.path.exists(buffer_path):
                with open(buffer_path, 'rb') as f:
                    buffer_data = pickle.load(f)
                    self.replay_buffer.extend(buffer_data)
                print(f"Loaded replay buffer with {len(self.replay_buffer):,} samples")

        return iteration
