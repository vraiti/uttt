import os
import pickle
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from tqdm import tqdm

from .network import UTTTNet, encode_state
from .mcts import MCTS


def self_play_worker(gpu_id, model_state, num_games, num_simulations, c_puct, temperature_threshold, return_queue):
    """
    Worker function to play self-play games on a specific GPU.

    Args:
        gpu_id: GPU device ID
        model_state: Model state dict
        num_games: Number of games to play
        num_simulations: MCTS simulations per move
        c_puct: Exploration constant
        temperature_threshold: Temperature threshold for move selection
        return_queue: Queue to return results
    """
    try:
        import torch
        from uttt_engine import UTTTGame

        # Set CUDA device
        torch.cuda.set_device(gpu_id)
        device = f'cuda:{gpu_id}'

        # Load model on this GPU
        model = UTTTNet().to(device)
        model.load_state_dict(model_state)
        model.eval()

        # Create MCTS
        mcts = MCTS(model, device=device, num_simulations=num_simulations, c_puct=c_puct)

        # Play games
        all_game_data = []

        for game_idx in range(num_games):
            game = UTTTGame()
            game_data = []
            move_count = 0

            while not game.is_terminal():
                # Use temperature for exploration in early game
                temperature = 1.0 if move_count < temperature_threshold else 0.1

                # Get MCTS policy
                mcts.temperature = temperature
                policy, move = mcts.search(game)

                # Store state and policy
                state = encode_state(game)
                game_data.append((state.cpu(), policy, game.current_player()))

                # Make move
                game = game.make_move(move[0], move[1])
                move_count += 1

            # Get final outcome
            result = game.get_result()
            if result is None:
                result = 0

            # Convert to training data with proper value assignment
            for state, policy, player in game_data:
                value = result * player
                all_game_data.append((state, policy, value))

        # Return results via queue
        return_queue.put((gpu_id, all_game_data))

    except Exception as e:
        # Return error via queue
        import traceback
        return_queue.put((gpu_id, []))
        print(f"GPU {gpu_id} worker error: {e}")
        traceback.print_exc()


class DistributedAlphaZeroTrainer:
    def __init__(
        self,
        model=None,
        device='cuda:0',
        num_gpus=None,
        num_simulations=800,
        c_puct=1.0,
        num_selfplay_games=100,
        num_epochs=10,
        batch_size=256,
        learning_rate=0.001,
        replay_buffer_size=10000,
        checkpoint_dir='checkpoints',
    ):
        """
        Distributed AlphaZero trainer for Ultimate Tic Tac Toe.

        Uses multiple GPUs for parallel self-play game generation.

        Args:
            model: Neural network (if None, creates new UTTTNet)
            device: Primary device for training
            num_gpus: Number of GPUs to use (if None, auto-detect)
            num_simulations: MCTS simulations per move
            c_puct: MCTS exploration constant
            num_selfplay_games: Number of self-play games per iteration
            num_epochs: Training epochs per iteration
            batch_size: Training batch size
            learning_rate: Optimizer learning rate
            replay_buffer_size: Maximum size of experience replay buffer
            checkpoint_dir: Directory to save checkpoints
        """
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.num_selfplay_games = num_selfplay_games
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir

        # Auto-detect number of GPUs
        if num_gpus is None:
            self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        else:
            self.num_gpus = num_gpus

        print(f"Using {self.num_gpus} GPU(s) for distributed self-play")

        # Initialize model
        if model is None:
            self.model = UTTTNet().to(device)
        else:
            self.model = model.to(device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # Replay buffer: stores (state, policy, value) tuples
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

    def generate_self_play_data(self, temperature_threshold=15):
        """Generate self-play games in parallel across GPUs."""
        print(f"Generating {self.num_selfplay_games} self-play games across {self.num_gpus} GPU(s)...")

        # Get current model state
        model_state = self.model.state_dict()

        # Distribute games across GPUs
        games_per_gpu = [self.num_selfplay_games // self.num_gpus] * self.num_gpus
        # Distribute remainder
        for i in range(self.num_selfplay_games % self.num_gpus):
            games_per_gpu[i] += 1

        print(f"Games per GPU: {games_per_gpu}")

        # Create multiprocessing context
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass  # Already set
        return_queue = mp.Queue()

        # Launch workers
        processes = []
        for gpu_id in range(self.num_gpus):
            p = mp.Process(
                target=self_play_worker,
                args=(
                    gpu_id,
                    model_state,
                    games_per_gpu[gpu_id],
                    self.num_simulations,
                    self.c_puct,
                    temperature_threshold,
                    return_queue
                )
            )
            p.start()
            processes.append(p)

        # Collect results with progress bar
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

        # Wait for all processes
        for p in processes:
            p.join()

        # Add to replay buffer
        self.replay_buffer.extend(all_data)

        print(f"Replay buffer size: {len(self.replay_buffer)}")

    def train_network(self):
        """Train the network on data from replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            print("Not enough data in replay buffer to train")
            return

        print(f"Training network for {self.num_epochs} epochs...")
        self.model.train()

        for epoch in range(self.num_epochs):
            # Sample random batches
            batch_losses = []
            policy_losses = []
            value_losses = []

            # Shuffle and create batches
            data = list(self.replay_buffer)
            random.shuffle(data)

            for i in range(0, len(data) - self.batch_size + 1, self.batch_size):
                batch = data[i:i + self.batch_size]

                # Prepare batch
                states = torch.stack([s for s, p, v in batch]).to(self.device)
                target_policies = torch.tensor([p for s, p, v in batch], dtype=torch.float32).to(self.device)
                target_values = torch.tensor([[v] for s, p, v in batch], dtype=torch.float32).to(self.device)

                # Forward pass
                log_policies, values = self.model(states)

                # Compute losses
                # Policy loss: cross-entropy
                policy_loss = -torch.sum(target_policies * log_policies) / self.batch_size

                # Value loss: MSE
                value_loss = torch.mean((values - target_values) ** 2)

                # Total loss
                loss = policy_loss + value_loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_losses.append(loss.item())
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

            avg_loss = np.mean(batch_losses)
            avg_policy_loss = np.mean(policy_losses)
            avg_value_loss = np.mean(value_losses)

            print(f"Epoch {epoch + 1}/{self.num_epochs} - "
                  f"Loss: {avg_loss:.4f} (Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f})")

    def train(self, num_iterations=100):
        """
        Main training loop.

        Args:
            num_iterations: Number of training iterations
        """
        for iteration in range(num_iterations):
            print(f"\n{'=' * 60}")
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"{'=' * 60}")

            # Self-play (distributed)
            self.generate_self_play_data()

            # Train (single GPU)
            self.train_network()

            # Save checkpoint
            if (iteration + 1) % 5 == 0:
                self.save_checkpoint(iteration + 1)

        print("\nTraining complete!")

    def save_checkpoint(self, iteration):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f'model_iter_{iteration}.pt')
        buffer_path = os.path.join(self.checkpoint_dir, f'buffer_iter_{iteration}.pkl')

        torch.save({
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)

        # Save replay buffer
        with open(buffer_path, 'wb') as f:
            pickle.dump(list(self.replay_buffer), f)

        print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path, load_buffer=True):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        iteration = checkpoint['iteration']
        print(f"Loaded checkpoint from iteration {iteration}")

        # Load replay buffer if exists
        if load_buffer:
            buffer_path = checkpoint_path.replace('model_', 'buffer_').replace('.pt', '.pkl')
            if os.path.exists(buffer_path):
                with open(buffer_path, 'rb') as f:
                    buffer_data = pickle.load(f)
                    self.replay_buffer.extend(buffer_data)
                print(f"Loaded replay buffer with {len(self.replay_buffer)} samples")

        return iteration
