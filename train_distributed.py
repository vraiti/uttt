#!/usr/bin/env python3
"""
Train AlphaZero on Ultimate Tic Tac Toe using distributed multi-GPU self-play.
"""
import argparse
import torch
from alphazero import DistributedAlphaZeroTrainer, UTTTNet

def main():
    parser = argparse.ArgumentParser(description='Distributed AlphaZero Training for UTTT')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of training iterations')
    parser.add_argument('--games-per-iter', type=int, default=100,
                        help='Number of self-play games per iteration')
    parser.add_argument('--simulations', type=int, default=800,
                        help='Number of MCTS simulations per move')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Training epochs per iteration')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--buffer-size', type=int, default=50000,
                        help='Replay buffer size')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Primary device for training (e.g., cuda:0)')
    parser.add_argument('--num-gpus', type=int, default=None,
                        help='Number of GPUs to use (default: auto-detect all)')
    parser.add_argument('--num-channels', type=int, default=256,
                        help='Number of channels in network')
    parser.add_argument('--num-res-blocks', type=int, default=10,
                        help='Number of residual blocks')

    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available, distributed training requires GPUs")
        return

    num_gpus = args.num_gpus if args.num_gpus else torch.cuda.device_count()

    print("=" * 60)
    print("Distributed AlphaZero Training for Ultimate Tic Tac Toe")
    print("=" * 60)
    print(f"Primary device: {args.device}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Iterations: {args.iterations}")
    print(f"Games per iteration: {args.games_per_iter}")
    print(f"MCTS simulations: {args.simulations}")
    print(f"Network: {args.num_channels} channels, {args.num_res_blocks} residual blocks")
    print("=" * 60)

    # Create model
    model = UTTTNet(num_channels=args.num_channels, num_res_blocks=args.num_res_blocks)

    # Create distributed trainer
    trainer = DistributedAlphaZeroTrainer(
        model=model,
        device=args.device,
        num_gpus=num_gpus,
        num_simulations=args.simulations,
        num_selfplay_games=args.games_per_iter,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        replay_buffer_size=args.buffer_size,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(num_iterations=args.iterations)

if __name__ == '__main__':
    main()
