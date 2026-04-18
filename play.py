#!/usr/bin/env python3
"""
Play against the trained AlphaZero model.
"""
import argparse
import torch
from alphazero import UTTTNet, MCTS
from uttt_engine import UTTTGame

def print_board(game):
    """Print the game board in a readable format."""
    state = game.get_state()
    x_pieces = state[0]  # 9x9
    o_pieces = state[1]  # 9x9

    print("\n" + "=" * 50)
    for global_row in range(3):
        for local_row in range(3):
            line = ""
            for global_col in range(3):
                board_idx = global_row * 3 + global_col
                if global_col > 0:
                    line += " | "
                for local_col in range(3):
                    cell_idx = local_row * 3 + local_col
                    if x_pieces[board_idx][cell_idx] == 1:
                        line += "X "
                    elif o_pieces[board_idx][cell_idx] == 1:
                        line += "O "
                    else:
                        line += ". "
            print(line)
        if global_row < 2:
            print("-" * 50)
    print("=" * 50)

    # Show legal moves
    legal_moves = game.legal_moves()
    print(f"\nLegal moves ({len(legal_moves)}): ", end="")
    for i, (g, l) in enumerate(legal_moves[:10]):  # Show first 10
        print(f"({g},{l})", end=" ")
    if len(legal_moves) > 10:
        print(f"... and {len(legal_moves) - 10} more")
    else:
        print()
    print()

def human_move(game):
    """Get move from human player."""
    while True:
        try:
            move_str = input("Your move (format: global,local e.g., '0,4'): ")
            parts = move_str.strip().split(',')
            if len(parts) != 2:
                print("Invalid format. Use: global,local")
                continue

            global_idx = int(parts[0])
            local_idx = int(parts[1])

            if (global_idx, local_idx) not in game.legal_moves():
                print("Illegal move. Try again.")
                continue

            return global_idx, local_idx
        except (ValueError, KeyboardInterrupt):
            print("\nInvalid input. Try again.")

def ai_move(game, mcts):
    """Get move from AI."""
    print("AI is thinking...")
    _, move = mcts.search(game)
    print(f"AI plays: ({move[0]},{move[1]})")
    return move

def play_game(model_path, device='cuda', num_simulations=800, human_plays_first=True):
    """Play a game against the AI."""
    # Load model
    model = UTTTNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {model_path}")

    # Create MCTS
    mcts = MCTS(model, device=device, num_simulations=num_simulations, temperature=0.1)

    # Start game
    game = UTTTGame()
    human_is_x = human_plays_first

    print("\n" + "=" * 50)
    print("Ultimate Tic Tac Toe - Human vs AI")
    print("=" * 50)
    print(f"You are playing as: {'X (first)' if human_is_x else 'O (second)'}")
    print(f"AI simulations per move: {num_simulations}")

    while not game.is_terminal():
        print_board(game)

        current_player = game.current_player()
        is_human_turn = (current_player == 1 and human_is_x) or (current_player == -1 and not human_is_x)

        if is_human_turn:
            move = human_move(game)
        else:
            move = ai_move(game, mcts)

        game = game.make_move(move[0], move[1])

    # Game over
    print_board(game)
    result = game.get_result()

    print("\n" + "=" * 50)
    if result == 0:
        print("Game ended in a DRAW!")
    elif (result == 1 and human_is_x) or (result == -1 and not human_is_x):
        print("You WIN! 🎉")
    else:
        print("AI WINS!")
    print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description='Play against AlphaZero')
    parser.add_argument('model', type=str, help='Path to model checkpoint')
    parser.add_argument('--simulations', type=int, default=800,
                        help='MCTS simulations per AI move')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--ai-first', action='store_true',
                        help='Let AI play first')

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    play_game(
        args.model,
        device=args.device,
        num_simulations=args.simulations,
        human_plays_first=not args.ai_first
    )

if __name__ == '__main__':
    main()
