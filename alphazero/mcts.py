import math
import numpy as np
import torch
from .network import encode_state

class MCTSNode:
    def __init__(self, game, parent=None, prior=0.0):
        self.game = game
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct=1.0):
        """Select child with highest UCB score."""
        best_score = -float('inf')
        best_move = None
        best_child = None

        for move, child in self.children.items():
            # UCB formula: Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
            q_value = child.value()
            u_value = c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def expand(self, policy_probs):
        """Expand node by adding children for all legal moves."""
        legal_moves = self.game.legal_moves()

        # Normalize policy to only legal moves
        legal_policy = {}
        total_prob = 0.0
        for global_idx, local_idx in legal_moves:
            move_idx = global_idx * 9 + local_idx
            prob = policy_probs[move_idx]
            legal_policy[(global_idx, local_idx)] = prob
            total_prob += prob

        # Normalize
        if total_prob > 0:
            for move in legal_policy:
                legal_policy[move] /= total_prob
        else:
            # Uniform if all probs are zero
            uniform_prob = 1.0 / len(legal_moves)
            for move in legal_policy:
                legal_policy[move] = uniform_prob

        # Create child nodes
        for move, prob in legal_policy.items():
            new_game = self.game.make_move(move[0], move[1])
            self.children[move] = MCTSNode(new_game, parent=self, prior=prob)

        self.is_expanded = True

    def backup(self, value):
        """Backpropagate value up the tree."""
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip value for opponent
            node = node.parent


class MCTS:
    def __init__(self, model, device='cuda', num_simulations=800, c_puct=1.0, temperature=1.0):
        """
        Monte Carlo Tree Search with neural network guidance.

        Args:
            model: Neural network (UTTTNet)
            device: 'cuda' or 'cpu'
            num_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant
            temperature: Temperature for policy sampling (higher = more exploration)
        """
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature

    @torch.no_grad()
    def evaluate(self, game):
        """Use neural network to get policy and value for a position."""
        self.model.eval()
        state = encode_state(game).unsqueeze(0).to(self.device)
        log_policy, value = self.model(state)
        policy = torch.exp(log_policy).cpu().numpy()[0]
        value = value.cpu().item()
        return policy, value

    def search(self, game):
        """
        Run MCTS from the given position.

        Returns:
            policy: Visit count distribution over moves (to be used for training)
            best_move: Tuple (global, local) of the best move to play
        """
        root = MCTSNode(game)

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # Selection: traverse tree until we reach a leaf
            while node.is_expanded and not node.game.is_terminal():
                move, node = node.select_child(self.c_puct)
                search_path.append(node)

            # Evaluate leaf node
            if node.game.is_terminal():
                # Terminal node - use actual result
                result = node.game.get_result()
                if result is None:
                    value = 0
                else:
                    # Result is from X's perspective, convert to current player's
                    value = result * node.game.current_player()
            else:
                # Expand and evaluate with neural network
                policy, value = self.evaluate(node.game)
                node.expand(policy)

            # Backpropagation
            node.backup(-value)  # Negate because backup flips perspective

        # Return visit count policy
        visit_counts = np.zeros(81)
        for move, child in root.children.items():
            move_idx = move[0] * 9 + move[1]
            visit_counts[move_idx] = child.visit_count

        # Temperature-based policy
        if self.temperature == 0:
            # Greedy: pick most visited
            policy = np.zeros(81)
            policy[np.argmax(visit_counts)] = 1.0
        else:
            # Boltzmann distribution
            visits_temp = visit_counts ** (1.0 / self.temperature)
            policy = visits_temp / (visits_temp.sum() + 1e-8)

        # Select best move
        legal_moves = game.legal_moves()
        best_idx = np.argmax([visit_counts[m[0] * 9 + m[1]] for m in legal_moves])
        best_move = legal_moves[best_idx]

        return policy, best_move

    def get_action_probs(self, game, temperature=None):
        """
        Get action probabilities for the current position.

        Args:
            game: Current game state
            temperature: Override temperature for this search

        Returns:
            policy: Probability distribution over all 81 positions
            move: Best move (global, local)
        """
        if temperature is not None:
            old_temp = self.temperature
            self.temperature = temperature

        policy, move = self.search(game)

        if temperature is not None:
            self.temperature = old_temp

        return policy, move
