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

        legal_policy = {}
        total_prob = 0.0
        for global_idx, local_idx in legal_moves:
            move_idx = global_idx * 9 + local_idx
            prob = policy_probs[move_idx]
            legal_policy[(global_idx, local_idx)] = prob
            total_prob += prob

        if total_prob > 0:
            for move in legal_policy:
                legal_policy[move] /= total_prob
        else:
            uniform_prob = 1.0 / len(legal_moves)
            for move in legal_policy:
                legal_policy[move] = uniform_prob

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
            value = -value
            node = node.parent


class BatchedMCTS:
    def __init__(self, model, device='cuda', num_simulations=800, c_puct=1.0, temperature=1.0, batch_size=32):
        """
        Batched MCTS with neural network guidance.

        Args:
            model: Neural network (UTTTNet)
            device: 'cuda' or 'cpu'
            num_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant
            temperature: Temperature for policy sampling
            batch_size: Number of positions to evaluate in parallel
        """
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.batch_size = batch_size

    @torch.no_grad()
    def evaluate_batch(self, games):
        """Evaluate multiple positions in a single batch."""
        self.model.eval()
        states = torch.stack([encode_state(game) for game in games]).to(self.device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            log_policies, values = self.model(states)

        # Convert to float32 before numpy (numpy doesn't support bfloat16)
        policies = torch.exp(log_policies).float().cpu().numpy()
        values = values.float().cpu().numpy().flatten()
        return policies, values

    def search(self, game):
        """
        Run batched MCTS from the given position.

        Returns:
            policy: Visit count distribution over moves
            best_move: Tuple (global, local) of the best move to play
        """
        root = MCTSNode(game)

        # Run simulations in batches
        for batch_start in range(0, self.num_simulations, self.batch_size):
            batch_size_actual = min(self.batch_size, self.num_simulations - batch_start)

            leaf_nodes = []
            leaf_games = []
            search_paths = []
            terminal_nodes = []

            # Selection phase: collect batch_size leaf nodes
            for _ in range(batch_size_actual):
                node = root
                path = [node]

                # Traverse tree until leaf
                while node.is_expanded and not node.game.is_terminal():
                    move, node = node.select_child(self.c_puct)
                    path.append(node)

                if node.game.is_terminal():
                    # Handle terminal nodes separately
                    result = node.game.get_result()
                    if result is None:
                        value = 0
                    else:
                        value = result * node.game.current_player()
                    terminal_nodes.append((node, -value))
                else:
                    # Collect for batch evaluation
                    leaf_nodes.append(node)
                    leaf_games.append(node.game)
                    search_paths.append(path)

            # Batch evaluation of non-terminal leaves
            if leaf_nodes:
                policies, values = self.evaluate_batch(leaf_games)

                # Expand and backup
                for i, (node, path) in enumerate(zip(leaf_nodes, search_paths)):
                    node.expand(policies[i])
                    node.backup(-values[i])

            # Backup terminal nodes
            for node, value in terminal_nodes:
                node.backup(value)

        # Return visit count policy
        visit_counts = np.zeros(81)
        for move, child in root.children.items():
            move_idx = move[0] * 9 + move[1]
            visit_counts[move_idx] = child.visit_count

        # Temperature-based policy
        if self.temperature == 0:
            policy = np.zeros(81)
            policy[np.argmax(visit_counts)] = 1.0
        else:
            visits_temp = visit_counts ** (1.0 / self.temperature)
            policy = visits_temp / (visits_temp.sum() + 1e-8)

        # Select best move
        legal_moves = game.legal_moves()
        best_idx = np.argmax([visit_counts[m[0] * 9 + m[1]] for m in legal_moves])
        best_move = legal_moves[best_idx]

        return policy, best_move
