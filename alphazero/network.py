import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class UTTTNet(nn.Module):
    """
    Neural network for Ultimate Tic Tac Toe.

    Input: 9x9 grid with multiple channels:
        - Channel 0: Current player's pieces (1 for X, 0 otherwise)
        - Channel 1: Opponent's pieces (1 for O, 0 otherwise)
        - Channel 2: Valid moves (1 if legal, 0 otherwise)
        - Channel 3: Next board constraint (1 if must play in this board)
        - Channel 4: Won boards by current player
        - Channel 5: Won boards by opponent
        - Channel 6: Drawn boards

    Output:
        - Policy: 81-dimensional vector (9x9 boards)
        - Value: Scalar in [-1, 1] representing win probability
    """

    def __init__(self, num_channels=256, num_res_blocks=10):
        super().__init__()

        # Initial convolution
        self.input_conv = nn.Conv2d(7, num_channels, 3, padding=1)
        self.input_bn = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 9 * 9, 81)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 32, 1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 9 * 9, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # x shape: (batch, 7, 9, 9)

        # Initial conv
        x = F.relu(self.input_bn(self.input_conv(x)))

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

def encode_state(game):
    """
    Encode game state as a 7x9x9 tensor.

    Args:
        game: UTTTGame instance

    Returns:
        torch.Tensor of shape (7, 9, 9)
    """
    state = torch.zeros(7, 9, 9)

    # Get basic state
    game_state = game.get_state()  # 3x9x9: [X pieces, O pieces, legal moves]
    current_player = game.current_player()

    if current_player == 1:  # X to move
        state[0] = torch.tensor(game_state[0], dtype=torch.float32)  # Current player (X)
        state[1] = torch.tensor(game_state[1], dtype=torch.float32)  # Opponent (O)
    else:  # O to move
        state[0] = torch.tensor(game_state[1], dtype=torch.float32)  # Current player (O)
        state[1] = torch.tensor(game_state[0], dtype=torch.float32)  # Opponent (X)

    state[2] = torch.tensor(game_state[2], dtype=torch.float32)  # Legal moves

    # Add board constraints and won boards
    # For simplicity, we derive this from legal moves
    legal_moves = game.legal_moves()
    if legal_moves:
        # Check if all legal moves are in one board
        boards = set(m[0] for m in legal_moves)
        if len(boards) == 1:
            board = list(boards)[0]
            state[3, board // 3 * 3:(board // 3 + 1) * 3,
                     board % 3 * 3:(board % 3 + 1) * 3] = 1

    # Won/drawn boards (simplified - would need to query game state)
    # Channels 4, 5, 6 left as zeros for now (can enhance later)

    return state
