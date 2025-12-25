import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.pgn
from torch.utils.data import Dataset, DataLoader
import requests
import gzip
import io

class ChessPositionEncoder:
    """Encode chess positions into neural network input format"""
    
    def __init__(self):
        # Map each piece type to a channel index
        self.piece_to_index = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
        }
    
    def encode_position(self, board):
        """Convert chess board to 8x8x12 tensor"""
        position = np.zeros((8, 8, 12), dtype=np.float32)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)
                piece_idx = self.piece_to_index[piece.symbol()]
                position[row, col, piece_idx] = 1.0
        
        return position
    
    def encode_move(self, move):
        """Convert chess move to integer index"""
        return move.from_square * 64 + move.to_square
    
    def decode_move(self, move_index):
        """Decode move index back to chess.Move"""
        from_square = move_index // 64
        to_square = move_index % 64
        return chess.Move(from_square, to_square)

class ChessDataset(Dataset):
    """Dataset for chess positions and moves"""
    
    def __init__(self, max_games=1000):
        self.encoder = ChessPositionEncoder()
        self.positions = []
        self.moves = []
        self.results = []
        self.load_lichess_data(max_games)
    
    def load_lichess_data(self, max_games):
        """Load chess games from Lichess database"""
        print("Downloading Lichess data...")
        
        # Download sample PGN data from Lichess
        url = "https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.bz2"
        
        # For demo, create sample games instead of downloading large file
        sample_games = self.create_sample_games(max_games)
        
        for game_pgn in sample_games:
            try:
                game = chess.pgn.read_game(io.StringIO(game_pgn))
                if game is None:
                    continue
                
                # Get game result
                result = game.headers.get('Result', '*')
                if result == '1-0':
                    game_result = 1.0
                elif result == '0-1':
                    game_result = -1.0
                else:
                    game_result = 0.0
                
                # Extract positions and moves
                board = game.board()
                for move in game.mainline_moves():
                    if board.is_legal(move):
                        position = self.encoder.encode_position(board)
                        turn_channel = np.ones((8, 8, 1), dtype=np.float32) * (1.0 if board.turn else -1.0)
                        position_with_turn = np.concatenate([position, turn_channel], axis=2)
                        
                        move_encoded = self.encoder.encode_move(move)
                        
                        self.positions.append(position_with_turn)
                        self.moves.append(move_encoded)
                        self.results.append(game_result)
                        
                        board.push(move)
            except:
                continue
        
        print(f"Loaded {len(self.positions)} positions from sample games")
    
    def create_sample_games(self, num_games):
        """Create sample PGN games for training"""
        sample_games = []
        
        # Common opening games
        games = [
            '''[Event "Sample Game 1"]
[Site "Online"]
[Date "2024.01.01"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. Nbd2 Bb7 12. Bc2 Re8 13. Nf1 Bf8 14. Ng3 g6 15. a4 c5 16. d5 c4 17. Be3 Nc5 18. Qd2 h6 19. Nh2 Kh7 20. f4 exf4 21. Bxf4 Nh5 22. Nxh5 gxh5 23. Qf2 Be7 24. Qf3 Bg5 25. Bxg5 hxg5 26. Qxf7+ Kh8 27. Qf3 Qf6 28. Qxf6+ Kg8 29. Re3 1-0''',
            
            '''[Event "Sample Game 2"]
[Site "Online"]
[Date "2024.01.02"]
[White "Player3"]
[Black "Player4"]
[Result "0-1"]

1. d4 d5 2. c4 c6 3. Nf3 Nf6 4. Nc3 dxc4 5. a4 Bf5 6. e3 e6 7. Bxc4 Bb4 8. O-O Nbd7 9. Qe2 Bg6 10. e4 O-O 11. Bd2 Bh5 12. Rfd1 Bg6 13. Rac1 Bh5 14. h3 Bxf3 15. Qxf3 e5 16. d5 Bxc3 17. bxc3 cxd5 18. exd5 Nc5 19. Bb4 Ncxe4 20. Bxf8 Qxf8 21. c4 Rc8 22. Rd3 Rxc4 23. Rxc4 Nxc4 24. Rd1 Qc5 25. Qf5 Ncd6 26. Qf3 Qc1+ 27. Rxc1 Nxc1 0-1''',
            
            '''[Event "Sample Game 3"]
[Site "Online"]
[Date "2024.01.03"]
[White "Player5"]
[Black "Player6"]
[Result "1/2-1/2"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e6 7. f3 b5 8. Qd2 Bb7 9. O-O-O Nbd7 10. h4 b4 11. Nd5 Bxd5 12. exd5 Rc8 13. Kb1 e5 14. Nb3 Be7 15. Bd3 O-O 16. Rhe1 Rc3 17. Re2 Qa5 18. Qxa5 Rxa5 19. Bxc3 bxc3 20. Re3 Rb5 21. Rxc3 Rxb3+ 22. axb3 Nxd5 23. Rc7 Nf4 24. Bf1 Bd8 25. Rc8 Kf8 26. b4 Ke7 27. b5 axb5 28. Bxb5 Nf6 29. Rc7+ Kf8 30. Rc8 Ke7 1/2-1/2'''
        ]
        
        # Repeat games to reach desired number
        while len(sample_games) < num_games:
            sample_games.extend(games)
        
        return sample_games[:num_games]
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        position = torch.FloatTensor(self.positions[idx]).permute(2, 0, 1)
        move = torch.LongTensor([self.moves[idx]])
        result = torch.FloatTensor([self.results[idx]])
        return position, move, result

class ResidualBlock(nn.Module):
    """Residual block for deep learning"""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class ChessNet(nn.Module):
    """Neural network for chess move prediction"""
    
    def __init__(self, input_channels=13, hidden_dim=128, num_residual_blocks=4):
        super(ChessNet, self).__init__()
        
        # Initial convolution
        self.conv_input = nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(hidden_dim)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(hidden_dim, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096)
        
        # Value head
        self.value_conv = nn.Conv2d(hidden_dim, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # Initial convolution
        x = self.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        policy = self.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = torch.softmax(policy, dim=1)
        
        # Value head
        value = self.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = self.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

class ChessTrainer:
    """Training pipeline for chess model"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.policy_criterion = nn.CrossEntropyLoss()
        self.value_criterion = nn.MSELoss()
        self.encoder = ChessPositionEncoder()
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (positions, moves, results) in enumerate(dataloader):
            positions = positions.to(self.device)
            moves = moves.to(self.device).squeeze()
            results = results.to(self.device).squeeze()
            
            self.optimizer.zero_grad()
            
            policy_pred, value_pred = self.model(positions)
            
            policy_loss = self.policy_criterion(policy_pred, moves)
            value_loss = self.value_criterion(value_pred.squeeze(), results)
            
            total_batch_loss = policy_loss + 0.5 * value_loss
            total_batch_loss.backward()
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}: Loss: {total_batch_loss.item():.4f}')
        
        return total_loss / len(dataloader)
    
    def predict_move(self, board):
        """Predict best move for given position"""
        self.model.eval()
        
        with torch.no_grad():
            position = self.encoder.encode_position(board)
            turn_channel = np.ones((8, 8, 1), dtype=np.float32) * (1.0 if board.turn else -1.0)
            position_with_turn = np.concatenate([position, turn_channel], axis=2)
            
            position_tensor = torch.FloatTensor(position_with_turn).permute(2, 0, 1).unsqueeze(0)
            position_tensor = position_tensor.to(self.device)
            
            policy, value = self.model(position_tensor)
            
            # Get legal moves and their probabilities
            legal_moves = list(board.legal_moves)
            move_probs = []
            
            for move in legal_moves:
                move_idx = self.encoder.encode_move(move)
                prob = policy[0, move_idx].item()
                move_probs.append((move, prob))
            
            # Sort by probability
            move_probs.sort(key=lambda x: x[1], reverse=True)
            
            return move_probs, value.item()

def train_chess_model():
    """Train the chess model"""
    print("Training Chess ML Model...")
    
    # Initialize model
    model = ChessNet()
    trainer = ChessTrainer(model)
    
    # Load dataset
    dataset = ChessDataset(max_games=100)  # Small dataset for demo
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        avg_loss = trainer.train_epoch(dataloader)
        print(f'Average Loss: {avg_loss:.4f}')
    
    # Save model
    torch.save(model.state_dict(), 'chess_ml_model.pth')
    print("Model saved as chess_ml_model.pth")
    
    return model, trainer

if __name__ == "__main__":
    # Train the model
    model, trainer = train_chess_model()
    
    # Test prediction
    board = chess.Board()
    move_probs, value = trainer.predict_move(board)
    print(f"Best move: {move_probs[0][0]} (prob: {move_probs[0][1]:.3f})")
    print(f"Position value: {value:.3f}")