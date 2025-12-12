"""
Advanced AI Chess Engine for Move Recommendation
"""

import numpy as np
import pickle
import os
from typing import List, Tuple, Dict, Optional

class ChessAIEngine:
    """Advanced neural network for chess move evaluation and recommendation"""
    
    def __init__(self, model_path="models/chess_ai_model.pkl"):
        self.model_path = model_path
        self.model = self._create_model()
        self.position_cache = {}
        self.learning_data = []
        
        # Piece values for evaluation
        self.piece_values = {
            'pawn': 100, 'knight': 320, 'bishop': 330,
            'rook': 500, 'queen': 900, 'king': 20000
        }
        
        # Position evaluation weights
        self.position_weights = {
            'center_control': 0.3,
            'piece_activity': 0.25,
            'king_safety': 0.2,
            'pawn_structure': 0.15,
            'material': 0.1
        }
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.load_model()
    
    def _create_model(self):
        """Create neural network model"""
        return {
            'weights1': np.random.randn(773, 512) * 0.1,
            'bias1': np.zeros((1, 512)),
            'weights2': np.random.randn(512, 256) * 0.1,
            'bias2': np.zeros((1, 256)),
            'weights3': np.random.randn(256, 128) * 0.1,
            'bias3': np.zeros((1, 128)),
            'weights4': np.random.randn(128, 1) * 0.1,
            'bias4': np.zeros((1, 1))
        }
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _forward(self, x):
        """Forward pass through neural network"""
        z1 = np.dot(x, self.model['weights1']) + self.model['bias1']
        a1 = self._relu(z1)
        
        z2 = np.dot(a1, self.model['weights2']) + self.model['bias2']
        a2 = self._relu(z2)
        
        z3 = np.dot(a2, self.model['weights3']) + self.model['bias3']
        a3 = self._relu(z3)
        
        z4 = np.dot(a3, self.model['weights4']) + self.model['bias4']
        return np.tanh(z4)
    
    def position_to_features(self, position):
        """Convert chess position to feature vector"""
        features = []
        
        # Board representation (8x8x12 = 768 features)
        piece_map = {
            ('pawn', 'white'): 0, ('pawn', 'black'): 1,
            ('knight', 'white'): 2, ('knight', 'black'): 3,
            ('bishop', 'white'): 4, ('bishop', 'black'): 5,
            ('rook', 'white'): 6, ('rook', 'black'): 7,
            ('queen', 'white'): 8, ('queen', 'black'): 9,
            ('king', 'white'): 10, ('king', 'black'): 11
        }
        
        board_features = np.zeros(768)
        for square, piece_info in position.items():
            if piece_info:
                row, col = self._square_to_coords(square)
                piece_type = piece_info['type']
                color = piece_info['color']
                
                if (piece_type, color) in piece_map:
                    piece_idx = piece_map[(piece_type, color)]
                    square_idx = row * 8 + col
                    board_features[square_idx * 12 + piece_idx] = 1
        
        features.extend(board_features)
        
        # Additional features (5 features)
        white_material = self._calculate_material(position, 'white')
        black_material = self._calculate_material(position, 'black')
        
        features.extend([
            white_material / 3900,  # Normalized material
            black_material / 3900,
            (white_material - black_material) / 3900,  # Material difference
            self._count_pieces(position, 'white') / 16,  # Piece count
            self._count_pieces(position, 'black') / 16
        ])
        
        return np.array(features).reshape(1, -1)
    
    def evaluate_position(self, position, current_player='white'):
        """Evaluate chess position using AI"""
        position_key = str(sorted(position.items()))
        
        if position_key in self.position_cache:
            return self.position_cache[position_key]
        
        features = self.position_to_features(position)
        ai_score = float(self._forward(features)[0][0])
        
        # Adjust for current player
        if current_player == 'black':
            ai_score = -ai_score
        
        self.position_cache[position_key] = ai_score
        return ai_score
    
    def get_best_moves(self, position, current_player, num_moves=5):
        """Get best moves with detailed analysis"""
        legal_moves = self._generate_legal_moves(position, current_player)
        move_evaluations = []
        
        for move in legal_moves:
            # Simulate move
            new_position = self._make_move(position.copy(), move)
            
            # Evaluate resulting position
            score = self.evaluate_position(new_position, current_player)
            
            # Get detailed analysis
            analysis = self._analyze_move(position, move, new_position, score)
            
            move_evaluations.append({
                'move': move,
                'score': score,
                'analysis': analysis,
                'position_after': new_position
            })
        
        # Sort by score (best first)
        move_evaluations.sort(key=lambda x: x['score'], reverse=True)
        
        return move_evaluations[:num_moves]
    
    def _analyze_move(self, position_before, move, position_after, score):
        """Detailed move analysis"""
        analysis = {
            'score': score,
            'confidence': self._calculate_confidence(score),
            'tactical_elements': self._identify_tactics(position_before, move, position_after),
            'positional_factors': self._evaluate_positional_factors(position_after),
            'material_change': self._calculate_material_change(position_before, position_after),
            'strategic_themes': self._identify_strategic_themes(position_before, position_after, move)
        }
        
        return analysis
    
    def _calculate_confidence(self, score):
        """Calculate confidence level based on score"""
        abs_score = abs(score)
        if abs_score > 0.7:
            return 'very_high'
        elif abs_score > 0.4:
            return 'high'
        elif abs_score > 0.2:
            return 'medium'
        elif abs_score > 0.1:
            return 'low'
        else:
            return 'very_low'
    
    def _identify_tactics(self, pos_before, move, pos_after):
        """Identify tactical elements in the move"""
        tactics = []
        
        # Check for captures
        if self._is_capture(pos_before, move):
            tactics.append('capture')
        
        # Check for checks
        if self._gives_check(pos_after, move['color']):
            tactics.append('check')
        
        # Check for forks
        if self._creates_fork(pos_after, move):
            tactics.append('fork')
        
        # Check for pins
        if self._creates_pin(pos_after, move):
            tactics.append('pin')
        
        return tactics
    
    def _evaluate_positional_factors(self, position):
        """Evaluate positional factors"""
        factors = {}
        
        # Center control
        factors['center_control'] = self._evaluate_center_control(position)
        
        # Piece activity
        factors['piece_activity'] = self._evaluate_piece_activity(position)
        
        # King safety
        factors['king_safety'] = self._evaluate_king_safety(position)
        
        # Pawn structure
        factors['pawn_structure'] = self._evaluate_pawn_structure(position)
        
        return factors
    
    def _identify_strategic_themes(self, pos_before, pos_after, move):
        """Identify strategic themes"""
        themes = []
        
        # Development
        if self._is_development_move(pos_before, move):
            themes.append('development')
        
        # Centralization
        if self._improves_centralization(pos_before, pos_after, move):
            themes.append('centralization')
        
        # Pawn breaks
        if self._is_pawn_break(move):
            themes.append('pawn_break')
        
        # Piece coordination
        if self._improves_coordination(pos_before, pos_after):
            themes.append('coordination')
        
        return themes
    
    def compare_moves(self, best_move, alternative_moves):
        """Compare best move with alternatives and explain differences"""
        comparisons = []
        
        best_score = best_move['score']
        best_analysis = best_move['analysis']
        
        for alt_move in alternative_moves:
            alt_score = alt_move['score']
            alt_analysis = alt_move['analysis']
            
            score_diff = best_score - alt_score
            
            comparison = {
                'move': alt_move['move'],
                'score_difference': score_diff,
                'reasons_worse': self._explain_why_worse(best_analysis, alt_analysis, score_diff),
                'alternative_merits': self._find_alternative_merits(alt_analysis)
            }
            
            comparisons.append(comparison)
        
        return comparisons
    
    def _explain_why_worse(self, best_analysis, alt_analysis, score_diff):
        """Explain why alternative move is worse"""
        reasons = []
        
        if score_diff > 0.3:
            reasons.append("Significantly weaker position evaluation")
        elif score_diff > 0.1:
            reasons.append("Moderately weaker position")
        
        # Compare tactical elements
        best_tactics = set(best_analysis['tactical_elements'])
        alt_tactics = set(alt_analysis['tactical_elements'])
        
        if 'capture' in best_tactics and 'capture' not in alt_tactics:
            reasons.append("Misses material gain opportunity")
        
        if 'check' in best_tactics and 'check' not in alt_tactics:
            reasons.append("Misses forcing move (check)")
        
        # Compare positional factors
        best_pos = best_analysis['positional_factors']
        alt_pos = alt_analysis['positional_factors']
        
        for factor in ['center_control', 'piece_activity', 'king_safety']:
            if best_pos.get(factor, 0) > alt_pos.get(factor, 0) + 0.2:
                reasons.append(f"Weaker {factor.replace('_', ' ')}")
        
        return reasons
    
    def _find_alternative_merits(self, alt_analysis):
        """Find positive aspects of alternative move"""
        merits = []
        
        tactics = alt_analysis['tactical_elements']
        if 'fork' in tactics:
            merits.append("Creates tactical fork")
        if 'pin' in tactics:
            merits.append("Establishes pin")
        
        themes = alt_analysis['strategic_themes']
        if 'development' in themes:
            merits.append("Improves piece development")
        if 'centralization' in themes:
            merits.append("Centralizes pieces")
        
        return merits
    
    def learn_from_game_result(self, game_moves, result):
        """Learn from completed game"""
        training_data = []
        
        for i, move_data in enumerate(game_moves):
            position_before = move_data['position_before']
            position_after = move_data['position_after']
            
            # Discount factor for move importance
            discount = 0.95 ** (len(game_moves) - i - 1)
            target_score = result * discount
            
            features = self.position_to_features(position_after)
            training_data.append((features, target_score))
        
        self.learning_data.extend(training_data)
        
        # Train if enough data accumulated
        if len(self.learning_data) >= 50:
            self._train_model()
    
    def _train_model(self, epochs=30):
        """Train the neural network"""
        if not self.learning_data:
            return
        
        print(f"Training AI model with {len(self.learning_data)} samples...")
        
        # Prepare data
        X = np.vstack([data[0] for data in self.learning_data])
        y = np.array([[data[1]] for data in self.learning_data])
        
        # Simple gradient descent training
        learning_rate = 0.001
        
        for epoch in range(epochs):
            # Forward pass
            predictions = self._forward(X)
            loss = np.mean((predictions - y) ** 2)
            
            # Backward pass (simplified)
            self._backward(X, y, predictions, learning_rate)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        # Clear training data and save model
        self.learning_data = []
        self.save_model()
        print("Model training completed!")
    
    def _backward(self, X, y, predictions, lr):
        """Simplified backpropagation"""
        m = X.shape[0]
        
        # Forward pass to get intermediate values
        z1 = np.dot(X, self.model['weights1']) + self.model['bias1']
        a1 = self._relu(z1)
        z2 = np.dot(a1, self.model['weights2']) + self.model['bias2']
        a2 = self._relu(z2)
        z3 = np.dot(a2, self.model['weights3']) + self.model['bias3']
        a3 = self._relu(z3)
        
        # Output layer gradient
        dz4 = (predictions - y) * (1 - predictions**2)  # tanh derivative
        dw4 = np.dot(a3.T, dz4) / m
        db4 = np.sum(dz4, axis=0, keepdims=True) / m
        
        # Hidden layer 3 gradient
        da3 = np.dot(dz4, self.model['weights4'].T)
        dz3 = da3 * (a3 > 0)  # ReLU derivative
        dw3 = np.dot(a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m
        
        # Update weights
        self.model['weights4'] -= lr * dw4
        self.model['bias4'] -= lr * db4
        self.model['weights3'] -= lr * dw3
        self.model['bias3'] -= lr * db3
    
    def save_model(self):
        """Save trained model"""
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load existing model"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"Model loaded from {self.model_path}")
            except Exception as e:
                print(f"Could not load model: {e}")
                print("Using fresh model")
    
    # Helper methods for chess logic
    def _square_to_coords(self, square):
        """Convert square notation to coordinates"""
        if isinstance(square, str) and len(square) == 2:
            file = ord(square[0]) - ord('a')
            rank = int(square[1]) - 1
            return (7 - rank, file)
        return square
    
    def _calculate_material(self, position, color):
        """Calculate material value for a color"""
        total = 0
        for piece_info in position.values():
            if piece_info and piece_info['color'] == color:
                total += self.piece_values.get(piece_info['type'], 0)
        return total
    
    def _count_pieces(self, position, color):
        """Count pieces for a color"""
        return sum(1 for piece_info in position.values() 
                  if piece_info and piece_info['color'] == color)
    
    def _calculate_material_change(self, pos_before, pos_after):
        """Calculate material change from move"""
        before_white = self._calculate_material(pos_before, 'white')
        before_black = self._calculate_material(pos_before, 'black')
        after_white = self._calculate_material(pos_after, 'white')
        after_black = self._calculate_material(pos_after, 'black')
        
        return {
            'white_change': after_white - before_white,
            'black_change': after_black - before_black
        }
    
    def _generate_legal_moves(self, position, color):
        """Generate legal moves (simplified)"""
        moves = []
        
        for square, piece_info in position.items():
            if piece_info and piece_info['color'] == color:
                piece_moves = self._get_piece_moves(square, piece_info, position)
                moves.extend(piece_moves)
        
        return moves[:20]  # Limit for performance
    
    def _get_piece_moves(self, square, piece_info, position):
        """Get moves for a specific piece"""
        moves = []
        piece_type = piece_info['type']
        color = piece_info['color']
        
        # Simplified move generation
        if piece_type == 'pawn':
            moves = self._get_pawn_moves(square, color, position)
        elif piece_type == 'knight':
            moves = self._get_knight_moves(square, color, position)
        elif piece_type == 'bishop':
            moves = self._get_bishop_moves(square, color, position)
        elif piece_type == 'rook':
            moves = self._get_rook_moves(square, color, position)
        elif piece_type == 'queen':
            moves = self._get_queen_moves(square, color, position)
        elif piece_type == 'king':
            moves = self._get_king_moves(square, color, position)
        
        return moves
    
    def _get_pawn_moves(self, square, color, position):
        """Get pawn moves"""
        moves = []
        row, col = self._square_to_coords(square)
        direction = -1 if color == 'white' else 1
        
        # Forward move
        new_row = row + direction
        if 0 <= new_row < 8:
            new_square = f"{chr(ord('a') + col)}{8 - new_row}"
            if new_square not in position or not position[new_square]:
                moves.append({
                    'from': square,
                    'to': new_square,
                    'piece': 'pawn',
                    'color': color
                })
        
        return moves
    
    def _get_knight_moves(self, square, color, position):
        """Get knight moves"""
        moves = []
        row, col = self._square_to_coords(square)
        
        knight_moves = [(2,1), (2,-1), (-2,1), (-2,-1), (1,2), (1,-2), (-1,2), (-1,-2)]
        
        for dr, dc in knight_moves:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                new_square = f"{chr(ord('a') + new_col)}{8 - new_row}"
                target = position.get(new_square)
                
                if not target or target['color'] != color:
                    moves.append({
                        'from': square,
                        'to': new_square,
                        'piece': 'knight',
                        'color': color
                    })
        
        return moves
    
    def _get_bishop_moves(self, square, color, position):
        """Get bishop moves"""
        return self._get_sliding_moves(square, color, position, 
                                     [(1,1), (1,-1), (-1,1), (-1,-1)], 'bishop')
    
    def _get_rook_moves(self, square, color, position):
        """Get rook moves"""
        return self._get_sliding_moves(square, color, position, 
                                     [(0,1), (0,-1), (1,0), (-1,0)], 'rook')
    
    def _get_queen_moves(self, square, color, position):
        """Get queen moves"""
        return self._get_sliding_moves(square, color, position, 
                                     [(0,1), (0,-1), (1,0), (-1,0), 
                                      (1,1), (1,-1), (-1,1), (-1,-1)], 'queen')
    
    def _get_king_moves(self, square, color, position):
        """Get king moves"""
        moves = []
        row, col = self._square_to_coords(square)
        
        directions = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                new_square = f"{chr(ord('a') + new_col)}{8 - new_row}"
                target = position.get(new_square)
                
                if not target or target['color'] != color:
                    moves.append({
                        'from': square,
                        'to': new_square,
                        'piece': 'king',
                        'color': color
                    })
        
        return moves
    
    def _get_sliding_moves(self, square, color, position, directions, piece_type):
        """Get moves for sliding pieces"""
        moves = []
        row, col = self._square_to_coords(square)
        
        for dr, dc in directions:
            for i in range(1, 8):
                new_row, new_col = row + dr * i, col + dc * i
                if not (0 <= new_row < 8 and 0 <= new_col < 8):
                    break
                
                new_square = f"{chr(ord('a') + new_col)}{8 - new_row}"
                target = position.get(new_square)
                
                if not target:
                    moves.append({
                        'from': square,
                        'to': new_square,
                        'piece': piece_type,
                        'color': color
                    })
                elif target['color'] != color:
                    moves.append({
                        'from': square,
                        'to': new_square,
                        'piece': piece_type,
                        'color': color
                    })
                    break
                else:
                    break
        
        return moves
    
    def _make_move(self, position, move):
        """Make a move on the position"""
        new_position = position.copy()
        
        # Remove piece from source
        if move['from'] in new_position:
            piece = new_position[move['from']]
            del new_position[move['from']]
            
            # Place piece at destination
            new_position[move['to']] = piece
        
        return new_position
    
    # Simplified evaluation methods
    def _is_capture(self, position, move):
        return move['to'] in position and position[move['to']] is not None
    
    def _gives_check(self, position, color):
        return False  # Simplified
    
    def _creates_fork(self, position, move):
        return False  # Simplified
    
    def _creates_pin(self, position, move):
        return False  # Simplified
    
    def _evaluate_center_control(self, position):
        return 0.5  # Simplified
    
    def _evaluate_piece_activity(self, position):
        return 0.5  # Simplified
    
    def _evaluate_king_safety(self, position):
        return 0.5  # Simplified
    
    def _evaluate_pawn_structure(self, position):
        return 0.5  # Simplified
    
    def _is_development_move(self, position, move):
        return move['from'][1] in '18'  # From back rank
    
    def _improves_centralization(self, pos_before, pos_after, move):
        return move['to'][0] in 'de' and move['to'][1] in '45'
    
    def _is_pawn_break(self, move):
        return move['piece'] == 'pawn'
    
    def _improves_coordination(self, pos_before, pos_after):
        return True  # Simplified