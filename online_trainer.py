#!/usr/bin/env python3
"""
Online Chess Data Trainer - Fetches real games and trains the AI
"""

import requests
import json
import time
import chess
import chess.pgn
from io import StringIO
from chess_ai_engine import ChessAIEngine

class OnlineTrainer:
    def __init__(self):
        self.engine = ChessAIEngine()
        
    def fetch_lichess_games(self, max_games=50):
        """Fetch recent games from Lichess"""
        print(f"Fetching {max_games} games from Lichess...")
        
        # Get recent games from Lichess TV
        url = "https://lichess.org/api/tv/channels"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                channels = response.json()
                game_ids = []
                
                # Collect game IDs from different channels
                for channel_name, channel_data in channels.items():
                    if 'gameId' in channel_data:
                        game_ids.append(channel_data['gameId'])
                
                # Fetch actual games
                games_pgn = []
                for game_id in game_ids[:min(max_games, len(game_ids))]:
                    game_pgn = self.fetch_game_pgn(game_id)
                    if game_pgn:
                        games_pgn.append(game_pgn)
                        time.sleep(0.5)  # Rate limiting
                
                return games_pgn
                
        except Exception as e:
            print(f"Error fetching from Lichess: {e}")
        
        # Fallback to sample data
        return self.generate_sample_pgn_games(max_games)
    
    def fetch_game_pgn(self, game_id):
        """Fetch a single game PGN"""
        url = f"https://lichess.org/game/export/{game_id}"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.text
        except Exception as e:
            print(f"Error fetching game {game_id}: {e}")
        
        return None
    
    def generate_sample_pgn_games(self, num_games):
        """Generate sample PGN games for testing"""
        print("Generating sample PGN games...")
        
        sample_games = []
        
        for i in range(num_games):
            # Create a simple sample game
            pgn = f'''[Event "Sample Game {i+1}"]
[Site "Training"]
[Date "2024.01.01"]
[Round "1"]
[White "Player1"]
[Black "Player2"]
[Result "{'1-0' if i % 3 == 0 else '0-1' if i % 3 == 1 else '1/2-1/2'}"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7
'''
            sample_games.append(pgn)
        
        return sample_games
    
    def pgn_to_training_data(self, pgn_games):
        """Convert PGN games to training data"""
        print("Converting PGN games to training data...")
        
        training_games = []
        
        for pgn_text in pgn_games:
            try:
                game = chess.pgn.read_game(StringIO(pgn_text))
                if not game:
                    continue
                
                # Get result
                result_str = game.headers.get("Result", "*")
                if result_str == "1-0":
                    result = 1
                elif result_str == "0-1":
                    result = -1
                elif result_str == "1/2-1/2":
                    result = 0
                else:
                    continue
                
                # Convert moves
                board = game.board()
                moves_data = []
                
                move_count = 0
                for move in game.mainline_moves():
                    if move_count > 50:  # Limit moves per game
                        break
                        
                    pos_before = self.board_to_position(board)
                    board.push(move)
                    pos_after = self.board_to_position(board)
                    
                    move_data = {
                        "position_before": pos_before,
                        "position_after": pos_after,
                        "move": {
                            "from": chess.square_name(move.from_square),
                            "to": chess.square_name(move.to_square),
                            "piece": self.piece_type_to_name(board.piece_type_at(move.to_square)),
                            "color": "white" if not board.turn else "black"
                        }
                    }
                    moves_data.append(move_data)
                    move_count += 1
                
                if len(moves_data) >= 10:  # Only use games with sufficient moves
                    training_games.append({
                        "moves": moves_data,
                        "result": result
                    })
                    
            except Exception as e:
                print(f"Error parsing PGN: {e}")
                continue
        
        return training_games
    
    def board_to_position(self, board):
        """Convert chess board to position dict"""
        position = {}
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                position[chess.square_name(square)] = {
                    "type": self.piece_type_to_name(piece.piece_type),
                    "color": "white" if piece.color else "black"
                }
        
        return position
    
    def piece_type_to_name(self, piece_type):
        """Convert piece type to name"""
        if piece_type is None:
            return "unknown"
            
        piece_names = {
            chess.PAWN: "pawn",
            chess.KNIGHT: "knight", 
            chess.BISHOP: "bishop",
            chess.ROOK: "rook",
            chess.QUEEN: "queen",
            chess.KING: "king"
        }
        return piece_names.get(piece_type, "unknown")
    
    def train_with_online_data(self, max_games=30):
        """Main training function"""
        print("Starting online data training...")
        print("=" * 40)
        
        # Fetch games
        pgn_games = self.fetch_lichess_games(max_games)
        
        if not pgn_games:
            print("No games fetched. Exiting.")
            return
        
        print(f"Fetched {len(pgn_games)} games")
        
        # Convert to training data
        training_games = self.pgn_to_training_data(pgn_games)
        
        if not training_games:
            print("No valid training games found.")
            return
        
        print(f"Converted {len(training_games)} games to training data")
        
        # Train the model
        total_positions = 0
        successful_games = 0
        
        for i, game_data in enumerate(training_games):
            try:
                self.engine.learn_from_game_result(game_data["moves"], game_data["result"])
                total_positions += len(game_data["moves"])
                successful_games += 1
                
                if (i + 1) % 5 == 0:
                    print(f"Processed {i + 1}/{len(training_games)} games")
                    
            except Exception as e:
                print(f"Error training on game {i}: {e}")
        
        print(f"\nTraining Summary:")
        print(f"- Games processed: {successful_games}/{len(training_games)}")
        print(f"- Total positions: {total_positions}")
        print(f"- Model saved automatically")
        
        # Test the trained model
        self.test_trained_model()
    
    def test_trained_model(self):
        """Test the trained model"""
        print("\nTesting trained model...")
        
        # Create test position
        test_position = {
            "e1": {"type": "king", "color": "white"},
            "e8": {"type": "king", "color": "black"},
            "d1": {"type": "queen", "color": "white"},
            "d8": {"type": "queen", "color": "black"},
            "e2": {"type": "pawn", "color": "white"},
            "e7": {"type": "pawn", "color": "black"}
        }
        
        # Evaluate position
        score = self.engine.evaluate_position(test_position, "white")
        print(f"Position evaluation: {score:.3f}")
        
        # Get move recommendations
        try:
            moves = self.engine.get_best_moves(test_position, "white", 3)
            print(f"Generated {len(moves)} move recommendations")
            
            if moves:
                best = moves[0]
                print(f"Best move: {best['move']['from']}-{best['move']['to']}")
                print(f"Score: {best['score']:.3f}")
                print(f"Confidence: {best['analysis']['confidence']}")
        except Exception as e:
            print(f"Error getting moves: {e}")

def main():
    trainer = OnlineTrainer()
    
    print("Chess AI Online Trainer")
    print("=" * 25)
    
    num_games = input("Number of games to fetch (default 20): ").strip()
    num_games = int(num_games) if num_games else 20
    
    trainer.train_with_online_data(num_games)

if __name__ == "__main__":
    main()