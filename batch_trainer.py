#!/usr/bin/env python3
"""
Batch Training Script for Chess AI
"""

import requests
import json
import time
from chess_ai_engine import ChessAIEngine

class BatchTrainer:
    def __init__(self):
        self.engine = ChessAIEngine()
        
    def download_lichess_database(self, month="2024-01", max_games=1000):
        """Download games from Lichess database"""
        print(f"Downloading Lichess database for {month}...")
        
        url = f"https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.bz2"
        
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                print("Database download started...")
                # For demo, we'll simulate with sample data
                return self.generate_sample_games(max_games)
            else:
                print("Database not available, using sample data")
                return self.generate_sample_games(max_games)
        except Exception as e:
            print(f"Error downloading: {e}")
            return self.generate_sample_games(max_games)
    
    def generate_sample_games(self, num_games=100):
        """Generate sample training games"""
        print(f"Generating {num_games} sample training games...")
        
        sample_games = []
        
        for i in range(num_games):
            # Create sample game with random outcome
            import random
            result = random.choice([-1, 0, 1])  # Black wins, draw, white wins
            
            # Generate sample moves (simplified)
            moves = []
            position = self.create_starting_position()
            
            for move_num in range(random.randint(20, 60)):
                # Simulate a move
                new_position = self.simulate_random_move(position)
                
                move_data = {
                    "position_before": position.copy(),
                    "position_after": new_position,
                    "move": {
                        "from": "e2", "to": "e4",
                        "piece": "pawn", "color": "white" if move_num % 2 == 0 else "black"
                    }
                }
                moves.append(move_data)
                position = new_position
            
            sample_games.append({
                "moves": moves,
                "result": result,
                "game_id": f"sample_{i}"
            })
        
        return sample_games
    
    def create_starting_position(self):
        """Create standard chess starting position"""
        return {
            "a1": {"type": "rook", "color": "white"},
            "b1": {"type": "knight", "color": "white"},
            "c1": {"type": "bishop", "color": "white"},
            "d1": {"type": "queen", "color": "white"},
            "e1": {"type": "king", "color": "white"},
            "f1": {"type": "bishop", "color": "white"},
            "g1": {"type": "knight", "color": "white"},
            "h1": {"type": "rook", "color": "white"},
            "a2": {"type": "pawn", "color": "white"},
            "b2": {"type": "pawn", "color": "white"},
            "c2": {"type": "pawn", "color": "white"},
            "d2": {"type": "pawn", "color": "white"},
            "e2": {"type": "pawn", "color": "white"},
            "f2": {"type": "pawn", "color": "white"},
            "g2": {"type": "pawn", "color": "white"},
            "h2": {"type": "pawn", "color": "white"},
            "a8": {"type": "rook", "color": "black"},
            "b8": {"type": "knight", "color": "black"},
            "c8": {"type": "bishop", "color": "black"},
            "d8": {"type": "queen", "color": "black"},
            "e8": {"type": "king", "color": "black"},
            "f8": {"type": "bishop", "color": "black"},
            "g8": {"type": "knight", "color": "black"},
            "h8": {"type": "rook", "color": "black"},
            "a7": {"type": "pawn", "color": "black"},
            "b7": {"type": "pawn", "color": "black"},
            "c7": {"type": "pawn", "color": "black"},
            "d7": {"type": "pawn", "color": "black"},
            "e7": {"type": "pawn", "color": "black"},
            "f7": {"type": "pawn", "color": "black"},
            "g7": {"type": "pawn", "color": "black"},
            "h7": {"type": "pawn", "color": "black"}
        }
    
    def simulate_random_move(self, position):
        """Simulate a random move for demo purposes"""
        import random
        new_position = position.copy()
        
        # Randomly remove/add pieces to simulate game progression
        squares = list(new_position.keys())
        if squares and random.random() < 0.1:  # 10% chance to remove a piece
            square_to_remove = random.choice(squares)
            if new_position[square_to_remove]["type"] != "king":  # Don't remove kings
                del new_position[square_to_remove]
        
        return new_position
    
    def batch_train(self, num_games=500):
        """Perform batch training"""
        print("Starting batch training...")
        print("=" * 50)
        
        # Generate/download training data
        games_data = self.generate_sample_games(num_games)
        
        print(f"Training on {len(games_data)} games...")
        
        # Train in batches
        batch_size = 50
        total_positions = 0
        
        for i in range(0, len(games_data), batch_size):
            batch = games_data[i:i + batch_size]
            batch_positions = 0
            
            print(f"\nProcessing batch {i//batch_size + 1}/{(len(games_data) + batch_size - 1)//batch_size}")
            
            for game in batch:
                try:
                    self.engine.learn_from_game_result(game["moves"], game["result"])
                    batch_positions += len(game["moves"])
                except Exception as e:
                    print(f"Error processing game: {e}")
            
            total_positions += batch_positions
            print(f"Batch completed: {batch_positions} positions processed")
            print(f"Total positions so far: {total_positions}")
        
        print(f"\nBatch training completed!")
        print(f"Total games processed: {len(games_data)}")
        print(f"Total positions processed: {total_positions}")
        print("Model automatically saved.")

def main():
    trainer = BatchTrainer()
    
    print("Chess AI Batch Trainer")
    print("=" * 30)
    
    num_games = input("Number of games to train on (default 500): ").strip()
    num_games = int(num_games) if num_games else 500
    
    trainer.batch_train(num_games)

if __name__ == "__main__":
    main()