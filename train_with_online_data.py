#!/usr/bin/env python3
"""
Train Chess AI with Online Game Data from Lichess
"""

import requests
import json
import time
from chess_ai_engine import ChessAIEngine
import chess
import chess.pgn
from io import StringIO

class OnlineDataTrainer:
    def __init__(self):
        self.engine = ChessAIEngine()
        self.lichess_api = "https://lichess.org/api"
        
    def fetch_games(self, username=None, max_games=100, time_control="blitz"):
        """Fetch games from Lichess API"""
        print(f"Fetching {max_games} games from Lichess...")
        
        if username:
            url = f"{self.lichess_api}/games/user/{username}"
        else:
            url = f"{self.lichess_api}/games/export/_ids"
            
        params = {
            "max": max_games,
            "perfType": time_control,
            "format": "pgn",
            "clocks": "false",
            "evals": "false",
            "opening": "false"
        }
        
        try:
            response = requests.get(url, params=params, stream=True)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching games: {e}")
            return None
    
    def fetch_top_games(self, max_games=50):
        """Fetch recent games from top players"""
        print(f"Fetching {max_games} games from top players...")
        
        # Get top players first
        try:
            response = requests.get(f"{self.lichess_api}/player/top/10/classical")
            top_players = response.json()["users"][:5]  # Top 5 players
            
            all_games = []
            for player in top_players:
                username = player["username"]
                games = self.fetch_games(username, max_games//5, "classical")
                if games:
                    all_games.append(games)
                time.sleep(1)  # Rate limiting
            
            return "\n\n".join(all_games)
        except Exception as e:
            print(f"Error fetching top games: {e}")
            return None
    
    def pgn_to_positions(self, pgn_text):
        """Convert PGN games to position sequences"""
        games_data = []
        
        for game_pgn in pgn_text.split("\n\n"):
            if not game_pgn.strip():
                continue
                
            try:
                game = chess.pgn.read_game(StringIO(game_pgn))
                if not game:
                    continue
                    
                # Get game result
                result_str = game.headers.get("Result", "*")
                if result_str == "1-0":
                    result = 1  # White wins
                elif result_str == "0-1":
                    result = -1  # Black wins
                elif result_str == "1/2-1/2":
                    result = 0  # Draw
                else:
                    continue  # Skip unfinished games
                
                # Extract moves and positions
                board = game.board()
                moves_data = []
                
                for move in game.mainline_moves():
                    pos_before = self.board_to_position(board)
                    board.push(move)
                    pos_after = self.board_to_position(board)
                    
                    move_data = {
                        "position_before": pos_before,
                        "position_after": pos_after,
                        "move": {
                            "from": chess.square_name(move.from_square),
                            "to": chess.square_name(move.to_square),
                            "piece": board.piece_type_at(move.to_square),
                            "color": "white" if board.turn else "black"
                        }
                    }
                    moves_data.append(move_data)
                
                if len(moves_data) > 10:  # Only use games with reasonable length
                    games_data.append({
                        "moves": moves_data,
                        "result": result
                    })
                    
            except Exception as e:
                print(f"Error parsing game: {e}")
                continue
        
        return games_data
    
    def board_to_position(self, board):
        """Convert python-chess board to engine position format"""
        position = {}
        
        piece_map = {
            chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
            chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                position[chess.square_name(square)] = {
                    "type": piece_map[piece.piece_type],
                    "color": "white" if piece.color else "black"
                }
        
        return position
    
    def train_from_online_data(self, max_games=100, source="top_players"):
        """Main training function"""
        print("Starting online data training...")
        
        # Fetch games
        if source == "top_players":
            pgn_data = self.fetch_top_games(max_games)
        else:
            pgn_data = self.fetch_games(max_games=max_games)
        
        if not pgn_data:
            print("No games fetched. Exiting.")
            return
        
        # Convert to training data
        print("Converting games to training data...")
        games_data = self.pgn_to_positions(pgn_data)
        
        if not games_data:
            print("No valid games found. Exiting.")
            return
        
        print(f"Successfully parsed {len(games_data)} games")
        
        # Train the model
        print("Training AI model...")
        total_positions = 0
        
        for i, game_data in enumerate(games_data):
            try:
                self.engine.learn_from_game_result(game_data["moves"], game_data["result"])
                total_positions += len(game_data["moves"])
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(games_data)} games ({total_positions} positions)")
                    
            except Exception as e:
                print(f"Error training on game {i}: {e}")
                continue
        
        print(f"Training completed! Processed {total_positions} positions from {len(games_data)} games")
        print("Model has been automatically saved.")

def main():
    """Main training script"""
    trainer = OnlineDataTrainer()
    
    print("Chess AI Online Training")
    print("=" * 40)
    
    # Training options
    print("Training options:")
    print("1. Train with top player games (50 games)")
    print("2. Train with random games (100 games)")
    print("3. Custom training")
    
    choice = input("Select option (1-3): ").strip()
    
    if choice == "1":
        trainer.train_from_online_data(max_games=50, source="top_players")
    elif choice == "2":
        trainer.train_from_online_data(max_games=100, source="random")
    elif choice == "3":
        max_games = int(input("Number of games: "))
        source = input("Source (top_players/random): ")
        trainer.train_from_online_data(max_games=max_games, source=source)
    else:
        print("Invalid choice. Using default: top player games")
        trainer.train_from_online_data(max_games=50, source="top_players")

if __name__ == "__main__":
    main()