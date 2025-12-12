#!/usr/bin/env python3
"""
Simple training script with sample data
"""

from chess_ai_engine import ChessAIEngine
import random

def create_sample_position():
    """Create a sample chess position"""
    return {
        "e1": {"type": "king", "color": "white"},
        "e8": {"type": "king", "color": "black"},
        "d1": {"type": "queen", "color": "white"},
        "d8": {"type": "queen", "color": "black"},
        "a1": {"type": "rook", "color": "white"},
        "h1": {"type": "rook", "color": "white"},
        "a8": {"type": "rook", "color": "black"},
        "h8": {"type": "rook", "color": "black"},
        "e2": {"type": "pawn", "color": "white"},
        "e7": {"type": "pawn", "color": "black"}
    }

def generate_training_game():
    """Generate a sample training game"""
    moves = []
    position = create_sample_position()
    
    # Simulate 20-30 moves
    for i in range(random.randint(20, 30)):
        # Create a slightly modified position
        new_position = position.copy()
        
        # Randomly modify position (simulate game progression)
        if random.random() < 0.3:  # 30% chance to remove a pawn
            pawns = [sq for sq, piece in new_position.items() 
                    if piece and piece["type"] == "pawn"]
            if pawns:
                del new_position[random.choice(pawns)]
        
        move_data = {
            "position_before": position,
            "position_after": new_position,
            "move": {
                "from": "e2", "to": "e4",
                "piece": "pawn",
                "color": "white" if i % 2 == 0 else "black"
            }
        }
        moves.append(move_data)
        position = new_position
    
    # Random game result
    result = random.choice([-1, 0, 1])
    return moves, result

def train_model():
    """Train the model with sample data"""
    print("Training Chess AI with sample data...")
    
    engine = ChessAIEngine()
    
    # Generate and train on sample games
    for i in range(10):  # Train on 10 games
        moves, result = generate_training_game()
        
        print(f"Training on game {i+1}/10 (result: {result})")
        
        try:
            engine.learn_from_game_result(moves, result)
        except Exception as e:
            print(f"Error in game {i+1}: {e}")
    
    print("Training completed!")
    print("Testing the trained model...")
    
    # Test the model
    test_position = create_sample_position()
    score = engine.evaluate_position(test_position, "white")
    print(f"Test position evaluation: {score:.3f}")
    
    # Get move recommendations
    try:
        best_moves = engine.get_best_moves(test_position, "white", 3)
        print(f"Found {len(best_moves)} move recommendations")
        
        if best_moves:
            best_move = best_moves[0]
            print(f"Best move: {best_move['move']}")
            print(f"Score: {best_move['score']:.3f}")
    except Exception as e:
        print(f"Error getting moves: {e}")

if __name__ == "__main__":
    train_model()