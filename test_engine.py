"""
Test the Chess AI Engine independently
"""

from chess_ai_engine import ChessAIEngine

def test_ai_engine():
    """Test the AI engine functionality"""
    print("Testing Chess AI Engine...")
    print("=" * 40)
    
    # Initialize engine
    engine = ChessAIEngine()
    print("✓ AI Engine initialized")
    
    # Create test position
    test_position = {
        "e1": {"type": "king", "color": "white"},
        "e8": {"type": "king", "color": "black"},
        "e2": {"type": "pawn", "color": "white"},
        "e7": {"type": "pawn", "color": "black"},
        "d1": {"type": "queen", "color": "white"},
        "d8": {"type": "queen", "color": "black"}
    }
    
    print("✓ Test position created")
    
    # Test position evaluation
    score = engine.evaluate_position(test_position, "white")
    print(f"✓ Position evaluation: {score:.3f}")
    
    # Test move recommendations
    moves = engine.get_best_moves(test_position, "white", 3)
    print(f"✓ Generated {len(moves)} move recommendations")
    
    if moves:
        best_move = moves[0]
        print(f"  Best move: {best_move['move']['from']} -> {best_move['move']['to']}")
        print(f"  Score: {best_move['score']:.3f}")
        print(f"  Confidence: {best_move['analysis']['confidence']}")
    
    # Test learning
    sample_game = [
        {
            'position_before': test_position,
            'position_after': test_position,
            'move': {'from': 'e2', 'to': 'e4', 'piece': 'pawn', 'color': 'white'}
        }
    ]
    
    engine.learn_from_game_result(sample_game, 1)
    print("✓ Learning from game result completed")
    
    print("\nAI Engine test completed successfully! 🎉")

if __name__ == "__main__":
    test_ai_engine()