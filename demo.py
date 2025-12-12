"""
Chess AI Play API Demo - Direct Engine Usage
"""

from chess_ai_engine import ChessAIEngine
import json

def create_starting_position():
    """Create standard chess starting position"""
    return {
        # White pieces
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
        
        # Black pieces
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

def create_tactical_position():
    """Create a tactical position for demonstration"""
    return {
        # White pieces
        "e1": {"type": "king", "color": "white"},
        "d4": {"type": "queen", "color": "white"},
        "c3": {"type": "knight", "color": "white"},
        "f3": {"type": "knight", "color": "white"},
        "c4": {"type": "bishop", "color": "white"},
        "a1": {"type": "rook", "color": "white"},
        "h1": {"type": "rook", "color": "white"},
        "a2": {"type": "pawn", "color": "white"},
        "b2": {"type": "pawn", "color": "white"},
        "e4": {"type": "pawn", "color": "white"},
        "f2": {"type": "pawn", "color": "white"},
        "g2": {"type": "pawn", "color": "white"},
        "h2": {"type": "pawn", "color": "white"},
        
        # Black pieces
        "e8": {"type": "king", "color": "black"},
        "d8": {"type": "queen", "color": "black"},
        "c6": {"type": "knight", "color": "black"},
        "f6": {"type": "knight", "color": "black"},
        "c5": {"type": "bishop", "color": "black"},
        "a8": {"type": "rook", "color": "black"},
        "h8": {"type": "rook", "color": "black"},
        "a7": {"type": "pawn", "color": "black"},
        "b7": {"type": "pawn", "color": "black"},
        "e5": {"type": "pawn", "color": "black"},
        "f7": {"type": "pawn", "color": "black"},
        "g7": {"type": "pawn", "color": "black"},
        "h7": {"type": "pawn", "color": "black"}
    }

def demo_move_recommendation():
    """Demonstrate move recommendation functionality"""
    print("🤖 Chess AI Move Recommendation Demo")
    print("=" * 50)
    
    # Initialize AI engine
    engine = ChessAIEngine()
    print("✓ AI Engine loaded and ready")
    print()
    
    # Demo 1: Starting position analysis
    print("📋 Demo 1: Opening Move Analysis")
    print("-" * 30)
    
    starting_position = create_starting_position()
    
    # Get position evaluation
    score = engine.evaluate_position(starting_position, "white")
    print(f"Position Evaluation: {score:.3f}")
    print(f"Assessment: {'Balanced starting position' if abs(score) < 0.1 else 'Slight advantage detected'}")
    print()
    
    # Get move recommendations
    recommendations = engine.get_best_moves(starting_position, "white", 3)
    
    print("🎯 AI Move Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        move = rec['move']
        analysis = rec['analysis']
        
        print(f"{i}. {move['from']} → {move['to']} ({move['piece']})")
        print(f"   Score: {rec['score']:.3f}")
        print(f"   Confidence: {analysis['confidence']}")
        print(f"   Tactics: {', '.join(analysis['tactical_elements']) or 'None'}")
        print(f"   Themes: {', '.join(analysis['strategic_themes']) or 'None'}")
        print()
    
    # Demo 2: Tactical position
    print("⚔️  Demo 2: Tactical Position Analysis")
    print("-" * 30)
    
    tactical_position = create_tactical_position()
    
    tactical_score = engine.evaluate_position(tactical_position, "white")
    print(f"Position Evaluation: {tactical_score:.3f}")
    
    if tactical_score > 0.3:
        assessment = "Strong advantage for White"
    elif tactical_score > 0.1:
        assessment = "Slight advantage for White"
    elif tactical_score > -0.1:
        assessment = "Balanced position"
    elif tactical_score > -0.3:
        assessment = "Slight advantage for Black"
    else:
        assessment = "Strong advantage for Black"
    
    print(f"Assessment: {assessment}")
    print()
    
    tactical_recs = engine.get_best_moves(tactical_position, "white", 2)
    
    print("🎯 Tactical Recommendations:")
    for i, rec in enumerate(tactical_recs, 1):
        move = rec['move']
        analysis = rec['analysis']
        
        print(f"{i}. {move['from']} → {move['to']} ({move['piece']})")
        print(f"   Score: {rec['score']:.3f}")
        print(f"   Material Change: {analysis.get('material_change', {})}")
        print(f"   Tactical Elements: {', '.join(analysis['tactical_elements']) or 'Positional'}")
        print()
    
    # Demo 3: Move comparison
    if len(recommendations) >= 2:
        print("🔍 Demo 3: Move Comparison Analysis")
        print("-" * 30)
        
        best_move = recommendations[0]
        alternatives = recommendations[1:2]
        
        comparisons = engine.compare_moves(best_move, alternatives)
        
        print(f"Best Move: {best_move['move']['from']} → {best_move['move']['to']}")
        print(f"Score: {best_move['score']:.3f}")
        print()
        
        for comp in comparisons:
            alt_move = comp['move']
            print(f"Alternative: {alt_move['from']} → {alt_move['to']}")
            print(f"Score Difference: {comp['score_difference']:.3f}")
            print("Why it's worse:")
            for reason in comp['reasons_worse']:
                print(f"  • {reason}")
            
            if comp['alternative_merits']:
                print("Alternative merits:")
                for merit in comp['alternative_merits']:
                    print(f"  • {merit}")
            print()
    
    # Demo 4: Learning simulation
    print("🧠 Demo 4: AI Learning Simulation")
    print("-" * 30)
    
    # Simulate a game for learning
    sample_game = [
        {
            'position_before': starting_position,
            'position_after': tactical_position,
            'move': {'from': 'e2', 'to': 'e4', 'piece': 'pawn', 'color': 'white'}
        }
    ]
    
    print("Submitting sample game for AI learning...")
    engine.learn_from_game_result(sample_game, 1)  # White wins
    
    print(f"✓ AI learned from {len(sample_game)} positions")
    print(f"✓ Learning data size: {len(engine.learning_data)}")
    print(f"✓ Position cache size: {len(engine.position_cache)}")
    
    print()
    print("🎉 Demo completed successfully!")
    print()
    print("Key Features Demonstrated:")
    print("• Neural network position evaluation")
    print("• AI-powered move recommendations")
    print("• Detailed tactical and strategic analysis")
    print("• Move comparison with explanations")
    print("• Continuous learning from game results")
    print("• Independent operation (no external dependencies)")

if __name__ == "__main__":
    demo_move_recommendation()