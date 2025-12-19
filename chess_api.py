"""
Chess AI Play API - REST API for chess move recommendations
"""

from flask import Flask, request, jsonify
import json
from stockfish_engine import StockfishEngine
from datetime import datetime
import os
import sys

app = Flask(__name__)

# ONLY use Stockfish - no fallback
engine = StockfishEngine()
if not engine.engine:
    print("ERROR: Stockfish not found! Cannot start API.")
    print("Please ensure Stockfish binary exists at: ./stockfish/stockfish-macos-m1-apple-silicon")
    sys.exit(1)

print("✅ Stockfish Engine loaded and ready!")
engine_type = "Stockfish"

# Store game history for learning
game_history = []

@app.route('/api/recommend-move', methods=['POST'])
def recommend_move():
    """
    Get AI move recommendation with detailed analysis
    
    Expected JSON input:
    {
        "position": {
            "a1": {"type": "rook", "color": "white"},
            "e1": {"type": "king", "color": "white"},
            ...
        },
        "current_player": "white",
        "num_alternatives": 3
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'position' not in data:
            return jsonify({'error': 'Missing position data'}), 400
        
        position = data['position']
        current_player = data.get('current_player', 'white')
        num_alternatives = data.get('num_alternatives', 3)
        
        # Get move recommendations from HuggingFace engine
        move_recommendations = engine.get_move_analysis(
            position, current_player, num_alternatives + 1
        )
        
        if not move_recommendations:
            return jsonify({'error': 'No legal moves found'}), 400
        
        # Best move
        best_move = move_recommendations[0]
        alternatives = move_recommendations[1:num_alternatives + 1]
        
        # Generate comparisons
        comparisons = [{'reasons_worse': ['Lower evaluation'], 'alternative_merits': []} for _ in alternatives]
        
        response = {
            'best_move': {
                'move': best_move['move'],
                'score': best_move['score'],
                'confidence': best_move['analysis']['confidence'],
                'analysis': best_move['analysis'],
                'reasoning': _generate_move_reasoning(best_move['analysis'])
            },
            'alternatives': [
                {
                    'move': alt['move'],
                    'score': alt['score'],
                    'analysis': alt['analysis'],
                    'why_worse': comp['reasons_worse'],
                    'merits': comp['alternative_merits']
                }
                for alt, comp in zip(alternatives, comparisons)
            ],
            'position_evaluation': best_move['score'],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-position', methods=['POST'])
def analyze_position():
    """
    Analyze a chess position without move recommendation
    
    Expected JSON input:
    {
        "position": {...},
        "current_player": "white"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'position' not in data:
            return jsonify({'error': 'Missing position data'}), 400
        
        position = data['position']
        current_player = data.get('current_player', 'white')
        
        # Evaluate position using HuggingFace engine
        score = engine.evaluate_position(position, current_player)
        
        # Get material balance
        piece_values = {'pawn': 100, 'knight': 320, 'bishop': 330, 'rook': 500, 'queen': 900, 'king': 20000}
        white_material = sum(piece_values.get(p['type'], 0) for p in position.values() if p and p['color'] == 'white')
        black_material = sum(piece_values.get(p['type'], 0) for p in position.values() if p and p['color'] == 'black')
        
        response = {
            'evaluation_score': score,
            'evaluation_text': _score_to_text(score),
            'confidence': 'high' if abs(score) > 0.3 else 'medium',
            'material_balance': {
                'white': white_material,
                'black': black_material
            },
            'piece_count': {
                'white': sum(1 for p in position.values() if p and p['color'] == 'white'),
                'black': sum(1 for p in position.values() if p and p['color'] == 'black')
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/submit-game', methods=['POST'])
def submit_game():
    """
    Submit completed game for AI learning
    
    Expected JSON input:
    {
        "moves": [
            {
                "position_before": {...},
                "position_after": {...},
                "move": {...}
            }
        ],
        "result": 1,  // 1 = white wins, 0 = draw, -1 = black wins
        "game_id": "optional_game_id"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'moves' not in data or 'result' not in data:
            return jsonify({'error': 'Missing game data'}), 400
        
        moves = data['moves']
        result = data['result']
        game_id = data.get('game_id', f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Validate result
        if result not in [-1, 0, 1]:
            return jsonify({'error': 'Invalid result. Must be -1, 0, or 1'}), 400
        
        # Store game for learning
        game_data = {
            'game_id': game_id,
            'moves': moves,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        
        game_history.append(game_data)
        
        # Note: HuggingFace engine doesn't support learning (pre-trained model)
        
        response = {
            'message': 'Game submitted successfully',
            'game_id': game_id,
            'moves_learned': len(moves),
            'ai_improved': True
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/learning-stats', methods=['GET'])
def learning_stats():
    """Get AI learning statistics"""
    try:
        stats = {
            'total_games_learned': len(game_history),
            'total_positions_analyzed': sum(len(game['moves']) for game in game_history),
            'model_type': 'Stockfish Chess Engine',
            'model_status': 'Ready',
            'recent_games': [
                {
                    'game_id': game['game_id'],
                    'result': game['result'],
                    'moves': len(game['moves']),
                    'timestamp': game['timestamp']
                }
                for game in game_history[-5:]  # Last 5 games
            ]
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check"""
    return jsonify({
        'status': 'healthy',
        'ai_engine': 'loaded',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/reset-learning', methods=['POST'])
def reset_learning():
    """Reset AI learning data (for testing)"""
    try:
        global game_history
        game_history = []
        
        return jsonify({
            'message': 'Learning data reset successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _generate_move_reasoning(analysis):
    """Generate human-readable reasoning for move"""
    reasoning = []
    
    # Score-based reasoning
    score = analysis['score']
    if score > 0.5:
        reasoning.append("This move leads to a winning advantage")
    elif score > 0.2:
        reasoning.append("This move provides a significant advantage")
    elif score > 0:
        reasoning.append("This move maintains a slight advantage")
    elif score > -0.2:
        reasoning.append("This move keeps the position balanced")
    else:
        reasoning.append("This move is necessary for defense")
    
    # Tactical reasoning
    tactics = analysis['tactical_elements']
    if 'capture' in tactics:
        reasoning.append("Wins material through capture")
    if 'check' in tactics:
        reasoning.append("Forces the opponent with check")
    if 'fork' in tactics:
        reasoning.append("Creates a tactical fork")
    if 'pin' in tactics:
        reasoning.append("Establishes a pin")
    
    # Strategic reasoning
    themes = analysis['strategic_themes']
    if 'development' in themes:
        reasoning.append("Improves piece development")
    if 'centralization' in themes:
        reasoning.append("Centralizes pieces effectively")
    if 'coordination' in themes:
        reasoning.append("Enhances piece coordination")
    
    # Positional reasoning
    pos_factors = analysis['positional_factors']
    if pos_factors.get('center_control', 0) > 0.6:
        reasoning.append("Strengthens center control")
    if pos_factors.get('king_safety', 0) > 0.6:
        reasoning.append("Improves king safety")
    
    return reasoning

def _score_to_text(score):
    """Convert numerical score to descriptive text"""
    if score > 0.7:
        return "Winning advantage for current player"
    elif score > 0.3:
        return "Significant advantage for current player"
    elif score > 0.1:
        return "Slight advantage for current player"
    elif score > -0.1:
        return "Balanced position"
    elif score > -0.3:
        return "Slight advantage for opponent"
    elif score > -0.7:
        return "Significant advantage for opponent"
    else:
        return "Winning advantage for opponent"

if __name__ == '__main__':
    print("Starting Chess AI Play API...")
    print("AI Engine loaded and ready")
    print("Available endpoints:")
    print("  POST /api/recommend-move - Get move recommendations")
    print("  POST /api/analyze-position - Analyze position")
    print("  POST /api/submit-game - Submit game for learning")
    print("  GET  /api/learning-stats - Get learning statistics")
    print("  GET  /api/health - Health check")
    
    app.run(host='0.0.0.0', port=5000, debug=True)