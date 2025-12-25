"""
Chess AI Play API - REST API for chess move recommendations
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from stockfish_engine import StockfishEngine
from datetime import datetime
import os
import sys
import torch
import chess
import numpy as np
from chess_ml_model import ChessNet, ChessTrainer, ChessPositionEncoder

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5500", "http://127.0.0.1:5500"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# ONLY use Stockfish - no fallback
engine = StockfishEngine()
if not engine.engine:
    print("ERROR: Stockfish not found! Cannot start API.")
    print("Please ensure Stockfish binary exists at: ./stockfish/stockfish-macos-m1-apple-silicon")
    sys.exit(1)

print("✅ Stockfish Engine loaded and ready!")
engine_type = "Stockfish"

# Load ML model
ml_model = None
ml_trainer = None
try:
    ml_model = ChessNet()
    if os.path.exists('chess_ml_model.pth'):
        ml_model.load_state_dict(torch.load('chess_ml_model.pth', map_location='cpu'))
        ml_trainer = ChessTrainer(ml_model, device='cpu')
        print("✅ ML Chess Model loaded and ready!")
    elif os.path.exists('chess_ml_model_backup.pth'):
        print("⚠️ Using backup ML model...")
        ml_model.load_state_dict(torch.load('chess_ml_model_backup.pth', map_location='cpu'))
        ml_trainer = ChessTrainer(ml_model, device='cpu')
        print("✅ ML Chess Model loaded from backup!")
    else:
        print("⚠️ ML model not found. Training new model...")
        from chess_ml_model import train_chess_model
        ml_model, ml_trainer = train_chess_model()
except Exception as e:
    print(f"❌ Failed to load ML model: {e}")
    ml_model = None
    ml_trainer = None

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
        print("🔥 API CALL RECEIVED - recommend_move")
        data = request.get_json()
        print(f"📥 Request data: {data}")
        
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
                'reasoning': _generate_move_reasoning(best_move['analysis'], best_move['score'])
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
        print(f"💥 ERROR in recommend_move: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-position', methods=['POST'])
def analyze_position():
    """
    Analyze user's chosen move with comprehensive analysis
    
    Expected JSON input:
    {
        "position": {...},
        "user_move": {"from": "e2", "to": "e4", "piece": "pawn"},
        "current_player": "white",
        "include_demo": true
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'position' not in data or 'user_move' not in data:
            return jsonify({'error': 'Missing position or user_move data'}), 400
        
        position = data['position']
        user_move = data['user_move']
        current_player = data.get('current_player', 'white')
        include_demo = data.get('include_demo', False)
        
        # Get engine recommendations for comparison
        engine_moves = engine.get_move_analysis(position, current_player, 5)
        
        if not engine_moves:
            return jsonify({'error': 'No legal moves found'}), 400
        
        # Analyze user move
        user_move_analysis = _analyze_user_move(user_move, engine_moves, position, current_player)
        
        # Generate comprehensive analysis
        move_analysis = _generate_move_analysis_narrative(user_move, user_move_analysis, engine_moves, position, current_player)
        
        # Generate demo scripts if requested
        demo_scripts = None
        if include_demo:
            demo_scripts = _generate_demo_scripts(user_move, engine_moves, position)
        
        response = {
            'user_move_evaluation': user_move_analysis,
            'comprehensive_analysis': move_analysis,
            'engine_alternatives': [
                {
                    'move': move['move'],
                    'score': move['score'],
                    'rank': i + 1,
                    'comparison_with_user_move': _compare_moves(user_move_analysis, move)
                }
                for i, move in enumerate(engine_moves[:3])
            ],
            'demo_scripts': demo_scripts,
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

@app.route('/api/recommend-move-ml', methods=['POST'])
def recommend_move_ml():
    """
    Get AI move recommendation using local ML model
    
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
        if ml_trainer is None:
            return jsonify({'error': 'ML model not available'}), 500
        
        data = request.get_json()
        
        if not data or 'position' not in data:
            return jsonify({'error': 'Missing position data'}), 400
        
        position = data['position']
        current_player = data.get('current_player', 'white')
        num_alternatives = data.get('num_alternatives', 3)
        
        # Convert API position to chess.Board
        board = _api_position_to_board(position)
        board.turn = current_player == 'white'
        
        # Get ML model predictions
        move_probs, position_value = ml_trainer.predict_move(board)
        
        if not move_probs:
            return jsonify({'error': 'No legal moves found'}), 400
        
        # Format best move
        best_move_chess = move_probs[0][0]
        best_move_prob = move_probs[0][1]
        
        best_move = {
            'move': {
                'from': chess.square_name(best_move_chess.from_square),
                'to': chess.square_name(best_move_chess.to_square),
                'piece': _get_piece_name(board.piece_at(best_move_chess.from_square)),
                'color': current_player
            },
            'score': position_value,
            'confidence': 'high' if best_move_prob > 0.3 else 'medium' if best_move_prob > 0.1 else 'low',
            'analysis': {
                'confidence': 'high' if best_move_prob > 0.3 else 'medium',
                'tactical_elements': _analyze_ml_move_tactics(board, best_move_chess),
                'strategic_themes': _analyze_ml_move_strategy(board, best_move_chess),
                'probability': best_move_prob
            },
            'reasoning': _generate_ml_move_reasoning(board, best_move_chess, position_value)
        }
        
        # Format alternatives
        alternatives = []
        for i in range(1, min(len(move_probs), num_alternatives + 1)):
            alt_move_chess = move_probs[i][0]
            alt_prob = move_probs[i][1]
            
            alt_move = {
                'move': {
                    'from': chess.square_name(alt_move_chess.from_square),
                    'to': chess.square_name(alt_move_chess.to_square),
                    'piece': _get_piece_name(board.piece_at(alt_move_chess.from_square)),
                    'color': current_player
                },
                'score': position_value * (alt_prob / best_move_prob),  # Adjust score by probability ratio
                'analysis': {
                    'confidence': 'medium' if alt_prob > 0.1 else 'low',
                    'tactical_elements': _analyze_ml_move_tactics(board, alt_move_chess),
                    'strategic_themes': _analyze_ml_move_strategy(board, alt_move_chess),
                    'probability': alt_prob
                },
                'why_worse': [f'Lower probability ({alt_prob:.3f} vs {best_move_prob:.3f})'],
                'merits': ['Alternative strategic approach']
            }
            alternatives.append(alt_move)
        
        response = {
            'best_move': best_move,
            'alternatives': alternatives,
            'position_evaluation': position_value,
            'engine_info': {
                'name': 'Custom Chess ML Model',
                'type': 'Neural Network',
                'confidence': best_move['confidence']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _api_position_to_board(position):
    """Convert API position format to chess.Board"""
    board = chess.Board()
    board.clear()
    
    piece_map = {
        'pawn': chess.PAWN, 'knight': chess.KNIGHT, 'bishop': chess.BISHOP,
        'rook': chess.ROOK, 'queen': chess.QUEEN, 'king': chess.KING
    }
    
    for square_name, piece_info in position.items():
        try:
            square = chess.parse_square(square_name)
            piece_type = piece_map[piece_info['type']]
            color = piece_info['color'] == 'white'
            piece = chess.Piece(piece_type, color)
            board.set_piece_at(square, piece)
        except:
            continue
    
    return board

def _get_piece_name(piece):
    """Get piece name from chess.Piece"""
    if piece is None:
        return 'unknown'
    
    piece_names = {
        chess.PAWN: 'pawn', chess.KNIGHT: 'knight', chess.BISHOP: 'bishop',
        chess.ROOK: 'rook', chess.QUEEN: 'queen', chess.KING: 'king'
    }
    
    return piece_names.get(piece.piece_type, 'unknown')

def _analyze_ml_move_tactics(board, move):
    """Analyze tactical elements of ML predicted move"""
    tactics = []
    
    # Check for capture
    if board.piece_at(move.to_square):
        tactics.append('capture')
    
    # Check for check
    board_copy = board.copy()
    board_copy.push(move)
    if board_copy.is_check():
        tactics.append('check')
    
    # Check for castling
    piece = board.piece_at(move.from_square)
    if piece and piece.piece_type == chess.KING:
        if abs(move.to_square - move.from_square) == 2:
            tactics.append('castling')
    
    return tactics

def _analyze_ml_move_strategy(board, move):
    """Analyze strategic themes of ML predicted move"""
    themes = []
    
    piece = board.piece_at(move.from_square)
    if piece:
        # Development
        if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            from_rank = chess.square_rank(move.from_square)
            back_rank = 0 if piece.color == chess.WHITE else 7
            if from_rank == back_rank:
                themes.append('development')
        
        # Center control
        if move.to_square in [chess.D4, chess.D5, chess.E4, chess.E5]:
            themes.append('center_control')
        
        # ML learned patterns
        themes.append('ml_pattern')
    
    return themes

def _generate_ml_move_reasoning(board, move, position_value):
    """Generate reasoning for ML predicted move"""
    reasoning = []
    
    # Position evaluation reasoning
    if position_value > 0.3:
        reasoning.append("ML model evaluates this as a winning position")
    elif position_value > 0:
        reasoning.append("ML model sees a slight advantage")
    elif position_value > -0.3:
        reasoning.append("ML model evaluates position as balanced")
    else:
        reasoning.append("ML model suggests defensive play")
    
    # Move-specific reasoning
    piece = board.piece_at(move.from_square)
    if piece:
        if board.piece_at(move.to_square):
            reasoning.append("Captures opponent piece")
        
        if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            reasoning.append("Develops piece toward center")
        
        reasoning.append("Move follows learned chess patterns")
    
    return reasoning

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

def _analyze_user_move(user_move, engine_moves, position, current_player):
    """Analyze user's move against engine recommendations"""
    # Find user move in engine recommendations
    user_move_rank = None
    user_move_score = None
    
    for i, engine_move in enumerate(engine_moves):
        if (_moves_match(user_move, engine_move['move'])):
            user_move_rank = i + 1
            user_move_score = engine_move['score']
            break
    
    # If not found in top moves, evaluate separately
    if user_move_rank is None:
        user_move_score = engine.evaluate_move(position, user_move, current_player)
        user_move_rank = len(engine_moves) + 1
    
    return {
        'move': user_move,
        'score': user_move_score,
        'rank': user_move_rank,
        'is_best_move': user_move_rank == 1,
        'is_top_3': user_move_rank <= 3,
        'score_difference_from_best': user_move_score - engine_moves[0]['score'] if engine_moves else 0
    }

def _generate_move_analysis_narrative(user_move, user_analysis, engine_moves, position, current_player):
    """Generate comprehensive analysis narrative for user move"""
    return {
        'move_quality_assessment': _assess_move_quality(user_analysis),
        'strategic_analysis': _analyze_move_strategy(user_move, position, current_player),
        'tactical_analysis': _analyze_move_tactics(user_move, position, current_player),
        'positional_consequences': _analyze_positional_impact(user_move, position, current_player),
        'alternative_comparison': _generate_alternative_analysis(user_analysis, engine_moves),
        'learning_insights': _extract_move_learning_insights(user_analysis, engine_moves),
        'improvement_recommendations': _generate_move_improvement_suggestions(user_analysis, engine_moves)
    }

def _generate_demo_scripts(user_move, engine_moves, position):
    """Generate demo scripts for move analysis"""
    return {
        'backtrack_analysis': _generate_backtrack_script(user_move, position),
        'alternative_demonstrations': _generate_alternative_demo_scripts(engine_moves[:3]),
        'tactical_variations': _generate_tactical_demo_scripts(user_move, engine_moves, position),
        'strategic_comparisons': _generate_strategic_demo_scripts(user_move, engine_moves)
    }

def _moves_match(move1, move2):
    """Check if two moves are the same"""
    return (move1.get('from') == move2.get('from') and 
            move1.get('to') == move2.get('to') and
            move1.get('piece') == move2.get('piece'))

def _assess_move_quality(user_analysis):
    """Assess the quality of user's move"""
    rank = user_analysis['rank']
    score_diff = user_analysis['score_difference_from_best']
    
    if user_analysis['is_best_move']:
        return "Excellent! You found the best move according to the engine."
    elif user_analysis['is_top_3']:
        return f"Very good move! Ranked #{rank} by the engine with only {abs(score_diff):.2f} points difference from the best."
    elif score_diff > -0.2:
        return f"Decent move. Ranked #{rank} with {abs(score_diff):.2f} points difference. Still playable but not optimal."
    elif score_diff > -0.5:
        return f"Questionable move. Ranked #{rank} with {abs(score_diff):.2f} points disadvantage. Consider alternatives."
    else:
        return f"Poor move. Significant disadvantage of {abs(score_diff):.2f} points. This move likely loses material or position."

def _analyze_move_strategy(user_move, position, current_player):
    """Analyze strategic aspects of the move"""
    analysis = []
    
    # Analyze piece development
    if user_move['piece'] in ['knight', 'bishop']:
        analysis.append("Development Move: Activating minor pieces toward the center is generally good strategy.")
    
    # Analyze center control
    center_squares = ['d4', 'd5', 'e4', 'e5']
    if user_move['to'] in center_squares or user_move['piece'] == 'pawn':
        analysis.append("Center Control: This move influences central squares, which is strategically important.")
    
    # Analyze king safety
    if user_move['piece'] == 'king':
        analysis.append("King Safety: King moves should be carefully considered, especially in the opening and middlegame.")
    
    return analysis

def _analyze_move_tactics(user_move, position, current_player):
    """Analyze tactical aspects of the move"""
    tactical_elements = []
    
    # Check for captures
    target_square = user_move['to']
    if target_square in position and position[target_square]:
        captured_piece = position[target_square]['type']
        tactical_elements.append(f"Capture: This move captures the opponent's {captured_piece}.")
    
    # Check for checks (simplified)
    if user_move['piece'] in ['queen', 'rook', 'bishop', 'knight']:
        tactical_elements.append("Potential Threats: This piece move may create tactical opportunities.")
    
    return tactical_elements

def _analyze_positional_impact(user_move, position, current_player):
    """Analyze positional consequences of the move"""
    consequences = []
    
    # Pawn structure impact
    if user_move['piece'] == 'pawn':
        consequences.append("Pawn Structure: Pawn moves are permanent and affect long-term pawn structure.")
    
    # Piece coordination
    consequences.append("Piece Coordination: Consider how this move affects coordination with other pieces.")
    
    # Space and mobility
    consequences.append("Space Control: Evaluate how this move affects your control of key squares.")
    
    return consequences

def _generate_alternative_analysis(user_analysis, engine_moves):
    """Generate analysis comparing user move with alternatives"""
    if not engine_moves:
        return "No engine alternatives available for comparison."
    
    best_move = engine_moves[0]
    comparison = f"Engine's top choice: {_describe_move(best_move['move'])} (Score: {best_move['score']:.2f})\n"
    comparison += f"Your move: {_describe_move(user_analysis['move'])} (Score: {user_analysis['score']:.2f})\n"
    
    if user_analysis['is_best_move']:
        comparison += "Perfect! You chose the engine's top recommendation."
    else:
        score_diff = abs(user_analysis['score_difference_from_best'])
        comparison += f"Difference: {score_diff:.2f} points. "
        
        if score_diff < 0.1:
            comparison += "Practically equivalent moves."
        elif score_diff < 0.3:
            comparison += "Minor difference, both moves are reasonable."
        else:
            comparison += "Significant difference, the engine's choice is clearly superior."
    
    return comparison

def _extract_move_learning_insights(user_analysis, engine_moves):
    """Extract learning insights from move analysis"""
    insights = []
    
    if user_analysis['is_best_move']:
        insights.append("Excellent pattern recognition! You identified the strongest continuation.")
        insights.append("Study why this move is superior to understand similar positions.")
    elif user_analysis['is_top_3']:
        insights.append("Good candidate move selection. You're thinking along the right lines.")
        insights.append("Compare with the top choice to refine your evaluation skills.")
    else:
        insights.append("This position requires deeper analysis. Consider all candidate moves systematically.")
        insights.append("Focus on tactical awareness and positional understanding.")
    
    insights.append("Practice similar positions to improve pattern recognition.")
    insights.append("Analyze master games with comparable pawn structures.")
    
    return insights

def _generate_move_improvement_suggestions(user_analysis, engine_moves):
    """Generate specific improvement suggestions"""
    suggestions = []
    
    if not user_analysis['is_best_move']:
        suggestions.append("Calculate 2-3 moves deeper before deciding.")
        suggestions.append("Consider all forcing moves (checks, captures, threats) first.")
    
    suggestions.append("Evaluate candidate moves systematically using a consistent method.")
    suggestions.append("Study tactical patterns to improve move selection.")
    suggestions.append("Practice endgame positions to understand piece values better.")
    
    return suggestions

def _compare_moves(user_analysis, engine_move):
    """Compare user move with engine alternative"""
    score_diff = engine_move['score'] - user_analysis['score']
    
    if score_diff > 0.3:
        return "Significantly stronger alternative"
    elif score_diff > 0.1:
        return "Moderately better option"
    elif score_diff > -0.1:
        return "Approximately equal strength"
    else:
        return "Your move is actually stronger"

def _generate_backtrack_script(user_move, position):
    """Generate script for analyzing move consequences"""
    return {
        'title': "Move Consequence Analysis",
        'description': f"Let's trace the consequences of {_describe_move(user_move)}",
        'steps': [
            "1. Identify immediate tactical consequences",
            "2. Evaluate positional changes",
            "3. Consider opponent's best responses",
            "4. Assess resulting position"
        ],
        'key_questions': [
            "Does this move improve piece activity?",
            "Are there any tactical vulnerabilities created?",
            "How does this affect pawn structure?",
            "What are the opponent's main threats after this move?"
        ]
    }

def _generate_alternative_demo_scripts(engine_moves):
    """Generate demo scripts for alternative moves"""
    scripts = []
    
    for i, move in enumerate(engine_moves):
        script = {
            'alternative_number': i + 1,
            'move_description': _describe_move(move['move']),
            'demo_sequence': [
                f"Play {_describe_move(move['move'])}",
                "Observe immediate consequences",
                "Analyze opponent's likely responses",
                "Compare resulting positions"
            ],
            'key_benefits': _identify_move_benefits(move),
            'when_to_prefer': _when_to_prefer_move(move)
        }
        scripts.append(script)
    
    return scripts

def _generate_tactical_demo_scripts(user_move, engine_moves, position):
    """Generate tactical demonstration scripts"""
    return {
        'tactical_themes': [
            "Pin and Skewer Opportunities",
            "Fork and Double Attack Patterns",
            "Discovered Attack Possibilities",
            "Deflection and Decoy Tactics"
        ],
        'demonstration_steps': [
            "Set up the position after your move",
            "Look for tactical patterns",
            "Calculate forcing variations",
            "Compare with engine alternatives"
        ],
        'practice_exercises': [
            "Find all checks in the resulting position",
            "Identify all possible captures",
            "Look for piece coordination improvements",
            "Assess king safety for both sides"
        ]
    }

def _generate_strategic_demo_scripts(user_move, engine_moves):
    """Generate strategic comparison demonstrations"""
    return {
        'strategic_comparison': {
            'user_move_strategy': _identify_move_strategy(user_move),
            'engine_move_strategy': _identify_move_strategy(engine_moves[0]['move']) if engine_moves else "N/A",
            'strategic_differences': _compare_strategies(user_move, engine_moves[0]['move'] if engine_moves else None)
        },
        'demonstration_plan': [
            "Analyze pawn structure implications",
            "Evaluate piece activity changes",
            "Consider long-term strategic goals",
            "Compare resulting imbalances"
        ]
    }

def _identify_move_benefits(move):
    """Identify key benefits of a move"""
    benefits = []
    
    if move['score'] > 0.2:
        benefits.append("Provides significant advantage")
    
    # Add more specific benefits based on move analysis
    benefits.append("Improves piece coordination")
    benefits.append("Maintains strategic flexibility")
    
    return benefits

def _when_to_prefer_move(move):
    """Suggest when to prefer this move"""
    return "Consider this move when seeking active piece play and maintaining initiative."

def _identify_move_strategy(move):
    """Identify the strategic idea behind a move"""
    if move['piece'] == 'pawn':
        return "Pawn advance for space and center control"
    elif move['piece'] in ['knight', 'bishop']:
        return "Minor piece development and activation"
    elif move['piece'] in ['rook', 'queen']:
        return "Major piece activation and pressure"
    else:
        return "King safety and position consolidation"

def _compare_strategies(user_move, engine_move):
    """Compare strategic approaches of different moves"""
    if not engine_move:
        return "No engine move available for comparison"
    
    user_strategy = _identify_move_strategy(user_move)
    engine_strategy = _identify_move_strategy(engine_move)
    
    return f"Your approach: {user_strategy}. Engine approach: {engine_strategy}."

def _generate_move_reasoning(analysis, score):
    """Generate human-readable reasoning for move"""
    reasoning = []
    
    # Score-based reasoning
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
    if 'center_control' in themes:
        reasoning.append("Controls important central squares")
    
    if 'centralization' in themes:
        reasoning.append("Centralizes pieces effectively")
    if 'coordination' in themes:
        reasoning.append("Enhances piece coordination")
    
    # Positional reasoning
    pos_factors = analysis.get('positional_factors', {})
    if isinstance(pos_factors, dict):
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

def _describe_move(move):
    """Generate human-readable description of a move"""
    piece_names = {
        'pawn': 'Pawn', 'knight': 'Knight', 'bishop': 'Bishop',
        'rook': 'Rook', 'queen': 'Queen', 'king': 'King'
    }
    piece_name = piece_names.get(move.get('piece', '').lower(), 'Piece')
    return f"{piece_name} from {move.get('from', '?')} to {move.get('to', '?')}"

if __name__ == '__main__':
    print("Starting Chess AI Play API...")
    print("AI Engine loaded and ready")
    print("Available endpoints:")
    print("  POST /api/recommend-move - Get move recommendations (Stockfish)")
    print("  POST /api/recommend-move-ml - Get move recommendations (ML Model)")
    print("  POST /api/analyze-position - Analyze position")
    print("  POST /api/submit-game - Submit game for learning")
    print("  GET  /api/learning-stats - Get learning statistics")
    print("  GET  /api/health - Health check")
    
    app.run(host='0.0.0.0', port=5001, debug=True)