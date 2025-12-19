#!/usr/bin/env python3
"""
Stockfish-Only Chess API - NO Neural Engine
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import chess
import chess.engine
import os
import sys
from datetime import datetime

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5500", "http://127.0.0.1:5500"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Initialize Stockfish directly - NO fallback
try:
    stockfish_path = "./stockfish/stockfish-macos-m1-apple-silicon"
    if not os.path.exists(stockfish_path):
        print(f"❌ ERROR: Stockfish not found at {stockfish_path}")
        sys.exit(1)
    
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    print("✅ STOCKFISH ENGINE LOADED - NO NEURAL ENGINE!")
    
except Exception as e:
    print(f"❌ FAILED TO LOAD STOCKFISH: {e}")
    sys.exit(1)

def position_to_board(position):
    """Convert API position to chess.Board"""
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

@app.route('/api/recommend-move', methods=['POST'])
def recommend_move():
    """Get comprehensive analysis from STOCKFISH"""
    print("\n" + "="*50)
    print("🔥 API CALL RECEIVED - STOCKFISH ANALYSIS")
    print("="*50)
    
    try:
        data = request.get_json()
        print(f"📥 Request data: {data}")
        
        position = data['position']
        current_player = data.get('current_player', 'white')
        num_alternatives = data.get('num_alternatives', 3)
        
        board = position_to_board(position)
        board.turn = current_player == 'white'
        
        print(f"♟️  Board FEN: {board.fen()}")
        print(f"🔄 Turn: {'White' if board.turn else 'Black'}")
        print(f"⚡ Legal moves: {len(list(board.legal_moves))}")
        
        # Get comprehensive analysis
        print("🤖 Getting Stockfish analysis...")
        analysis_info = engine.analyse(board, chess.engine.Limit(time=2.0), multipv=min(num_alternatives + 1, 5))
        
        # Get position evaluation
        main_eval = analysis_info[0]['score'].relative
        
        moves_analysis = []
        for i, info in enumerate(analysis_info):
            if 'pv' in info and info['pv']:
                move = info['pv'][0]
                score = info['score'].relative
                depth = info.get('depth', 0)
                nodes = info.get('nodes', 0)
                pv_line = info['pv'][:5]  # First 5 moves of principal variation
                
                # Convert score to centipawns
                if score.is_mate():
                    cp_score = 32000 if score.mate() > 0 else -32000
                    score_text = f"Mate in {abs(score.mate())}"
                else:
                    cp_score = score.score()
                    score_text = f"{cp_score/100:.2f}" if cp_score else "0.00"
                
                piece = board.piece_at(move.from_square)
                # Get SAN notation safely
                try:
                    san_notation = board.san(move)
                except:
                    san_notation = f"{chess.square_name(move.from_square)}{chess.square_name(move.to_square)}"
                
                move_data = {
                    'from': chess.square_name(move.from_square),
                    'to': chess.square_name(move.to_square),
                    'piece': piece.symbol().lower() if piece else 'p',
                    'color': current_player,
                    'san': san_notation
                }
                
                # Analyze move characteristics
                board_copy = board.copy()
                board_copy.push(move)
                
                # Get detailed depth analysis
                depth_analysis = get_depth_analysis(board, move, engine)
                score_breakdown = get_score_breakdown(info, board, move)
                
                move_analysis = {
                    'move': move_data,
                    'score': cp_score,
                    'score_text': score_text,
                    'depth': depth,
                    'nodes': nodes,
                    'principal_variation': get_safe_pv_notation(board, pv_line),
                    'is_capture': board.is_capture(move),
                    'is_check': board_copy.is_check(),
                    'is_checkmate': board_copy.is_checkmate(),
                    'gives_check': board_copy.is_check(),
                    'analysis': get_move_characteristics(board, move, board_copy),
                    'depth_analysis': depth_analysis,
                    'score_breakdown': score_breakdown
                }
                
                moves_analysis.append(move_analysis)
        
        best_move = moves_analysis[0] if moves_analysis else None
        alternatives = moves_analysis[1:] if len(moves_analysis) > 1 else []
        
        # Position analysis
        position_analysis = get_position_analysis(board, main_eval)
        
        print(f"✅ Best move: {best_move['move']['san'] if best_move else 'None'}")
        print(f"📊 Evaluation: {best_move['score_text'] if best_move else 'N/A'}")
        
        if best_move:
            response = {
                'best_move': {
                    'move': best_move['move'],
                    'score': best_move['score'],
                    'score_text': best_move['score_text'],
                    'depth': best_move['depth'],
                    'nodes': best_move['nodes'],
                    'principal_variation': best_move['principal_variation'],
                    'confidence': 'high',
                    'analysis': best_move['analysis'],
                    'reasoning': generate_move_reasoning(best_move, board)
                },
                'alternatives': [
                    {
                        'move': alt['move'],
                        'score': alt['score'],
                        'score_text': alt['score_text'],
                        'depth': alt['depth'],
                        'principal_variation': alt['principal_variation'],
                        'analysis': alt['analysis'],
                        'score_difference': best_move['score'] - alt['score']
                    } for alt in alternatives
                ],
                'position_analysis': position_analysis,
                'engine_info': {
                    'name': 'Stockfish 17.1',
                    'depth_analyzed': best_move['depth'],
                    'nodes_searched': best_move['nodes'],
                    'analysis_time': '2.0s'
                },
                'timestamp': datetime.now().isoformat(),
                'engine': 'STOCKFISH_COMPREHENSIVE'
            }
            
            print("✅ COMPREHENSIVE ANALYSIS SENT")
            print("="*50)
            return jsonify(response)
        
        print("❌ No legal moves found")
        return jsonify({'error': 'No legal moves'}), 400
        
    except Exception as e:
        print(f"💥 ERROR: {e}")
        print("="*50)
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-position', methods=['POST'])
def analyze_position():
    """Analyze user's move with Stockfish"""
    try:
        data = request.get_json()
        
        if not data or 'position' not in data or 'user_move' not in data:
            return jsonify({'error': 'Missing position or user_move data'}), 400
        
        position = data['position']
        user_move = data['user_move']
        current_player = data.get('current_player', 'white')
        include_demo = data.get('include_demo', False)
        
        board = position_to_board(position)
        board.turn = current_player == 'white'
        
        # Get engine recommendations
        analysis_info = engine.analyse(board, chess.engine.Limit(time=1.0), multipv=5)
        
        # Find user move in legal moves
        user_chess_move = None
        try:
            from_square = chess.parse_square(user_move['from'])
            to_square = chess.parse_square(user_move['to'])
            user_chess_move = chess.Move(from_square, to_square)
            
            # Handle promotion
            if user_move.get('promotion'):
                promotion_map = {'q': chess.QUEEN, 'r': chess.ROOK, 'b': chess.BISHOP, 'n': chess.KNIGHT}
                user_chess_move.promotion = promotion_map.get(user_move['promotion'].lower(), chess.QUEEN)
        except:
            return jsonify({'error': 'Invalid move format'}), 400
        
        if user_chess_move not in board.legal_moves:
            return jsonify({'error': 'Illegal move'}), 400
        
        # Analyze user move
        board_after = board.copy()
        board_after.push(user_chess_move)
        user_analysis = engine.analyse(board_after, chess.engine.Limit(time=1.0))
        
        user_score = user_analysis['score'].relative
        if user_score.is_mate():
            user_cp_score = 32000 if user_score.mate() > 0 else -32000
        else:
            user_cp_score = user_score.score()
        
        # Find user move rank
        user_rank = None
        for i, info in enumerate(analysis_info):
            if 'pv' in info and info['pv'] and info['pv'][0] == user_chess_move:
                user_rank = i + 1
                break
        
        if user_rank is None:
            user_rank = len(analysis_info) + 1
        
        # Get best move score
        best_score = analysis_info[0]['score'].relative
        if best_score.is_mate():
            best_cp_score = 32000 if best_score.mate() > 0 else -32000
        else:
            best_cp_score = best_score.score()
        
        user_move_evaluation = {
            'move': user_move,
            'score': user_cp_score / 100.0,
            'rank': user_rank,
            'is_best_move': user_rank == 1,
            'is_top_3': user_rank <= 3,
            'score_difference_from_best': (user_cp_score - best_cp_score) / 100.0
        }
        
        # Generate comprehensive analysis
        comprehensive_analysis = {
            'move_quality_assessment': _assess_move_quality(user_move_evaluation),
            'strategic_analysis': _analyze_move_strategy(user_move, board),
            'tactical_analysis': _analyze_move_tactics(user_move, board),
            'positional_consequences': _analyze_positional_impact(user_move, board),
            'alternative_comparison': _generate_alternative_analysis(user_move_evaluation, analysis_info),
            'learning_insights': _extract_move_learning_insights(user_move_evaluation),
            'improvement_recommendations': _generate_move_improvement_suggestions(user_move_evaluation)
        }
        
        # Get engine alternatives
        engine_alternatives = []
        for i, info in enumerate(analysis_info[:3]):
            if 'pv' in info and info['pv']:
                move = info['pv'][0]
                score = info['score'].relative
                
                if score.is_mate():
                    cp_score = 32000 if score.mate() > 0 else -32000
                else:
                    cp_score = score.score()
                
                piece = board.piece_at(move.from_square)
                engine_alternatives.append({
                    'move': {
                        'from': chess.square_name(move.from_square),
                        'to': chess.square_name(move.to_square),
                        'piece': piece.symbol().lower() if piece else 'p'
                    },
                    'score': cp_score / 100.0,
                    'rank': i + 1
                })
        
        response = {
            'user_move_evaluation': user_move_evaluation,
            'comprehensive_analysis': comprehensive_analysis,
            'engine_alternatives': engine_alternatives,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

def _analyze_move_strategy(user_move, board):
    """Analyze strategic aspects of the move"""
    analysis = []
    
    if user_move['piece'] in ['knight', 'bishop']:
        analysis.append("Development Move: Activating minor pieces toward the center is generally good strategy.")
    
    center_squares = ['d4', 'd5', 'e4', 'e5']
    if user_move['to'] in center_squares or user_move['piece'] == 'pawn':
        analysis.append("Center Control: This move influences central squares, which is strategically important.")
    
    if user_move['piece'] == 'king':
        analysis.append("King Safety: King moves should be carefully considered, especially in the opening and middlegame.")
    
    return analysis

def _analyze_move_tactics(user_move, board):
    """Analyze tactical aspects of the move"""
    tactical_elements = []
    
    target_square = chess.parse_square(user_move['to'])
    if board.piece_at(target_square):
        captured_piece = board.piece_at(target_square)
        tactical_elements.append(f"Capture: This move captures the opponent's {captured_piece.symbol().upper()}.")
    
    if user_move['piece'] in ['queen', 'rook', 'bishop', 'knight']:
        tactical_elements.append("Potential Threats: This piece move may create tactical opportunities.")
    
    return tactical_elements

def _analyze_positional_impact(user_move, board):
    """Analyze positional consequences of the move"""
    consequences = []
    
    if user_move['piece'] == 'pawn':
        consequences.append("Pawn Structure: Pawn moves are permanent and affect long-term pawn structure.")
    
    consequences.append("Piece Coordination: Consider how this move affects coordination with other pieces.")
    consequences.append("Space Control: Evaluate how this move affects your control of key squares.")
    
    return consequences

def _generate_alternative_analysis(user_analysis, engine_moves):
    """Generate analysis comparing user move with alternatives"""
    if not engine_moves:
        return "No engine alternatives available for comparison."
    
    best_move = engine_moves[0]['pv'][0] if 'pv' in engine_moves[0] and engine_moves[0]['pv'] else None
    if not best_move:
        return "Engine analysis unavailable."
    
    best_score = engine_moves[0]['score'].relative
    if best_score.is_mate():
        best_cp = 32000 if best_score.mate() > 0 else -32000
    else:
        best_cp = best_score.score()
    
    comparison = f"Engine's top choice: {chess.square_name(best_move.from_square)}-{chess.square_name(best_move.to_square)} (Score: {best_cp/100:.2f})\n"
    comparison += f"Your move: {user_analysis['move']['from']}-{user_analysis['move']['to']} (Score: {user_analysis['score']:.2f})\n"
    
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

def _extract_move_learning_insights(user_analysis):
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

def _generate_move_improvement_suggestions(user_analysis):
    """Generate specific improvement suggestions"""
    suggestions = []
    
    if not user_analysis['is_best_move']:
        suggestions.append("Calculate 2-3 moves deeper before deciding.")
        suggestions.append("Consider all forcing moves (checks, captures, threats) first.")
    
    suggestions.append("Evaluate candidate moves systematically using a consistent method.")
    suggestions.append("Study tactical patterns to improve move selection.")
    suggestions.append("Practice endgame positions to understand piece values better.")
    
    return suggestions

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'engine': 'STOCKFISH_ONLY',
        'timestamp': datetime.now().isoformat()
    })

def get_move_characteristics(board, move, board_after):
    """Analyze move characteristics"""
    characteristics = {
        'tactical_elements': [],
        'strategic_themes': [],
        'positional_factors': {}
    }
    
    # Tactical elements
    if board.is_capture(move):
        captured_piece = board.piece_at(move.to_square)
        characteristics['tactical_elements'].append(f"Captures {captured_piece.symbol().upper()}")
    
    if board_after.is_check():
        characteristics['tactical_elements'].append("Gives check")
    
    if board_after.is_checkmate():
        characteristics['tactical_elements'].append("Checkmate!")
    
    # Strategic themes
    piece = board.piece_at(move.from_square)
    if piece:
        # Development
        if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            from_rank = chess.square_rank(move.from_square)
            back_rank = 0 if piece.color == chess.WHITE else 7
            if from_rank == back_rank:
                characteristics['strategic_themes'].append("Piece development")
        
        # Center control
        if move.to_square in [chess.D4, chess.D5, chess.E4, chess.E5]:
            characteristics['strategic_themes'].append("Center control")
        
        # Castling
        if piece.piece_type == chess.KING and abs(chess.square_file(move.to_square) - chess.square_file(move.from_square)) == 2:
            side = "Kingside" if chess.square_file(move.to_square) > chess.square_file(move.from_square) else "Queenside"
            characteristics['strategic_themes'].append(f"{side} castling")
    
    return characteristics

def get_position_analysis(board, evaluation):
    """Analyze current position"""
    analysis = {
        'material_balance': calculate_material_balance(board),
        'king_safety': analyze_king_safety(board),
        'center_control': analyze_center_control(board),
        'piece_activity': analyze_piece_activity(board),
        'pawn_structure': analyze_pawn_structure(board),
        'evaluation_summary': get_evaluation_summary(evaluation)
    }
    return analysis

def calculate_material_balance(board):
    """Calculate material balance"""
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    white_material = sum(piece_values.get(piece.piece_type, 0) for piece in board.piece_map().values() if piece.color == chess.WHITE)
    black_material = sum(piece_values.get(piece.piece_type, 0) for piece in board.piece_map().values() if piece.color == chess.BLACK)
    
    return {
        'white': white_material,
        'black': black_material,
        'difference': white_material - black_material
    }

def analyze_king_safety(board):
    """Analyze king safety for both sides"""
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    
    return {
        'white_king_safety': 'safe' if white_king and not board.is_attacked_by(chess.BLACK, white_king) else 'exposed',
        'black_king_safety': 'safe' if black_king and not board.is_attacked_by(chess.WHITE, black_king) else 'exposed',
        'white_in_check': board.is_check() and board.turn == chess.WHITE,
        'black_in_check': board.is_check() and board.turn == chess.BLACK
    }

def analyze_center_control(board):
    """Analyze center control"""
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    white_control = sum(1 for sq in center_squares if board.is_attacked_by(chess.WHITE, sq))
    black_control = sum(1 for sq in center_squares if board.is_attacked_by(chess.BLACK, sq))
    
    return {
        'white_control': white_control,
        'black_control': black_control,
        'advantage': 'white' if white_control > black_control else 'black' if black_control > white_control else 'equal'
    }

def analyze_piece_activity(board):
    """Analyze piece activity"""
    white_pieces = len([p for p in board.piece_map().values() if p.color == chess.WHITE and p.piece_type != chess.KING])
    black_pieces = len([p for p in board.piece_map().values() if p.color == chess.BLACK and p.piece_type != chess.KING])
    
    return {
        'white_pieces': white_pieces,
        'black_pieces': black_pieces,
        'mobility_advantage': 'white' if len(list(board.legal_moves)) > 20 else 'limited'
    }

def analyze_pawn_structure(board):
    """Analyze pawn structure"""
    white_pawns = len([p for p in board.piece_map().values() if p.color == chess.WHITE and p.piece_type == chess.PAWN])
    black_pawns = len([p for p in board.piece_map().values() if p.color == chess.BLACK and p.piece_type == chess.PAWN])
    
    return {
        'white_pawns': white_pawns,
        'black_pawns': black_pawns,
        'structure': 'normal'  # Simplified analysis
    }

def get_evaluation_summary(evaluation):
    """Convert evaluation to human-readable summary"""
    if evaluation.is_mate():
        mate_in = evaluation.mate()
        return f"Mate in {abs(mate_in)} for {'White' if mate_in > 0 else 'Black'}"
    
    score = evaluation.score() / 100.0
    if score > 2:
        return "White has winning advantage"
    elif score > 0.5:
        return "White has significant advantage"
    elif score > 0.1:
        return "White has slight advantage"
    elif score > -0.1:
        return "Position is balanced"
    elif score > -0.5:
        return "Black has slight advantage"
    elif score > -2:
        return "Black has significant advantage"
    else:
        return "Black has winning advantage"

def get_depth_analysis(board, move, engine):
    """Analyze move at different depths"""
    depth_results = []
    
    for depth in [5, 10, 15, 20]:
        try:
            board_copy = board.copy()
            board_copy.push(move)
            
            info = engine.analyse(board_copy, chess.engine.Limit(depth=depth, time=0.5))
            score = info['score'].relative
            
            if score.is_mate():
                eval_text = f"Mate in {abs(score.mate())}"
                cp_value = 32000 if score.mate() > 0 else -32000
            else:
                cp_value = score.score()
                eval_text = f"{cp_value/100:.2f}"
            
            depth_results.append({
                'depth': depth,
                'evaluation': cp_value,
                'evaluation_text': eval_text,
                'nodes': info.get('nodes', 0),
                'time_ms': info.get('time', 0) * 1000 if 'time' in info else 0,
                'pv_first_move': get_safe_san(board_copy, info['pv'][0]) if 'pv' in info and info['pv'] else None
            })
            
        except Exception as e:
            depth_results.append({
                'depth': depth,
                'evaluation': 0,
                'evaluation_text': "Error",
                'error': str(e)
            })
    
    return depth_results

def get_score_breakdown(analysis_info, board, move):
    """Break down how the score is calculated"""
    breakdown = {
        'total_score': 0,
        'components': [],
        'calculation_method': 'Stockfish Evaluation Function',
        'factors_considered': []
    }
    
    # Extract score
    score = analysis_info['score'].relative
    if score.is_mate():
        breakdown['total_score'] = 32000 if score.mate() > 0 else -32000
        breakdown['components'].append({
            'factor': 'Checkmate',
            'value': breakdown['total_score'],
            'description': f"Forced mate in {abs(score.mate())} moves"
        })
    else:
        breakdown['total_score'] = score.score()
    
    # Analyze position factors that contribute to score
    board_copy = board.copy()
    board_copy.push(move)
    
    # Material analysis
    material_balance = calculate_material_balance(board_copy)
    material_diff = material_balance['difference'] * 100  # Convert to centipawns
    
    breakdown['components'].extend([
        {
            'factor': 'Material Balance',
            'value': material_diff,
            'description': f"White: {material_balance['white']} points, Black: {material_balance['black']} points"
        },
        {
            'factor': 'Positional Evaluation',
            'value': breakdown['total_score'] - material_diff,
            'description': 'King safety, piece activity, pawn structure, etc.'
        }
    ])
    
    # Factors considered by Stockfish
    breakdown['factors_considered'] = [
        'Material balance (piece values)',
        'King safety and castling rights',
        'Piece mobility and activity',
        'Pawn structure and weaknesses',
        'Center control and space',
        'Tactical threats and pins',
        'Endgame knowledge',
        'Opening book knowledge'
    ]
    
    # Move-specific factors
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        piece_values = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900}
        capture_value = piece_values.get(captured.piece_type, 0)
        breakdown['components'].append({
            'factor': 'Capture Bonus',
            'value': capture_value,
            'description': f"Captures {captured.symbol().upper()} worth {capture_value} centipawns"
        })
    
    if board_copy.is_check():
        breakdown['components'].append({
            'factor': 'Check Bonus',
            'value': 50,
            'description': 'Gives check, forcing opponent response'
        })
    
    return breakdown

def get_safe_san(board, move):
    """Get SAN notation safely"""
    try:
        return board.san(move)
    except:
        return f"{chess.square_name(move.from_square)}{chess.square_name(move.to_square)}"

def get_safe_pv_notation(board, pv_line):
    """Get principal variation notation safely"""
    pv_notation = []
    board_copy = board.copy()
    
    for move in pv_line:
        try:
            san = board_copy.san(move)
            pv_notation.append(san)
            board_copy.push(move)
        except:
            # If move is illegal, stop here
            break
    
    return pv_notation

def generate_move_reasoning(move_analysis, board):
    """Generate human-readable reasoning"""
    reasoning = []
    
    # Score-based reasoning
    score = move_analysis['score']
    if abs(score) > 1000:
        reasoning.append("Decisive advantage")
    elif abs(score) > 300:
        reasoning.append("Significant positional advantage")
    elif abs(score) > 100:
        reasoning.append("Clear advantage")
    else:
        reasoning.append("Maintains balance")
    
    # Add score breakdown reasoning
    if 'score_breakdown' in move_analysis:
        for component in move_analysis['score_breakdown']['components']:
            if abs(component['value']) > 50:
                reasoning.append(component['description'])
    
    # Add tactical elements
    reasoning.extend(move_analysis['analysis']['tactical_elements'])
    reasoning.extend(move_analysis['analysis']['strategic_themes'])
    
    return reasoning

if __name__ == '__main__':
    print("🚀 STARTING COMPREHENSIVE STOCKFISH API ON PORT 5001")
    print("🚀 FULL ANALYSIS MODE - DETAILED STOCKFISH INSIGHTS!")
    print("🔍 LOGGING ENABLED - WATCH FOR API CALLS")
    print("🌐 CORS ENABLED FOR ALL ORIGINS")
    print("="*50)
    app.run(host='0.0.0.0', port=5001, debug=False)