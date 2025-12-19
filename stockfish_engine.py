#!/usr/bin/env python3
"""
Stockfish Chess Engine Wrapper
"""

import chess
import chess.engine
import os

class StockfishEngine:
    def __init__(self, stockfish_path="./stockfish/stockfish-macos-m1-apple-silicon"):
        self.stockfish_path = stockfish_path
        self.engine = None
        self.initialize_engine()
    
    def initialize_engine(self):
        """Initialize Stockfish engine with timeout"""
        try:
            if os.path.exists(self.stockfish_path):
                # Add timeout to prevent hanging
                import subprocess
                self.engine = chess.engine.SimpleEngine.popen_uci(
                    self.stockfish_path,
                    timeout=10
                )
                print(f"Stockfish engine loaded from {self.stockfish_path}")
            else:
                print(f"Stockfish not found at {self.stockfish_path}")
                self.engine = None
        except Exception as e:
            print(f"Failed to load Stockfish: {e}")
            self.engine = None
    
    def position_to_board(self, position):
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
    
    def get_best_move(self, position, current_player='white', time_limit=1.0):
        """Get best move from Stockfish"""
        if not self.engine:
            return None
        
        try:
            board = self.position_to_board(position)
            board.turn = current_player == 'white'
            
            result = self.engine.play(board, chess.engine.Limit(time=time_limit))
            
            if result.move:
                piece = board.piece_at(result.move.from_square)
                return {
                    'from': chess.square_name(result.move.from_square),
                    'to': chess.square_name(result.move.to_square),
                    'piece': piece.symbol().lower() if piece else 'p',
                    'color': current_player
                }
        except Exception as e:
            print(f"Stockfish move error: {e}")
        
        return None
    
    def evaluate_position(self, position, current_player='white', time_limit=0.1):
        """Get position evaluation from Stockfish"""
        if not self.engine:
            return 0.0
        
        try:
            board = self.position_to_board(position)
            board.turn = current_player == 'white'
            
            info = self.engine.analyse(board, chess.engine.Limit(time=time_limit))
            score = info["score"].relative
            
            # Convert to float between -1 and 1
            if score.is_mate():
                return 1.0 if score.mate() > 0 else -1.0
            else:
                # Convert centipawn to normalized score
                cp_score = score.score() / 100.0  # Convert to pawn units
                return max(-1.0, min(1.0, cp_score / 10.0))  # Normalize to -1,1
                
        except Exception as e:
            print(f"Stockfish evaluation error: {e}")
        
        return 0.0
    
    def get_move_analysis(self, position, current_player='white', num_moves=3):
        """Get multiple move analysis from Stockfish"""
        if not self.engine:
            return []
        
        try:
            board = self.position_to_board(position)
            board.turn = current_player == 'white'
            
            # Get multiple variations
            info = self.engine.analyse(board, chess.engine.Limit(time=0.5), multipv=num_moves)
            
            move_scores = []
            for i, analysis in enumerate(info):
                if 'pv' in analysis and analysis['pv']:
                    move = analysis['pv'][0]
                    score_info = analysis['score'].relative
                    
                    if score_info.is_mate():
                        score = 1.0 if score_info.mate() > 0 else -1.0
                    else:
                        score = max(-1.0, min(1.0, score_info.score() / 1000.0))
                    
                    piece = board.piece_at(move.from_square)
                    move_scores.append({
                        'move': {
                            'from': chess.square_name(move.from_square),
                            'to': chess.square_name(move.to_square),
                            'piece': piece.symbol().lower() if piece else 'p',
                            'color': current_player
                        },
                        'score': float(score),
                        'analysis': {
                            'confidence': 'high',
                            'tactical_elements': [],
                            'strategic_themes': ['stockfish_analysis'],
                            'positional_factors': []
                        }
                    })
            
            return move_scores
            
        except Exception as e:
            print(f"Stockfish analysis error: {e}")
        
        return []
    
    def quit(self):
        """Close Stockfish engine"""
        if self.engine:
            self.engine.quit()
            self.engine = None