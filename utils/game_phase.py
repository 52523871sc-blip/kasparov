import chess

class GamePhaseAnalyzer:
    @staticmethod
    def get_game_phase(board):
        """Determine current game phase"""
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        total_material = sum(piece_values.get(piece.piece_type, 0) for piece in board.piece_map().values())
        queens_count = len([p for p in board.piece_map().values() if p.piece_type == chess.QUEEN])
        
        if board.fullmove_number <= 10 and total_material > 60:
            phase = "opening"
        elif total_material <= 20 or queens_count == 0:
            phase = "endgame"
        else:
            phase = "middlegame"
        
        return {
            'phase': phase,
            'move_count': board.fullmove_number,
            'material_count': total_material,
            'queens_on_board': queens_count,
            'pieces_developed': GamePhaseAnalyzer._count_developed_pieces(board)
        }
    
    @staticmethod
    def _count_developed_pieces(board):
        """Count developed pieces"""
        developed = {'white': 0, 'black': 0}
        
        # Check knights and bishops off back rank
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                rank = chess.square_rank(square)
                if piece.color == chess.WHITE and rank > 0:
                    developed['white'] += 1
                elif piece.color == chess.BLACK and rank < 7:
                    developed['black'] += 1
        
        return developed