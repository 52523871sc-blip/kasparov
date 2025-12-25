import chess

class OpeningAnalyzer:
    @staticmethod
    def get_opening_info(board):
        """Analyze opening information"""
        return {
            'opening_name': OpeningAnalyzer._classify_opening(board),
            'opening_principles': OpeningAnalyzer._check_opening_principles(board),
            'tempo_count': OpeningAnalyzer._calculate_tempo_advantage(board),
            'development_score': OpeningAnalyzer._calculate_development_score(board),
            'opening_mistakes': OpeningAnalyzer._identify_opening_mistakes(board)
        }
    
    @staticmethod
    def _classify_opening(board):
        """Classify the opening based on moves played"""
        # Simple opening classification based on first few moves
        move_history = []
        temp_board = chess.Board()
        
        # This is simplified - in practice you'd need full move history
        if board.fullmove_number <= 10:
            # Check for common opening patterns
            if board.piece_at(chess.E4) and board.piece_at(chess.E4).piece_type == chess.PAWN:
                if board.piece_at(chess.E5) and board.piece_at(chess.E5).piece_type == chess.PAWN:
                    return "King's Pawn Game (1.e4 e5)"
                else:
                    return "King's Pawn Opening (1.e4)"
            elif board.piece_at(chess.D4) and board.piece_at(chess.D4).piece_type == chess.PAWN:
                return "Queen's Pawn Opening (1.d4)"
            elif board.piece_at(chess.F3) and board.piece_at(chess.F3).piece_type == chess.KNIGHT:
                return "Reti Opening (1.Nf3)"
            elif board.piece_at(chess.C4) and board.piece_at(chess.C4).piece_type == chess.PAWN:
                return "English Opening (1.c4)"
        
        return "Unknown Opening"
    
    @staticmethod
    def _check_opening_principles(board):
        """Check adherence to opening principles"""
        principles = {
            'center_control': OpeningAnalyzer._check_center_control(board),
            'piece_development': OpeningAnalyzer._check_piece_development(board),
            'king_safety': OpeningAnalyzer._check_king_safety(board),
            'avoid_moving_same_piece_twice': OpeningAnalyzer._check_piece_moves(board),
            'castle_early': OpeningAnalyzer._check_castling_status(board)
        }
        
        return principles
    
    @staticmethod
    def _check_center_control(board):
        """Check center control in opening"""
        center_pawns = {'white': 0, 'black': 0}
        
        for square in [chess.D4, chess.D5, chess.E4, chess.E5]:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                color = 'white' if piece.color == chess.WHITE else 'black'
                center_pawns[color] += 1
        
        return {
            'white_center_pawns': center_pawns['white'],
            'black_center_pawns': center_pawns['black'],
            'evaluation': 'good' if center_pawns['white'] + center_pawns['black'] >= 2 else 'needs_improvement'
        }
    
    @staticmethod
    def _check_piece_development(board):
        """Check piece development status"""
        development = {'white': {'knights': 0, 'bishops': 0}, 'black': {'knights': 0, 'bishops': 0}}
        
        for color in [chess.WHITE, chess.BLACK]:
            color_name = 'white' if color == chess.WHITE else 'black'
            back_rank = 0 if color == chess.WHITE else 7
            
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color == color:
                    rank = chess.square_rank(square)
                    if piece.piece_type == chess.KNIGHT and rank != back_rank:
                        development[color_name]['knights'] += 1
                    elif piece.piece_type == chess.BISHOP and rank != back_rank:
                        development[color_name]['bishops'] += 1
        
        return development
    
    @staticmethod
    def _check_king_safety(board):
        """Check king safety in opening"""
        safety = {'white': 'unknown', 'black': 'unknown'}
        
        for color in [chess.WHITE, chess.BLACK]:
            color_name = 'white' if color == chess.WHITE else 'black'
            
            # Check if castled
            if board.has_castling_rights(color):
                safety[color_name] = 'can_castle'
            else:
                king_square = board.king(color)
                if king_square:
                    # Check if king is on starting square
                    starting_square = chess.E1 if color == chess.WHITE else chess.E8
                    if king_square == starting_square:
                        safety[color_name] = 'not_castled'
                    else:
                        safety[color_name] = 'moved_early'
        
        return safety
    
    @staticmethod
    def _check_piece_moves(board):
        """Check for repeated piece moves (simplified)"""
        # This would require move history - simplified implementation
        return {'evaluation': 'unknown', 'repeated_moves': []}
    
    @staticmethod
    def _check_castling_status(board):
        """Check castling status"""
        return {
            'white_can_castle': board.has_castling_rights(chess.WHITE),
            'black_can_castle': board.has_castling_rights(chess.BLACK),
            'white_castled': not board.has_castling_rights(chess.WHITE) and board.king(chess.WHITE) != chess.E1,
            'black_castled': not board.has_castling_rights(chess.BLACK) and board.king(chess.BLACK) != chess.E8
        }
    
    @staticmethod
    def _calculate_tempo_advantage(board):
        """Calculate tempo advantage (simplified)"""
        # Simplified tempo calculation based on development
        white_tempo = 0
        black_tempo = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                rank = chess.square_rank(square)
                if piece.color == chess.WHITE and rank > 0:
                    white_tempo += 1
                elif piece.color == chess.BLACK and rank < 7:
                    black_tempo += 1
        
        return {'white': white_tempo, 'black': black_tempo, 'advantage': white_tempo - black_tempo}
    
    @staticmethod
    def _calculate_development_score(board):
        """Calculate development score"""
        score = {'white': 0, 'black': 0}
        
        # Points for developed pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                color_name = 'white' if piece.color == chess.WHITE else 'black'
                rank = chess.square_rank(square)
                
                if piece.piece_type == chess.KNIGHT:
                    back_rank = 0 if piece.color == chess.WHITE else 7
                    if rank != back_rank:
                        score[color_name] += 2
                elif piece.piece_type == chess.BISHOP:
                    back_rank = 0 if piece.color == chess.WHITE else 7
                    if rank != back_rank:
                        score[color_name] += 2
                elif piece.piece_type == chess.KING:
                    # Penalty for not castling
                    if board.has_castling_rights(piece.color):
                        score[color_name] -= 1
        
        return score
    
    @staticmethod
    def _identify_opening_mistakes(board):
        """Identify common opening mistakes"""
        mistakes = []
        
        # Check for early queen moves
        for color in [chess.WHITE, chess.BLACK]:
            queen_square = None
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.QUEEN and piece.color == color:
                    queen_square = square
                    break
            
            if queen_square:
                starting_square = chess.D1 if color == chess.WHITE else chess.D8
                if queen_square != starting_square and board.fullmove_number <= 5:
                    color_name = 'White' if color == chess.WHITE else 'Black'
                    mistakes.append(f\"{color_name} moved queen too early\")\n        
        return mistakes