import chess

class PieceAnalyzer:
    @staticmethod
    def analyze_piece_coordination(board):
        """Analyze piece coordination and activity"""
        return {
            'piece_harmony': PieceAnalyzer._calculate_piece_harmony(board),
            'piece_mobility': PieceAnalyzer._calculate_piece_mobility(board),
            'piece_centralization': PieceAnalyzer._analyze_centralization(board),
            'piece_development': PieceAnalyzer._analyze_development(board),
            'piece_activity_score': PieceAnalyzer._calculate_activity_score(board)
        }
    
    @staticmethod
    def _calculate_piece_harmony(board):
        """Calculate how well pieces work together"""
        harmony = {'white': 0, 'black': 0}
        
        for color in [chess.WHITE, chess.BLACK]:
            color_name = 'white' if color == chess.WHITE else 'black'
            pieces = []
            
            # Collect all pieces
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color == color and piece.piece_type != chess.KING:
                    pieces.append((square, piece))
            
            # Calculate coordination score
            coordination_score = 0
            for i, (sq1, p1) in enumerate(pieces):
                for sq2, p2 in pieces[i+1:]:
                    if PieceAnalyzer._pieces_coordinate(board, sq1, p1, sq2, p2):
                        coordination_score += 1
            
            harmony[color_name] = coordination_score
        
        return harmony
    
    @staticmethod
    def _pieces_coordinate(board, sq1, piece1, sq2, piece2):
        """Check if two pieces coordinate well"""
        # Simplified: pieces coordinate if they attack same squares
        attacks1 = set(board.attacks(sq1))
        attacks2 = set(board.attacks(sq2))
        
        # Count common attack squares
        common_attacks = len(attacks1.intersection(attacks2))
        return common_attacks > 0
    
    @staticmethod
    def _calculate_piece_mobility(board):
        """Calculate piece mobility for each side"""
        mobility = {'white': 0, 'black': 0}
        
        for color in [chess.WHITE, chess.BLACK]:
            color_name = 'white' if color == chess.WHITE else 'black'
            total_mobility = 0
            
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color == color:
                    # Count legal moves for this piece
                    piece_mobility = len(list(board.attacks(square)))
                    
                    # Weight by piece value
                    piece_weights = {
                        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 1
                    }
                    weight = piece_weights.get(piece.piece_type, 1)
                    total_mobility += piece_mobility * weight
            
            mobility[color_name] = total_mobility
        
        return mobility
    
    @staticmethod
    def _analyze_centralization(board):
        """Analyze piece centralization"""
        center_squares = {chess.D4, chess.D5, chess.E4, chess.E5}
        extended_center = {chess.C3, chess.C4, chess.C5, chess.C6,
                          chess.D3, chess.D6, chess.E3, chess.E6,
                          chess.F3, chess.F4, chess.F5, chess.F6}
        
        centralization = {'white': {'center': 0, 'extended': 0}, 
                         'black': {'center': 0, 'extended': 0}}
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                color_name = 'white' if piece.color == chess.WHITE else 'black'
                
                if square in center_squares:
                    centralization[color_name]['center'] += 1
                elif square in extended_center:
                    centralization[color_name]['extended'] += 1
        
        return centralization
    
    @staticmethod
    def _analyze_development(board):
        """Analyze piece development"""
        development = {'white': {'developed': 0, 'total': 0}, 
                      'black': {'developed': 0, 'total': 0}}
        
        for color in [chess.WHITE, chess.BLACK]:
            color_name = 'white' if color == chess.WHITE else 'black'
            back_rank = 0 if color == chess.WHITE else 7
            
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color == color:
                    if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                        development[color_name]['total'] += 1
                        if chess.square_rank(square) != back_rank:
                            development[color_name]['developed'] += 1
        
        return development
    
    @staticmethod
    def _calculate_activity_score(board):
        """Calculate overall piece activity score"""
        activity = {'white': 0, 'black': 0}
        
        for color in [chess.WHITE, chess.BLACK]:
            color_name = 'white' if color == chess.WHITE else 'black'
            total_activity = 0
            
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color == color:
                    # Base activity from mobility
                    mobility = len(list(board.attacks(square)))
                    
                    # Bonus for active squares
                    rank = chess.square_rank(square)
                    file = chess.square_file(square)
                    
                    # Central files bonus
                    if 2 <= file <= 5:
                        mobility += 1
                    
                    # Advanced pieces bonus
                    if color == chess.WHITE and rank >= 4:
                        mobility += 1
                    elif color == chess.BLACK and rank <= 3:
                        mobility += 1
                    
                    total_activity += mobility
            
            activity[color_name] = total_activity
        
        return activity