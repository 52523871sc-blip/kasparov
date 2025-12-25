import chess

class TacticalAnalyzer:
    @staticmethod
    def detect_tactical_motifs(board):
        """Detect tactical patterns"""
        return {
            'pins': TacticalAnalyzer._detect_pins(board),
            'forks': TacticalAnalyzer._detect_forks(board),
            'skewers': TacticalAnalyzer._detect_skewers(board),
            'discovered_attacks': TacticalAnalyzer._detect_discovered_attacks(board),
            'back_rank_threats': TacticalAnalyzer._detect_back_rank_threats(board),
            'tactical_count': 0  # Will be calculated
        }
    
    @staticmethod
    def _detect_pins(board):
        """Detect pinned pieces"""
        pins = []
        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if not king_square:
                continue
            
            # Check for pins along ranks, files, and diagonals
            for direction in [(0,1), (1,0), (1,1), (1,-1), (0,-1), (-1,0), (-1,-1), (-1,1)]:
                pinned = TacticalAnalyzer._find_pin_in_direction(board, king_square, direction, color)
                if pinned:
                    pins.append(pinned)
        
        return pins
    
    @staticmethod
    def _find_pin_in_direction(board, king_square, direction, king_color):
        """Find pin in specific direction"""
        dr, dc = direction
        r, c = chess.square_rank(king_square), chess.square_file(king_square)
        
        pieces_in_line = []
        r, c = r + dr, c + dc
        
        while 0 <= r < 8 and 0 <= c < 8:
            square = chess.square(c, r)
            piece = board.piece_at(square)
            if piece:
                pieces_in_line.append((square, piece))
                if len(pieces_in_line) == 2:
                    break
            r, c = r + dr, c + dc
        
        if len(pieces_in_line) == 2:
            first_piece = pieces_in_line[0][1]
            second_piece = pieces_in_line[1][1]
            
            if (first_piece.color == king_color and 
                second_piece.color != king_color and
                TacticalAnalyzer._can_attack_along_line(second_piece, direction)):
                return {
                    'pinned_square': chess.square_name(pieces_in_line[0][0]),
                    'pinning_piece': chess.square_name(pieces_in_line[1][0]),
                    'direction': direction
                }
        
        return None
    
    @staticmethod
    def _can_attack_along_line(piece, direction):
        """Check if piece can attack along given direction"""
        dr, dc = direction
        if piece.piece_type == chess.ROOK:
            return dr == 0 or dc == 0
        elif piece.piece_type == chess.BISHOP:
            return abs(dr) == abs(dc)
        elif piece.piece_type == chess.QUEEN:
            return True
        return False
    
    @staticmethod
    def _detect_forks(board):
        """Detect fork opportunities"""
        forks = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.KNIGHT:
                targets = TacticalAnalyzer._get_knight_fork_targets(board, square, piece.color)
                if len(targets) >= 2:
                    forks.append({
                        'forking_piece': chess.square_name(square),
                        'targets': [chess.square_name(t) for t in targets]
                    })
        return forks
    
    @staticmethod
    def _get_knight_fork_targets(board, knight_square, knight_color):
        """Get potential fork targets for knight"""
        targets = []
        knight_moves = [(-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1)]
        
        r, c = chess.square_rank(knight_square), chess.square_file(knight_square)
        
        for dr, dc in knight_moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                target_square = chess.square(nc, nr)
                target_piece = board.piece_at(target_square)
                if target_piece and target_piece.color != knight_color:
                    if target_piece.piece_type in [chess.KING, chess.QUEEN, chess.ROOK]:
                        targets.append(target_square)
        
        return targets
    
    @staticmethod
    def _detect_skewers(board):
        """Detect skewer patterns"""
        # Simplified skewer detection
        return []
    
    @staticmethod
    def _detect_discovered_attacks(board):
        """Detect discovered attack patterns"""
        # Simplified discovered attack detection
        return []
    
    @staticmethod
    def _detect_back_rank_threats(board):
        """Detect back rank mate threats"""
        threats = []
        
        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if not king_square:
                continue
            
            king_rank = chess.square_rank(king_square)
            back_rank = 0 if color == chess.WHITE else 7
            
            if king_rank == back_rank:
                # Check if king is trapped by own pawns
                escape_squares = TacticalAnalyzer._count_king_escape_squares(board, king_square)
                if escape_squares == 0:
                    threats.append({
                        'threatened_king': chess.square_name(king_square),
                        'color': 'white' if color == chess.WHITE else 'black',
                        'severity': 'high'
                    })
        
        return threats
    
    @staticmethod
    def _count_king_escape_squares(board, king_square):
        """Count available escape squares for king"""
        count = 0
        r, c = chess.square_rank(king_square), chess.square_file(king_square)
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    escape_square = chess.square(nc, nr)
                    if not board.piece_at(escape_square):
                        count += 1
        
        return count