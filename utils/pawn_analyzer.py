import chess

class PawnAnalyzer:
    @staticmethod
    def analyze_pawn_structure_detailed(board):
        """Comprehensive pawn structure analysis"""
        return {
            'isolated_pawns': PawnAnalyzer._count_isolated_pawns(board),
            'doubled_pawns': PawnAnalyzer._count_doubled_pawns(board),
            'backward_pawns': PawnAnalyzer._count_backward_pawns(board),
            'passed_pawns': PawnAnalyzer._count_passed_pawns(board),
            'pawn_chains': PawnAnalyzer._analyze_pawn_chains(board),
            'pawn_storms': PawnAnalyzer._detect_pawn_storms(board),
            'pawn_islands': PawnAnalyzer._count_pawn_islands(board)
        }
    
    @staticmethod
    def _count_isolated_pawns(board):
        """Count isolated pawns for each side"""
        isolated = {'white': 0, 'black': 0}
        
        for color in [chess.WHITE, chess.BLACK]:
            color_name = 'white' if color == chess.WHITE else 'black'
            pawn_files = set()
            
            # Get all pawn files for this color
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    pawn_files.add(chess.square_file(square))
            
            # Check each pawn for isolation
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    file = chess.square_file(square)
                    # Check adjacent files
                    has_support = False
                    for adj_file in [file - 1, file + 1]:
                        if 0 <= adj_file < 8 and adj_file in pawn_files:
                            has_support = True
                            break
                    
                    if not has_support:
                        isolated[color_name] += 1
        
        return isolated
    
    @staticmethod
    def _count_doubled_pawns(board):
        """Count doubled pawns for each side"""
        doubled = {'white': 0, 'black': 0}
        
        for color in [chess.WHITE, chess.BLACK]:
            color_name = 'white' if color == chess.WHITE else 'black'
            file_counts = [0] * 8
            
            # Count pawns per file
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    file_counts[chess.square_file(square)] += 1
            
            # Count doubled pawns
            for count in file_counts:
                if count > 1:
                    doubled[color_name] += count - 1
        
        return doubled
    
    @staticmethod
    def _count_backward_pawns(board):
        """Count backward pawns (simplified)"""
        return {'white': 0, 'black': 0}  # Simplified implementation
    
    @staticmethod
    def _count_passed_pawns(board):
        """Count passed pawns for each side"""
        passed = {'white': [], 'black': []}
        
        for color in [chess.WHITE, chess.BLACK]:
            color_name = 'white' if color == chess.WHITE else 'black'
            
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    if PawnAnalyzer._is_passed_pawn(board, square, color):
                        passed[color_name].append(chess.square_name(square))
        
        return passed
    
    @staticmethod
    def _is_passed_pawn(board, pawn_square, pawn_color):
        """Check if pawn is passed"""
        file = chess.square_file(pawn_square)
        rank = chess.square_rank(pawn_square)
        
        # Check files: current, left, right
        check_files = [f for f in [file - 1, file, file + 1] if 0 <= f < 8]
        
        # Direction to check based on color
        if pawn_color == chess.WHITE:
            check_ranks = range(rank + 1, 8)
        else:
            check_ranks = range(0, rank)
        
        # Look for opposing pawns
        for check_file in check_files:
            for check_rank in check_ranks:
                check_square = chess.square(check_file, check_rank)
                piece = board.piece_at(check_square)
                if piece and piece.piece_type == chess.PAWN and piece.color != pawn_color:
                    return False
        
        return True
    
    @staticmethod
    def _analyze_pawn_chains(board):
        """Analyze pawn chain structures"""
        chains = {'white': [], 'black': []}
        
        for color in [chess.WHITE, chess.BLACK]:
            color_name = 'white' if color == chess.WHITE else 'black'
            visited = set()
            
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if (piece and piece.piece_type == chess.PAWN and 
                    piece.color == color and square not in visited):
                    
                    chain = PawnAnalyzer._find_pawn_chain(board, square, color, visited)
                    if len(chain) >= 2:
                        chains[color_name].append([chess.square_name(s) for s in chain])
        
        return chains
    
    @staticmethod
    def _find_pawn_chain(board, start_square, color, visited):
        """Find connected pawn chain starting from square"""
        chain = [start_square]
        visited.add(start_square)
        
        # Simple chain detection (diagonal support)
        r, c = chess.square_rank(start_square), chess.square_file(start_square)
        
        # Check diagonal support
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                support_square = chess.square(nc, nr)
                piece = board.piece_at(support_square)
                if (piece and piece.piece_type == chess.PAWN and 
                    piece.color == color and support_square not in visited):
                    chain.extend(PawnAnalyzer._find_pawn_chain(board, support_square, color, visited))
        
        return chain
    
    @staticmethod
    def _detect_pawn_storms(board):
        """Detect pawn storm attacks"""
        storms = {'white': [], 'black': []}
        
        # Simplified: detect advanced pawns near enemy king
        for color in [chess.WHITE, chess.BLACK]:
            color_name = 'white' if color == chess.WHITE else 'black'
            enemy_king = board.king(not color)
            
            if enemy_king:
                king_file = chess.square_file(enemy_king)
                
                # Look for advanced pawns near enemy king
                for square in chess.SQUARES:
                    piece = board.piece_at(square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        pawn_file = chess.square_file(square)
                        pawn_rank = chess.square_rank(square)
                        
                        # Check if pawn is advanced and near enemy king
                        if abs(pawn_file - king_file) <= 2:
                            if ((color == chess.WHITE and pawn_rank >= 4) or 
                                (color == chess.BLACK and pawn_rank <= 3)):
                                storms[color_name].append(chess.square_name(square))
        
        return storms
    
    @staticmethod
    def _count_pawn_islands(board):
        """Count pawn islands for each side"""
        islands = {'white': 0, 'black': 0}
        
        for color in [chess.WHITE, chess.BLACK]:
            color_name = 'white' if color == chess.WHITE else 'black'
            pawn_files = set()
            
            # Get all files with pawns
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    pawn_files.add(chess.square_file(square))
            
            # Count islands (groups of consecutive files)
            if pawn_files:
                sorted_files = sorted(pawn_files)
                island_count = 1
                
                for i in range(1, len(sorted_files)):
                    if sorted_files[i] - sorted_files[i-1] > 1:
                        island_count += 1
                
                islands[color_name] = island_count
        
        return islands