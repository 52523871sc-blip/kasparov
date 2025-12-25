import chess

class NarrativeGenerator:
    @staticmethod
    def generate_training_narrative(board, move, analysis, stockfish_info):
        """Generate comprehensive training narrative"""
        return {
            'position_story': NarrativeGenerator._describe_position_story(board, analysis),
            'move_explanation': NarrativeGenerator._explain_move_choice(move, stockfish_info),
            'learning_points': NarrativeGenerator._extract_learning_points(analysis, stockfish_info),
            'common_mistakes': NarrativeGenerator._identify_common_mistakes(board, stockfish_info),
            'improvement_suggestions': NarrativeGenerator._suggest_improvements(analysis, stockfish_info),
            'tactical_lessons': NarrativeGenerator._extract_tactical_lessons(board, move, stockfish_info),
            'strategic_lessons': NarrativeGenerator._extract_strategic_lessons(board, analysis)
        }
    
    @staticmethod
    def _describe_position_story(board, analysis):
        """Describe the current position narrative"""
        story = []
        
        # Game phase
        phase = analysis.get('game_phase', {}).get('phase', 'unknown')
        story.append(f"This is a {phase} position")
        
        # Material situation
        material = analysis.get('material_balance', {})
        if material.get('difference', 0) > 2:
            story.append(f"White has a material advantage of {material['difference']} points")
        elif material.get('difference', 0) < -2:
            story.append(f"Black has a material advantage of {abs(material['difference'])} points")
        else:
            story.append("Material is roughly equal")
        
        # King safety
        king_safety = analysis.get('king_safety', {})
        if king_safety.get('white_king_safety') == 'exposed':
            story.append("White's king is exposed and vulnerable")
        if king_safety.get('black_king_safety') == 'exposed':
            story.append("Black's king is exposed and vulnerable")
        
        # Center control
        center = analysis.get('center_control', {})
        if center.get('advantage') != 'equal':
            story.append(f"{center['advantage'].capitalize()} controls the center")
        
        return '. '.join(story) + '.'\n    \n    @staticmethod\n    def _explain_move_choice(move, stockfish_info):
        \"\"\"Explain why this move was chosen\"\"\"\n        explanation = []\n        \n        score = stockfish_info.get('score', 0)\n        score_text = stockfish_info.get('score_text', '0.00')\n        \n        # Score interpretation\n        if abs(score) > 1000:\n            explanation.append(f\"This move leads to a decisive advantage (evaluation: {score_text})\")\n        elif abs(score) > 300:\n            explanation.append(f\"This move provides a significant advantage (evaluation: {score_text})\")\n        elif abs(score) > 100:\n            explanation.append(f\"This move gives a clear edge (evaluation: {score_text})\")\n        else:\n            explanation.append(f\"This move maintains balance (evaluation: {score_text})\")\n        \n        # Tactical elements\n        if stockfish_info.get('is_capture'):\n            explanation.append(\"The move captures material\")\n        \n        if stockfish_info.get('gives_check'):\n            explanation.append(\"The move gives check, forcing the opponent's response\")\n        \n        # Principal variation\n        pv = stockfish_info.get('principal_variation', [])\n        if len(pv) > 1:\n            pv_text = ' '.join(pv[:3])\n            explanation.append(f\"Best continuation: {pv_text}\")\n        \n        return ' '.join(explanation)\n    \n    @staticmethod\n    def _extract_learning_points(analysis, stockfish_info):
        \"\"\"Extract key learning points\"\"\"\n        points = []\n        \n        # Tactical lessons\n        tactical = analysis.get('tactical_motifs', {})\n        if tactical.get('pins'):\n            points.append(\"Learn to recognize and exploit pinned pieces\")\n        if tactical.get('forks'):\n            points.append(\"Knight forks can win material\")\n        if tactical.get('back_rank_threats'):\n            points.append(\"Always be aware of back rank mate threats\")\n        \n        # Strategic lessons\n        pawn_structure = analysis.get('pawn_structure', {})\n        if pawn_structure.get('isolated_pawns', {}).get('black', 0) > 0:\n            points.append(\"Isolated pawns are weak and should be avoided\")\n        if pawn_structure.get('passed_pawns', {}).get('white', []):\n            points.append(\"Passed pawns are powerful in the endgame\")\n        \n        # Development lessons\n        development = analysis.get('piece_coordination', {}).get('piece_development', {})\n        if development.get('white', {}).get('developed', 0) < 2:\n            points.append(\"Develop knights and bishops early in the game\")\n        \n        return points
    
    @staticmethod
    def _identify_common_mistakes(board, stockfish_info):
        \"\"\"Identify common chess mistakes in position\"\"\"\n        mistakes = []\n        \n        # Check for king safety violations
        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if king_square:
                king_rank = chess.square_rank(king_square)
                king_file = chess.square_file(king_square)
                
                # Check for weakened king position
                pawn_shield_count = 0
                for df in [-1, 0, 1]:
                    for dr in [1, -1]:
                        check_file = king_file + df
                        check_rank = king_rank + dr
                        
                        if 0 <= check_file < 8 and 0 <= check_rank < 8:
                            check_square = chess.square(check_file, check_rank)
                            piece = board.piece_at(check_square)
                            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                                pawn_shield_count += 1
                
                if pawn_shield_count < 2:
                    color_name = 'White' if color == chess.WHITE else 'Black'
                    mistakes.append(f\"{color_name}'s king lacks pawn shelter - vulnerable to attacks\")\n        
        return mistakes
    
    @staticmethod
    def _suggest_improvements(analysis, stockfish_info):
        \"\"\"Suggest improvements based on analysis\"\"\"\n        suggestions = []
        
        # Development suggestions
        development = analysis.get('piece_coordination', {}).get('piece_development', {})
        for color in ['white', 'black']:
            dev_info = development.get(color, {})
            if dev_info.get('developed', 0) < dev_info.get('total', 0):
                suggestions.append(f\"{color.capitalize()} should complete piece development\")\n        \n        # Center control suggestions
        center = analysis.get('center_control', {})
        if center.get('advantage') == 'equal':
            suggestions.append(\"Fight for central control with pawns and pieces\")\n        \        # Pawn structure suggestions
        pawn_structure = analysis.get('pawn_structure', {})
        if pawn_structure.get('isolated_pawns', {}).get('white', 0) > 0:
            suggestions.append(\"White should protect or advance isolated pawns\")\n        if pawn_structure.get('isolated_pawns', {}).get('black', 0) > 0:
            suggestions.append(\"Black should protect or advance isolated pawns\")\n        
        return suggestions
    
    @staticmethod
    def _extract_tactical_lessons(board, move, stockfish_info):
        \"\"\"Extract tactical lessons from move\"\"\"\n        lessons = []
        
        if stockfish_info.get('is_capture'):
            lessons.append(\"Capturing material is often the strongest continuation\")\n        
        if stockfish_info.get('gives_check'):
            lessons.append(\"Checks can force the opponent into unfavorable positions\")\n        
        # Check for tactical themes in score breakdown
        score_breakdown = stockfish_info.get('score_breakdown', {})
        for component in score_breakdown.get('components', []):
            if component['factor'] == 'Capture Bonus':
                lessons.append(f\"Material gain: {component['description']}\")\n        
        return lessons
    
    @staticmethod
    def _extract_strategic_lessons(board, analysis):
        \"\"\"Extract strategic lessons from position\"\"\"\n        lessons = []
        
        # Game phase lessons
        phase = analysis.get('game_phase', {}).get('phase')
        if phase == 'opening':
            lessons.append(\"In the opening, focus on development, center control, and king safety\")\n        elif phase == 'middlegame':
            lessons.append(\"In the middlegame, look for tactical opportunities and improve piece positions\")\
        elif phase == 'endgame':
            lessons.append(\"In the endgame, activate your king and push passed pawns\")\n        
        # Pawn structure lessons
        pawn_structure = analysis.get('pawn_structure', {})
        if pawn_structure.get('passed_pawns', {}).get('white', []):
            lessons.append(\"Passed pawns should be pushed and supported\")\n        
        return lessons