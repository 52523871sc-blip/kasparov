"""
Chess AI API Client - Example usage and testing
"""

import requests
import json
from typing import Dict, List, Optional

class ChessAIClient:
    """Client for interacting with Chess AI Play API"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def recommend_move(self, position: Dict, current_player: str = "white", 
                      num_alternatives: int = 3) -> Dict:
        """
        Get move recommendation from AI
        
        Args:
            position: Chess position as dict
            current_player: "white" or "black"
            num_alternatives: Number of alternative moves to analyze
        
        Returns:
            API response with best move and alternatives
        """
        url = f"{self.base_url}/api/recommend-move"
        
        payload = {
            "position": position,
            "current_player": current_player,
            "num_alternatives": num_alternatives
        }
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def analyze_position(self, position: Dict, current_player: str = "white") -> Dict:
        """
        Analyze chess position
        
        Args:
            position: Chess position as dict
            current_player: "white" or "black"
        
        Returns:
            Position analysis
        """
        url = f"{self.base_url}/api/analyze-position"
        
        payload = {
            "position": position,
            "current_player": current_player
        }
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def submit_game(self, moves: List[Dict], result: int, game_id: Optional[str] = None) -> Dict:
        """
        Submit completed game for AI learning
        
        Args:
            moves: List of move data with positions
            result: 1 (white wins), 0 (draw), -1 (black wins)
            game_id: Optional game identifier
        
        Returns:
            Submission confirmation
        """
        url = f"{self.base_url}/api/submit-game"
        
        payload = {
            "moves": moves,
            "result": result
        }
        
        if game_id:
            payload["game_id"] = game_id
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def get_learning_stats(self) -> Dict:
        """Get AI learning statistics"""
        url = f"{self.base_url}/api/learning-stats"
        
        response = self.session.get(url)
        response.raise_for_status()
        
        return response.json()
    
    def health_check(self) -> Dict:
        """Check API health"""
        url = f"{self.base_url}/api/health"
        
        response = self.session.get(url)
        response.raise_for_status()
        
        return response.json()

def create_sample_position():
    """Create sample chess position for testing"""
    return {
        # White pieces
        "a1": {"type": "rook", "color": "white"},
        "b1": {"type": "knight", "color": "white"},
        "c1": {"type": "bishop", "color": "white"},
        "d1": {"type": "queen", "color": "white"},
        "e1": {"type": "king", "color": "white"},
        "f1": {"type": "bishop", "color": "white"},
        "g1": {"type": "knight", "color": "white"},
        "h1": {"type": "rook", "color": "white"},
        "a2": {"type": "pawn", "color": "white"},
        "b2": {"type": "pawn", "color": "white"},
        "c2": {"type": "pawn", "color": "white"},
        "d2": {"type": "pawn", "color": "white"},
        "e2": {"type": "pawn", "color": "white"},
        "f2": {"type": "pawn", "color": "white"},
        "g2": {"type": "pawn", "color": "white"},
        "h2": {"type": "pawn", "color": "white"},
        
        # Black pieces
        "a8": {"type": "rook", "color": "black"},
        "b8": {"type": "knight", "color": "black"},
        "c8": {"type": "bishop", "color": "black"},
        "d8": {"type": "queen", "color": "black"},
        "e8": {"type": "king", "color": "black"},
        "f8": {"type": "bishop", "color": "black"},
        "g8": {"type": "knight", "color": "black"},
        "h8": {"type": "rook", "color": "black"},
        "a7": {"type": "pawn", "color": "black"},
        "b7": {"type": "pawn", "color": "black"},
        "c7": {"type": "pawn", "color": "black"},
        "d7": {"type": "pawn", "color": "black"},
        "e7": {"type": "pawn", "color": "black"},
        "f7": {"type": "pawn", "color": "black"},
        "g7": {"type": "pawn", "color": "black"},
        "h7": {"type": "pawn", "color": "black"}
    }

def create_mid_game_position():
    """Create a mid-game position for testing"""
    return {
        # White pieces
        "a1": {"type": "rook", "color": "white"},
        "e1": {"type": "king", "color": "white"},
        "h1": {"type": "rook", "color": "white"},
        "c3": {"type": "knight", "color": "white"},
        "f3": {"type": "knight", "color": "white"},
        "c4": {"type": "bishop", "color": "white"},
        "d4": {"type": "queen", "color": "white"},
        "a2": {"type": "pawn", "color": "white"},
        "b2": {"type": "pawn", "color": "white"},
        "e4": {"type": "pawn", "color": "white"},
        "f2": {"type": "pawn", "color": "white"},
        "g2": {"type": "pawn", "color": "white"},
        "h2": {"type": "pawn", "color": "white"},
        
        # Black pieces
        "a8": {"type": "rook", "color": "black"},
        "e8": {"type": "king", "color": "black"},
        "h8": {"type": "rook", "color": "black"},
        "c6": {"type": "knight", "color": "black"},
        "f6": {"type": "knight", "color": "black"},
        "c5": {"type": "bishop", "color": "black"},
        "d8": {"type": "queen", "color": "black"},
        "a7": {"type": "pawn", "color": "black"},
        "b7": {"type": "pawn", "color": "black"},
        "e5": {"type": "pawn", "color": "black"},
        "f7": {"type": "pawn", "color": "black"},
        "g7": {"type": "pawn", "color": "black"},
        "h7": {"type": "pawn", "color": "black"}
    }

def demo_api_usage():
    """Demonstrate API usage"""
    client = ChessAIClient()
    
    print("Chess AI Play API Demo")
    print("=" * 40)
    
    try:
        # Health check
        print("1. Health Check")
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print()
        
        # Analyze starting position
        print("2. Analyze Starting Position")
        start_position = create_sample_position()
        analysis = client.analyze_position(start_position, "white")
        print(f"   Evaluation: {analysis['evaluation_text']}")
        print(f"   Score: {analysis['evaluation_score']:.3f}")
        print(f"   Confidence: {analysis['confidence']}")
        print()
        
        # Get move recommendation
        print("3. Get Move Recommendation")
        recommendation = client.recommend_move(start_position, "white", 2)
        
        best_move = recommendation['best_move']
        print(f"   Best Move: {best_move['move']['from']} -> {best_move['move']['to']}")
        print(f"   Score: {best_move['score']:.3f}")
        print(f"   Confidence: {best_move['confidence']}")
        print("   Reasoning:")
        for reason in best_move['reasoning']:
            print(f"     - {reason}")
        
        print("\\n   Alternatives:")
        for i, alt in enumerate(recommendation['alternatives'], 1):
            print(f"     {i}. {alt['move']['from']} -> {alt['move']['to']} (Score: {alt['score']:.3f})")
            print(f"        Why worse: {', '.join(alt['why_worse'])}")
            if alt['merits']:
                print(f"        Merits: {', '.join(alt['merits'])}")
        print()
        
        # Mid-game analysis
        print("4. Mid-Game Position Analysis")
        mid_position = create_mid_game_position()
        mid_recommendation = client.recommend_move(mid_position, "white", 2)
        
        best_mid = mid_recommendation['best_move']
        print(f"   Best Move: {best_mid['move']['from']} -> {best_mid['move']['to']}")
        print(f"   Score: {best_mid['score']:.3f}")
        print("   Analysis:")
        analysis = best_mid['analysis']
        print(f"     Tactics: {', '.join(analysis['tactical_elements']) or 'None'}")
        print(f"     Themes: {', '.join(analysis['strategic_themes']) or 'None'}")
        print()
        
        # Submit sample game for learning
        print("5. Submit Game for Learning")
        sample_moves = [
            {
                "position_before": start_position,
                "position_after": mid_position,
                "move": {"from": "e2", "to": "e4", "piece": "pawn", "color": "white"}
            }
        ]
        
        submission = client.submit_game(sample_moves, 1, "demo_game")
        print(f"   Game submitted: {submission['game_id']}")
        print(f"   Moves learned: {submission['moves_learned']}")
        print()
        
        # Learning statistics
        print("6. Learning Statistics")
        stats = client.get_learning_stats()
        print(f"   Total games: {stats['total_games_learned']}")
        print(f"   Total positions: {stats['total_positions_analyzed']}")
        print(f"   Cache size: {stats['cache_size']}")
        print(f"   Model exists: {stats['model_file_exists']}")
        
        if stats['recent_games']:
            print("   Recent games:")
            for game in stats['recent_games']:
                result_text = {1: "White wins", 0: "Draw", -1: "Black wins"}[game['result']]
                print(f"     - {game['game_id']}: {result_text} ({game['moves']} moves)")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure the server is running.")
        print("Start the server with: python chess_api.py")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    demo_api_usage()