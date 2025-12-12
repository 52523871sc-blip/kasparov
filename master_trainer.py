#!/usr/bin/env python3
"""
Master Training Script - Multiple training options for Chess AI
"""

import sys
import os
from online_trainer import OnlineTrainer
from simple_trainer import train_model as simple_train
from chess_ai_engine import ChessAIEngine

def show_menu():
    """Display training options menu"""
    print("\n" + "="*50)
    print("🧠 CHESS AI MASTER TRAINER")
    print("="*50)
    print("1. 🌐 Online Training (Lichess games)")
    print("2. 📊 Sample Data Training (Quick)")
    print("3. 🔄 Continuous Training (Multiple rounds)")
    print("4. 📈 Model Statistics")
    print("5. 🧪 Test Current Model")
    print("6. 🗑️  Reset Model")
    print("0. ❌ Exit")
    print("="*50)

def online_training():
    """Run online training"""
    print("\n🌐 ONLINE TRAINING MODE")
    print("-" * 30)
    
    trainer = OnlineTrainer()
    
    print("Options:")
    print("1. Quick training (10 games)")
    print("2. Standard training (25 games)")
    print("3. Intensive training (50 games)")
    print("4. Custom amount")
    
    choice = input("Select option (1-4): ").strip()
    
    if choice == "1":
        num_games = 10
    elif choice == "2":
        num_games = 25
    elif choice == "3":
        num_games = 50
    elif choice == "4":
        num_games = int(input("Number of games: "))
    else:
        num_games = 10
    
    print(f"\nStarting online training with {num_games} games...")
    trainer.train_with_online_data(num_games)

def sample_training():
    """Run sample data training"""
    print("\n📊 SAMPLE DATA TRAINING")
    print("-" * 30)
    
    print("Training with generated sample data...")
    simple_train()

def continuous_training():
    """Run multiple training rounds"""
    print("\n🔄 CONTINUOUS TRAINING MODE")
    print("-" * 30)
    
    rounds = int(input("Number of training rounds (default 3): ") or "3")
    games_per_round = int(input("Games per round (default 15): ") or "15")
    
    trainer = OnlineTrainer()
    
    for round_num in range(1, rounds + 1):
        print(f"\n--- ROUND {round_num}/{rounds} ---")
        trainer.train_with_online_data(games_per_round)
        
        if round_num < rounds:
            print("Waiting before next round...")
            import time
            time.sleep(2)
    
    print(f"\n✅ Completed {rounds} training rounds!")

def show_model_stats():
    """Show model statistics"""
    print("\n📈 MODEL STATISTICS")
    print("-" * 30)
    
    engine = ChessAIEngine()
    
    # Check if model exists
    if os.path.exists(engine.model_path):
        print(f"✅ Model file exists: {engine.model_path}")
        
        # Get file size
        size = os.path.getsize(engine.model_path)
        print(f"📁 Model size: {size:,} bytes ({size/1024:.1f} KB)")
        
        # Test model performance
        print("\n🧪 Testing model performance...")
        
        test_position = {
            "e1": {"type": "king", "color": "white"},
            "e8": {"type": "king", "color": "black"},
            "d1": {"type": "queen", "color": "white"},
            "d8": {"type": "queen", "color": "black"}
        }
        
        import time
        start_time = time.time()
        score = engine.evaluate_position(test_position, "white")
        eval_time = time.time() - start_time
        
        print(f"⚡ Evaluation speed: {eval_time*1000:.1f}ms")
        print(f"🎯 Sample evaluation: {score:.3f}")
        
        # Cache statistics
        print(f"💾 Position cache size: {len(engine.position_cache)}")
        
    else:
        print("❌ No trained model found")
        print("Run training first to create a model")

def test_model():
    """Test current model"""
    print("\n🧪 MODEL TESTING")
    print("-" * 30)
    
    engine = ChessAIEngine()
    
    # Test positions
    test_positions = [
        {
            "name": "Starting Position",
            "position": {
                "e1": {"type": "king", "color": "white"},
                "e8": {"type": "king", "color": "black"},
                "d1": {"type": "queen", "color": "white"},
                "d8": {"type": "queen", "color": "black"},
                "a1": {"type": "rook", "color": "white"},
                "h1": {"type": "rook", "color": "white"},
                "a8": {"type": "rook", "color": "black"},
                "h8": {"type": "rook", "color": "black"}
            }
        },
        {
            "name": "Endgame Position",
            "position": {
                "e1": {"type": "king", "color": "white"},
                "e8": {"type": "king", "color": "black"},
                "d1": {"type": "queen", "color": "white"}
            }
        }
    ]
    
    for test in test_positions:
        print(f"\n🔍 Testing: {test['name']}")
        
        try:
            score = engine.evaluate_position(test['position'], "white")
            print(f"   Evaluation: {score:.3f}")
            
            moves = engine.get_best_moves(test['position'], "white", 2)
            print(f"   Best moves found: {len(moves)}")
            
            if moves:
                best = moves[0]
                print(f"   Top move: {best['move']['from']}-{best['move']['to']}")
                print(f"   Confidence: {best['analysis']['confidence']}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def reset_model():
    """Reset the model"""
    print("\n🗑️  MODEL RESET")
    print("-" * 30)
    
    engine = ChessAIEngine()
    
    if os.path.exists(engine.model_path):
        confirm = input("⚠️  Are you sure you want to reset the model? (yes/no): ")
        if confirm.lower() == 'yes':
            os.remove(engine.model_path)
            print("✅ Model reset successfully")
            print("The model will be recreated on next training")
        else:
            print("❌ Reset cancelled")
    else:
        print("ℹ️  No model file found to reset")

def main():
    """Main training interface"""
    print("🚀 Initializing Chess AI Training System...")
    
    # Check dependencies
    try:
        import chess
        import requests
        print("✅ All dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return
    
    while True:
        show_menu()
        
        choice = input("\nSelect option (0-6): ").strip()
        
        if choice == "0":
            print("\n👋 Goodbye! Happy chess playing!")
            break
        elif choice == "1":
            online_training()
        elif choice == "2":
            sample_training()
        elif choice == "3":
            continuous_training()
        elif choice == "4":
            show_model_stats()
        elif choice == "5":
            test_model()
        elif choice == "6":
            reset_model()
        else:
            print("❌ Invalid option. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()