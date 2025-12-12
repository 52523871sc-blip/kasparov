#!/usr/bin/env python3
"""
Setup and run training with online data
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def run_training():
    """Run the training script"""
    print("\nStarting training with online data...")
    
    try:
        # Import and run training
        from train_with_online_data import OnlineDataTrainer
        
        trainer = OnlineDataTrainer()
        
        # Quick training with sample data
        print("Running quick training session...")
        trainer.train_from_online_data(max_games=20, source="top_players")
        
        print("\nTraining completed successfully!")
        print("You can now test the improved model with: python api_client.py")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Falling back to batch trainer...")
        
        try:
            from batch_trainer import BatchTrainer
            trainer = BatchTrainer()
            trainer.batch_train(num_games=50)
        except Exception as e2:
            print(f"Training failed: {e2}")

def main():
    print("Chess AI Training Setup")
    print("=" * 30)
    
    # Check if we're in the right directory
    if not os.path.exists("chess_ai_engine.py"):
        print("Error: Please run this script from the project directory")
        return
    
    # Install dependencies
    if not install_dependencies():
        print("Failed to install dependencies. Please install manually:")
        print("pip install -r requirements.txt")
        return
    
    # Run training
    run_training()

if __name__ == "__main__":
    main()