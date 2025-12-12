# Chess AI Training Guide

## 🎯 Overview

This guide explains how to train the Chess AI with online game data to improve its performance.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training
```bash
# Quick training (recommended for first time)
python3 simple_trainer.py

# Online training with real games
python3 online_trainer.py

# Interactive training interface
python3 master_trainer.py
```

## 📚 Training Scripts

### `simple_trainer.py`
- **Purpose**: Quick training with sample data
- **Time**: ~30 seconds
- **Games**: 10 sample games
- **Best for**: Testing and initial setup

### `online_trainer.py`
- **Purpose**: Training with real Lichess games
- **Time**: 2-5 minutes
- **Games**: 10-50 real games
- **Best for**: Improving AI with real data

### `master_trainer.py`
- **Purpose**: Interactive training interface
- **Features**: Multiple training modes, statistics, testing
- **Best for**: Comprehensive training management

### `train_with_online_data.py`
- **Purpose**: Advanced online training with PGN parsing
- **Features**: Top player games, batch processing
- **Best for**: Large-scale training

## 🎮 Training Options

### 1. Online Training
Fetches real games from Lichess API:
- **Quick**: 10 games (~2 minutes)
- **Standard**: 25 games (~5 minutes)  
- **Intensive**: 50+ games (~10+ minutes)

### 2. Sample Training
Uses generated sample games:
- **Fast**: Immediate training
- **Consistent**: Reproducible results
- **Testing**: Good for development

### 3. Continuous Training
Multiple training rounds:
- **Rounds**: 3-10 training sessions
- **Progressive**: Builds on previous training
- **Robust**: More stable learning

## 📊 Model Performance

### Before Training
- Random evaluations
- Basic move generation
- No strategic understanding

### After Training
- Improved position evaluation
- Better move recommendations
- Strategic pattern recognition
- Confidence scoring

## 🔧 Training Configuration

### Model Architecture
- **Input**: 773 features (board + metadata)
- **Hidden Layers**: 512 → 256 → 128 neurons
- **Output**: Position evaluation (-1 to +1)
- **Activation**: ReLU + Tanh

### Training Parameters
- **Learning Rate**: 0.001
- **Epochs**: 30 per batch
- **Batch Size**: 50+ positions
- **Loss Function**: Mean Squared Error

### Data Sources
- **Lichess API**: Live tournament games
- **Sample Games**: Generated positions
- **Custom Games**: User-submitted via API

## 📈 Monitoring Progress

### Model Statistics
```bash
python3 master_trainer.py
# Select option 4: Model Statistics
```

Shows:
- Model file size
- Evaluation speed
- Cache statistics
- Sample evaluations

### Testing Performance
```bash
python3 master_trainer.py  
# Select option 5: Test Current Model
```

Tests:
- Position evaluation accuracy
- Move generation quality
- Confidence levels
- Error handling

## 🎯 Training Best Practices

### 1. Start Small
- Begin with `simple_trainer.py`
- Verify everything works
- Then move to online training

### 2. Progressive Training
- Train with 10-20 games initially
- Gradually increase to 50+ games
- Use continuous training for best results

### 3. Monitor Performance
- Check model statistics regularly
- Test on known positions
- Compare before/after training

### 4. Data Quality
- Online games provide better training
- Mix of game types (blitz, classical)
- Include games with different outcomes

## 🚨 Troubleshooting

### Common Issues

**"No games fetched"**
- Check internet connection
- Lichess API might be rate-limited
- Use sample training as fallback

**"Training errors"**
- Model dimensions mismatch
- Restart with fresh model
- Use `master_trainer.py` option 6 to reset

**"Slow training"**
- Reduce number of games
- Use sample training for testing
- Check system resources

### Solutions

**Reset Model**
```bash
python3 master_trainer.py
# Select option 6: Reset Model
```

**Check Dependencies**
```bash
pip install -r requirements.txt
```

**Verify Installation**
```bash
python3 -c "import chess, requests, numpy; print('All dependencies OK')"
```

## 🎉 Success Metrics

### Training Success Indicators
- ✅ Model file created/updated
- ✅ Training loss decreases
- ✅ No error messages
- ✅ Evaluation speed < 100ms

### Performance Improvements
- 📈 More consistent evaluations
- 📈 Better move recommendations
- 📈 Higher confidence scores
- 📈 Strategic understanding

## 🔄 Continuous Improvement

### Regular Training
- Train weekly with new games
- Use different game types
- Monitor performance trends

### Advanced Training
- Increase training data size
- Experiment with parameters
- Add custom game collections

---

**Happy Training! Your AI will get stronger with each game! 🧠♟️**