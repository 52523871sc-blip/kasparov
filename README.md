# Calnic AI Chess Play API

An intelligent chess API that provides AI-powered move recommendations with detailed analysis. The system uses neural networks to evaluate positions and continuously learns from submitted games.

## 🤖 Features

- **AI Move Recommendations**: Get the best moves with confidence scores
- **Detailed Analysis**: Understand why moves are good or bad
- **Alternative Evaluation**: Compare multiple move options with explanations
- **Position Analysis**: Evaluate any chess position
- **Continuous Learning**: AI improves from submitted game results
- **Independent System**: No external dependencies or chess engines required

## 🚀 Quick Start

### Installation

```bash
cd calnic-ai-chess-play-api
pip install -r requirements.txt
```

### Start the API Server

```bash
python chess_api.py
```

The API will be available at `http://localhost:5000`

### Test the API

```bash
python api_client.py
```

## 📡 API Endpoints

### 1. Get Move Recommendation

**POST** `/api/recommend-move`

Get AI-powered move recommendations with detailed analysis.

```json
{
  "position": {
    "e1": {"type": "king", "color": "white"},
    "e8": {"type": "king", "color": "black"},
    "e2": {"type": "pawn", "color": "white"}
  },
  "current_player": "white",
  "num_alternatives": 3
}
```

**Response:**
```json
{
  "best_move": {
    "move": {"from": "e2", "to": "e4", "piece": "pawn", "color": "white"},
    "score": 0.234,
    "confidence": "medium",
    "analysis": {
      "tactical_elements": ["development"],
      "strategic_themes": ["center_control"],
      "positional_factors": {...}
    },
    "reasoning": [
      "This move maintains a slight advantage",
      "Improves center control",
      "Develops pieces effectively"
    ]
  },
  "alternatives": [
    {
      "move": {"from": "d2", "to": "d4", "piece": "pawn", "color": "white"},
      "score": 0.198,
      "why_worse": ["Slightly weaker center control"],
      "merits": ["Alternative development"]
    }
  ]
}
```

### 2. Analyze Position

**POST** `/api/analyze-position`

Evaluate a chess position without move recommendations.

```json
{
  "position": {...},
  "current_player": "white"
}
```

### 3. Submit Game for Learning

**POST** `/api/submit-game`

Submit completed games to improve the AI.

```json
{
  "moves": [
    {
      "position_before": {...},
      "position_after": {...},
      "move": {...}
    }
  ],
  "result": 1,
  "game_id": "optional_id"
}
```

- `result`: `1` (white wins), `0` (draw), `-1` (black wins)

### 4. Learning Statistics

**GET** `/api/learning-stats`

Get AI learning progress and statistics.

### 5. Health Check

**GET** `/api/health`

Check API status and availability.

## 🧠 AI Analysis Features

### Move Evaluation
- **Neural Network Scoring**: Positions evaluated from -1 to +1
- **Confidence Levels**: very_low, low, medium, high, very_high
- **Tactical Recognition**: Captures, checks, forks, pins
- **Strategic Themes**: Development, centralization, coordination

### Detailed Comparisons
- **Score Differences**: Quantified move quality gaps
- **Weakness Explanations**: Why alternatives are inferior
- **Alternative Merits**: Positive aspects of other moves
- **Reasoning**: Human-readable move explanations

### Continuous Learning
- **Game Result Learning**: AI learns from win/loss/draw outcomes
- **Position Evaluation**: Improves position assessment accuracy
- **Pattern Recognition**: Discovers new tactical and strategic patterns
- **Model Persistence**: Saves learned knowledge automatically

## 💡 Usage Examples

### Python Client

```python
from api_client import ChessAIClient

client = ChessAIClient("http://localhost:5000")

# Get move recommendation
position = {
    "e1": {"type": "king", "color": "white"},
    "e8": {"type": "king", "color": "black"},
    # ... more pieces
}

recommendation = client.recommend_move(position, "white", 3)
print(f"Best move: {recommendation['best_move']['move']}")
print(f"Score: {recommendation['best_move']['score']}")

# Submit game for learning
moves = [...]  # Game moves with positions
client.submit_game(moves, result=1)  # White wins
```

### cURL Examples

```bash
# Get move recommendation
curl -X POST http://localhost:5000/api/recommend-move \
  -H "Content-Type: application/json" \
  -d '{"position": {...}, "current_player": "white"}'

# Check learning stats
curl http://localhost:5000/api/learning-stats
```

## 🎯 Position Format

Positions are represented as dictionaries where:
- **Keys**: Square names (e.g., "e1", "a8", "h4")
- **Values**: Piece objects with `type` and `color`

```json
{
  "e1": {"type": "king", "color": "white"},
  "d1": {"type": "queen", "color": "white"},
  "a1": {"type": "rook", "color": "white"},
  "e8": {"type": "king", "color": "black"},
  "d8": {"type": "queen", "color": "black"},
  "a8": {"type": "rook", "color": "black"}
}
```

**Piece Types**: `king`, `queen`, `rook`, `bishop`, `knight`, `pawn`
**Colors**: `white`, `black`

## 🔧 Configuration

### Model Settings
- Model saved to: `models/chess_ai_model.pkl`
- Training triggers: Every 50 new positions
- Cache size: Unlimited (clears on restart)

### API Settings
- Host: `0.0.0.0` (all interfaces)
- Port: `5000`
- Debug mode: Enabled in development

## 📊 Learning Process

### 1. Game Submission
Submit completed games with move sequences and results

### 2. Feature Extraction
Convert positions to 773-dimensional feature vectors

### 3. Neural Network Training
4-layer network learns position evaluation

### 4. Model Persistence
Trained models automatically saved and loaded

### 5. Continuous Improvement
AI gets better with each submitted game

## 🛠️ Development

### Project Structure
```
calnic-ai-chess-play-api/
├── chess_ai_engine.py    # Core AI engine
├── chess_api.py          # REST API server
├── api_client.py         # Client and examples
├── requirements.txt      # Dependencies
├── README.md            # This file
└── models/              # AI model storage
    └── chess_ai_model.pkl
```

### Running Tests
```bash
# Start API server
python chess_api.py

# In another terminal, run client demo
python api_client.py
```

### Adding Features
- Extend `ChessAIEngine` for new analysis features
- Add endpoints in `chess_api.py` for new functionality
- Update `api_client.py` with new client methods

## 🎮 Example Scenarios

### Opening Analysis
```python
# Analyze opening position
opening_position = create_sample_position()
analysis = client.analyze_position(opening_position)
print(f"Opening evaluation: {analysis['evaluation_text']}")
```

### Mid-Game Tactics
```python
# Get tactical recommendations
recommendation = client.recommend_move(tactical_position, "white")
tactics = recommendation['best_move']['analysis']['tactical_elements']
print(f"Tactical elements: {tactics}")
```

### Endgame Precision
```python
# Analyze endgame positions
endgame_rec = client.recommend_move(endgame_position, "white")
print(f"Endgame best move: {endgame_rec['best_move']['move']}")
```

## 🚀 Performance

- **Move Generation**: ~20 candidate moves per position
- **Evaluation Speed**: <100ms per position
- **Learning**: Automatic after 50+ positions
- **Memory**: Efficient caching and model storage

## 🎓 Training the AI

### Quick Training
```bash
# Simple training with sample data
python3 simple_trainer.py

# Online training with real games
python3 online_trainer.py
```

### Master Training Interface
```bash
# Interactive training with multiple options
python3 master_trainer.py
```

**Training Options:**
- 🌐 **Online Training**: Fetch real games from Lichess
- 📊 **Sample Training**: Quick training with generated data
- 🔄 **Continuous Training**: Multiple training rounds
- 📈 **Model Statistics**: View model performance metrics
- 🧪 **Model Testing**: Test current model capabilities

### Training Data Sources
- **Lichess API**: Real tournament and casual games
- **Sample Games**: Generated training positions
- **Custom Games**: Submit your own game data via API

## 🔮 Future Enhancements

- Opening book integration
- Endgame tablebase support
- Multi-threading for faster analysis
- Advanced tactical pattern recognition
- Tournament-strength play levels
- Large-scale database training

---

**The AI continuously learns and improves from every game!** 🧠♟️