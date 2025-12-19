# Chess AI Engine with Stockfish Integration

A powerful chess engine API that integrates Stockfish for world-class chess AI, with optional neural network training capabilities.

## Features

✅ **Stockfish Integration** - World's strongest chess engine (3500+ ELO)  
✅ **Neural Engine** - Trainable AI that learns from chess games  
✅ **REST API** - Easy integration with web frontends  
✅ **Incremental Learning** - Neural engine improves with each training session  
✅ **CORS Enabled** - Works with web applications  

## Quick Start

### 1. Install Dependencies
```bash
pip3 install chess flask flask-cors numpy requests
```

### 2. Download Stockfish
```bash
mkdir stockfish
cd stockfish
# Download for your platform from: https://stockfishchess.org/download/
# Rename binary to: stockfish-macos-m1-apple-silicon
chmod +x stockfish-macos-m1-apple-silicon
```

### 3. Start API Server
```bash
python3 stockfish_only_api.py
```

### 4. Test API
```bash
curl -X POST http://localhost:5001/api/recommend-move \
  -H "Content-Type: application/json" \
  -d '{
    "position": {
      "e1": {"type": "king", "color": "white"},
      "e8": {"type": "king", "color": "black"},
      "e2": {"type": "pawn", "color": "white"}
    },
    "current_player": "white"
  }'
```

## API Endpoints

### POST `/api/recommend-move`
Get best move recommendation from Stockfish.

**Request:**
```json
{
  "position": {
    "e1": {"type": "king", "color": "white"},
    "e8": {"type": "king", "color": "black"}
  },
  "current_player": "white",
  "num_alternatives": 3
}
```

**Response:**
```json
{
  "best_move": {
    "move": {"from": "e2", "to": "e4", "piece": "p", "color": "white"},
    "score": 0.3,
    "confidence": "high"
  },
  "engine": "STOCKFISH_ONLY"
}
```

### GET `/api/health`
Check API status.

## Neural Engine Training

### Train on Chess Games
```bash
python3 smart_trainer.py
```

### Fast Training (for testing)
```bash
python3 fast_trainer.py
```

## Files

| File | Purpose |
|------|---------|
| `stockfish_only_api.py` | **Main API** - Stockfish-only engine |
| `stockfish_engine.py` | Stockfish wrapper class |
| `neural_chess_engine.py` | Neural network engine |
| `smart_trainer.py` | Downloads & trains on massive chess data |
| `fast_trainer.py` | Quick training for testing |
| `chess_model.pkl` | Trained neural network weights |

## Engine Comparison

| Engine | Strength | Learning | Dependencies |
|--------|----------|----------|--------------|
| **Stockfish** | 3500+ ELO | ❌ | Binary required |
| **Neural** | ~1200 ELO | ✅ | Python only |

## Configuration

### Stockfish Path
Update path in `stockfish_only_api.py`:
```python
stockfish_path = "./stockfish/your-stockfish-binary"
```

### API Port
Change port in startup:
```python
app.run(host='0.0.0.0', port=5001)
```

## Troubleshooting

### CORS Errors
API includes CORS headers. If issues persist:
```python
from flask_cors import CORS
CORS(app, origins=["http://localhost:3000"])
```

### Stockfish Not Found
```bash
# Check binary exists and is executable
ls -la stockfish/
chmod +x stockfish/stockfish-*
```

### API Not Responding
```bash
# Check if API is running
curl http://localhost:5001/api/health

# Check logs for errors
python3 stockfish_only_api.py
```

## Performance

- **Stockfish**: ~0.1s per move (depth 15+)
- **Neural**: ~0.01s per move (instant)
- **Training**: 10-20x faster with optimizations

## License

MIT License - Use freely for personal and commercial projects.

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Submit pull request

## Support

For issues or questions:
- Check troubleshooting section
- Review API logs
- Test with curl commands