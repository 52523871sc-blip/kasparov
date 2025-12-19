# /api/analyze-position - Sample Request and Response

## Sample Request

```json
POST /api/analyze-position
Content-Type: application/json

{
  "position": {
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
  },
  "user_move": {
    "from": "e2",
    "to": "e4",
    "piece": "pawn"
  },
  "current_player": "white",
  "include_demo": true
}
```

## Sample Response

```json
{
  "user_move_evaluation": {
    "move": {
      "from": "e2",
      "to": "e4",
      "piece": "pawn"
    },
    "score": 0.15,
    "rank": 1,
    "is_best_move": true,
    "is_top_3": true,
    "score_difference_from_best": 0.0
  },
  "comprehensive_analysis": {
    "move_quality_assessment": "Excellent! You found the best move according to the engine.",
    "strategic_analysis": [
      "Center Control: This move influences central squares, which is strategically important."
    ],
    "tactical_analysis": [],
    "positional_consequences": [
      "Pawn Structure: Pawn moves are permanent and affect long-term pawn structure.",
      "Piece Coordination: Consider how this move affects coordination with other pieces.",
      "Space Control: Evaluate how this move affects your control of key squares."
    ],
    "alternative_comparison": "Engine's top choice: Pawn from e2 to e4 (Score: 0.15)\nYour move: Pawn from e2 to e4 (Score: 0.15)\nPerfect! You chose the engine's top recommendation.",
    "learning_insights": [
      "Excellent pattern recognition! You identified the strongest continuation.",
      "Study why this move is superior to understand similar positions.",
      "Practice similar positions to improve pattern recognition.",
      "Analyze master games with comparable pawn structures."
    ],
    "improvement_recommendations": [
      "Evaluate candidate moves systematically using a consistent method.",
      "Study tactical patterns to improve move selection.",
      "Practice endgame positions to understand piece values better."
    ]
  },
  "engine_alternatives": [
    {
      "move": {
        "from": "e2",
        "to": "e4",
        "piece": "pawn"
      },
      "score": 0.15,
      "rank": 1,
      "comparison_with_user_move": "Approximately equal strength"
    },
    {
      "move": {
        "from": "d2",
        "to": "d4",
        "piece": "pawn"
      },
      "score": 0.12,
      "rank": 2,
      "comparison_with_user_move": "Your move is actually stronger"
    },
    {
      "move": {
        "from": "g1",
        "to": "f3",
        "piece": "knight"
      },
      "score": 0.08,
      "rank": 3,
      "comparison_with_user_move": "Your move is actually stronger"
    }
  ],
  "demo_scripts": {
    "backtrack_analysis": {
      "title": "Move Consequence Analysis",
      "description": "Let's trace the consequences of Pawn from e2 to e4",
      "steps": [
        "1. Identify immediate tactical consequences",
        "2. Evaluate positional changes",
        "3. Consider opponent's best responses",
        "4. Assess resulting position"
      ],
      "key_questions": [
        "Does this move improve piece activity?",
        "Are there any tactical vulnerabilities created?",
        "How does this affect pawn structure?",
        "What are the opponent's main threats after this move?"
      ]
    },
    "alternative_demonstrations": [
      {
        "alternative_number": 1,
        "move_description": "Pawn from e2 to e4",
        "demo_sequence": [
          "Play Pawn from e2 to e4",
          "Observe immediate consequences",
          "Analyze opponent's likely responses",
          "Compare resulting positions"
        ],
        "key_benefits": [
          "Provides significant advantage",
          "Improves piece coordination",
          "Maintains strategic flexibility"
        ],
        "when_to_prefer": "Consider this move when seeking active piece play and maintaining initiative."
      },
      {
        "alternative_number": 2,
        "move_description": "Pawn from d2 to d4",
        "demo_sequence": [
          "Play Pawn from d2 to d4",
          "Observe immediate consequences",
          "Analyze opponent's likely responses",
          "Compare resulting positions"
        ],
        "key_benefits": [
          "Improves piece coordination",
          "Maintains strategic flexibility"
        ],
        "when_to_prefer": "Consider this move when seeking active piece play and maintaining initiative."
      },
      {
        "alternative_number": 3,
        "move_description": "Knight from g1 to f3",
        "demo_sequence": [
          "Play Knight from g1 to f3",
          "Observe immediate consequences",
          "Analyze opponent's likely responses",
          "Compare resulting positions"
        ],
        "key_benefits": [
          "Improves piece coordination",
          "Maintains strategic flexibility"
        ],
        "when_to_prefer": "Consider this move when seeking active piece play and maintaining initiative."
      }
    ],
    "tactical_variations": {
      "tactical_themes": [
        "Pin and Skewer Opportunities",
        "Fork and Double Attack Patterns",
        "Discovered Attack Possibilities",
        "Deflection and Decoy Tactics"
      ],
      "demonstration_steps": [
        "Set up the position after your move",
        "Look for tactical patterns",
        "Calculate forcing variations",
        "Compare with engine alternatives"
      ],
      "practice_exercises": [
        "Find all checks in the resulting position",
        "Identify all possible captures",
        "Look for piece coordination improvements",
        "Assess king safety for both sides"
      ]
    },
    "strategic_comparisons": {
      "strategic_comparison": {
        "user_move_strategy": "Pawn advance for space and center control",
        "engine_move_strategy": "Pawn advance for space and center control",
        "strategic_differences": "Your approach: Pawn advance for space and center control. Engine approach: Pawn advance for space and center control."
      },
      "demonstration_plan": [
        "Analyze pawn structure implications",
        "Evaluate piece activity changes",
        "Consider long-term strategic goals",
        "Compare resulting imbalances"
      ]
    }
  },
  "timestamp": "2024-01-15T14:30:45.123456"
}
```

## Example: Suboptimal Move Analysis

### Request with a weaker move:

```json
{
  "position": { /* same starting position */ },
  "user_move": {
    "from": "a2",
    "to": "a3",
    "piece": "pawn"
  },
  "current_player": "white",
  "include_demo": true
}
```

### Response (excerpt):

```json
{
  "user_move_evaluation": {
    "move": {
      "from": "a2",
      "to": "a3",
      "piece": "pawn"
    },
    "score": -0.05,
    "rank": 8,
    "is_best_move": false,
    "is_top_3": false,
    "score_difference_from_best": -0.20
  },
  "comprehensive_analysis": {
    "move_quality_assessment": "Decent move. Ranked #8 with 0.20 points difference. Still playable but not optimal.",
    "strategic_analysis": [
      "Center Control: This move influences central squares, which is strategically important."
    ],
    "tactical_analysis": [],
    "positional_consequences": [
      "Pawn Structure: Pawn moves are permanent and affect long-term pawn structure.",
      "Piece Coordination: Consider how this move affects coordination with other pieces.",
      "Space Control: Evaluate how this move affects your control of key squares."
    ],
    "alternative_comparison": "Engine's top choice: Pawn from e2 to e4 (Score: 0.15)\nYour move: Pawn from a2 to a3 (Score: -0.05)\nDifference: 0.20 points. Significant difference, the engine's choice is clearly superior.",
    "learning_insights": [
      "This position requires deeper analysis. Consider all candidate moves systematically.",
      "Focus on tactical awareness and positional understanding.",
      "Practice similar positions to improve pattern recognition.",
      "Analyze master games with comparable pawn structures."
    ],
    "improvement_recommendations": [
      "Calculate 2-3 moves deeper before deciding.",
      "Consider all forcing moves (checks, captures, threats) first.",
      "Evaluate candidate moves systematically using a consistent method.",
      "Study tactical patterns to improve move selection.",
      "Practice endgame positions to understand piece values better."
    ]
  },
  "engine_alternatives": [
    {
      "move": {
        "from": "e2",
        "to": "e4",
        "piece": "pawn"
      },
      "score": 0.15,
      "rank": 1,
      "comparison_with_user_move": "Significantly stronger alternative"
    },
    {
      "move": {
        "from": "d2",
        "to": "d4",
        "piece": "pawn"
      },
      "score": 0.12,
      "rank": 2,
      "comparison_with_user_move": "Significantly stronger alternative"
    },
    {
      "move": {
        "from": "g1",
        "to": "f3",
        "piece": "knight"
      },
      "score": 0.08,
      "rank": 3,
      "comparison_with_user_move": "Moderately better option"
    }
  ],
  "demo_scripts": {
    /* Full demo scripts as shown above */
  }
}
```

## Key Features

1. **User Move Evaluation**: Ranks the user's move against engine recommendations
2. **Comprehensive Analysis**: Multi-perspective analysis including:
   - Move quality assessment
   - Strategic analysis
   - Tactical analysis
   - Positional consequences
   - Alternative comparison
   - Learning insights
   - Improvement recommendations

3. **Engine Alternatives**: Top 3 engine moves with comparison to user move

4. **Demo Scripts** (when `include_demo: true`):
   - **Backtrack Analysis**: Step-by-step consequence analysis
   - **Alternative Demonstrations**: Detailed scripts for each alternative move
   - **Tactical Variations**: Tactical themes and practice exercises
   - **Strategic Comparisons**: Strategic approach comparisons

## Use Cases

- **Training Mode**: Students can submit their moves and receive detailed feedback
- **Game Analysis**: Post-game review with comprehensive move-by-move analysis
- **Interactive Learning**: Demo scripts guide users through alternative variations
- **Skill Assessment**: Evaluate player strength based on move quality
