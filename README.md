# Rocket League Neural Index

Training Data: 
Rocket League replays collected from the following seasons and parsed with the [AnalyzeRL Boxcars library](https://pypi.org/project/analyzerl-boxcars):
- 2021-22
- 2022-23
- 2024
- 2025

Testing Data:
Rocket League replays collected from the following seasons and parsed with the AnalyzeRL Boxcars library:
- 2026

Replay Files were collected from the following sources:
- https://www.kaggle.com/datasets/dster/rocket-league-rlcs-season-2021-22-replays
- https://www.kaggle.com/datasets/dster/rocket-league-rlcs-season-2022-23-replays
- https://www.kaggle.com/datasets/dster/rocket-league-rlcs-season-2024-replays
- https://www.kaggle.com/datasets/dster/rocket-league-rlcs-season-2025-replays
- https://ballchasing.com

## [Latent Goal-Probability State](https://www.kaggle.com/models/owensingh38/latent-goal-probability-state)

### Estimating the latent quantification of the current momentum of Rocket League matches by frame.

Given high-resolution information on the position of a Rocket League replay frame in play that occurs in between goals, how accurately can we identify its "latent goal-probability state"; that is, the underlying probabilities of one or neither team scoring within some period of time that describe the current "momentum" of the game.

Outputs from replay parsing yield data in 30 frames per second, or 30hz.  For each replay in these seasons, the full replay frames dataset is first downsampled into 10hz (hinged on the frames directly before a goal is scored) and then is reduced into the following "valid" training frames:

- Frame that occurs during an "active" (mid-play) frame state
- Frame occuring in a game-state in which neither team is winning by more than three goals
- Frame occuring in the game-state in which there is at most three players on the field for a single team

Model Metrics, Out-of-Sample Test Data:
- Log Loss (Collapsed, Three-Way): 0.402510
- Brier Score: 0.206863
- Macro ROC-AUC: 0.773212
- Macro PR-AUC: 0.577936

### Architecture

```
LGPSTransformer(
  (ball_stream): LGPSAgent(
    (network): Sequential(
      (0): Linear(13 → 128)
      (1): GELU
      (2): Linear(128 → 128)
      (3): LayerNorm(128)
    )
  )
  (player_stream): LGPSAgent(
    (network): Sequential(
      (0): Linear(58 → 128)
      (1): GELU
      (2): Linear(128 → 128)
      (3): LayerNorm(128)
    )
  )
  (goal_stream): LGPSAgent(
    (network): Sequential(
      (0): Linear(9 → 128)
      (1): GELU
      (2): Linear(128 → 128)
      (3): LayerNorm(128)
    )
  )
  (big_boost_stream): LGPSAgent(
    (network): Sequential(
      (0): Linear(6 → 128)
      (1): GELU
      (2): Linear(128 → 128)
      (3): LayerNorm(128)
    )
  )
  (context_token): LGPSAgent(
    (network): Sequential(
      (0): Linear(4 → 128)
      (1): GELU
      (2): Linear(128 → 128)
      (3): LayerNorm(128)
    )
  )

  (type_embedding): Embedding(4, 128)
  (team_embedding): Embedding(3, 128)
  (geometry): RelationalGeometry()

  (blocks): ModuleList(
    4 × LGPSTransformerBlock(
      (norm1): LayerNorm(128)
      (attention): LGPSAttention(
        (q): Linear(128 → 128)
        (k): Linear(128 → 128)
        (v): Linear(128 → 128)
        (out): Linear(128 → 128)

        (geometry_bias): Sequential(
          Linear(156 → 128)
          GELU
          Linear(128 → 8)
        )

        (pairwise_source): Linear(128 → 256)
        (pairwise_target): Linear(128 → 256)
        (pairwise_source_role): Linear(13 → 8)
        (pairwise_target_role): Linear(13 → 8)

        (pad_geometry_bias): Sequential(
          Linear(3 → 128)
          GELU
          Linear(128 → 8)
        )

        (dropout): Dropout(p=0.05)
      )

      (norm2): LayerNorm(128)
      (ffn): Sequential(
        Linear(128 → 256)
        GELU
        Dropout(p=0.05)
        Linear(256 → 128)
        Dropout(p=0.05)
      )
    )
  )

  (pool): LearnedQueryPooling(
    (norm): LayerNorm(128)
    (attention): MultiheadAttention(
      embed_dim=128
      num_heads=8
    )
  )

  (head): Sequential(
    LayerNorm(256)
    Linear(256 → 128)
    GELU
    Dropout(p=0.05)
    Linear(128 → 21)
  )

  (ball_state_aux): LGPSBallStateAux(
    (heads): ModuleDict(
      (10): Sequential(LayerNorm(128), Linear(128 → 128), GELU, Linear(128 → 13))
      (30): Sequential(LayerNorm(128), Linear(128 → 128), GELU, Linear(128 → 13))
      (60): Sequential(LayerNorm(128), Linear(128 → 128), GELU, Linear(128 → 13))
      (100): Sequential(LayerNorm(128), Linear(128 → 128), GELU, Linear(128 → 13))
    )
  )

  (player_state_aux): LGPSPlayerStateAux(
    (heads): ModuleDict(
      (10): Sequential(LayerNorm(256), Linear(256 → 128), GELU, Linear(128 → 29))
      (30): Sequential(LayerNorm(256), Linear(256 → 128), GELU, Linear(128 → 29))
      (60): Sequential(LayerNorm(256), Linear(256 → 128), GELU, Linear(128 → 29))
      (100): Sequential(LayerNorm(256), Linear(256 → 128), GELU, Linear(128 → 29))
    )
  )

  (next_touch_aux): LGPSNextTouchAux(
    (player_score): Linear(128 → 1)
    (global_score): Linear(128 → 128)
    (player_time): Linear(128 → 1)
  )
)
```

### Example Inference

```python
from analyzerl_boxcars import parse_replay

REPLAY = "data/frames/game.replay"
CHECKPOINT = "data/models/lgps/lgps_transformer.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

frames = parse_replay(REPLAY, event_tagging=False)

checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)

model_config = LGPSModelConfig(**checkpoint["model_config"])
model = create_model(model_config, train_config).to(device).eval(); 

model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in 
checkpoint["model_state_dict"].items()})

batch = {key: value.to(device) for key, value in 
EntityTokenizer(model_entities=model_config.model_entities)(frames).items()}

with torch.inference_mode():
    logits = model(batch["ball"], batch["players"], batch["goals"], batch["entity_types"], batch["entity_teams"], batch["game_context"], batch.get("big_boosts"), batch.get("small_boosts"))

probabilities = collapse_probabilities(logits.softmax(-1), model_config.targets_per_team).cpu().numpy()
```

### Example In-Game Probabilities
![LGPS Goal Probabilities](https://cdn.discordapp.com/attachments/1554343458339295272/1554421273189421057/lgps_goal_probabilities.png?ex=6ac0c79f&is=6abf761f&hm=5223f8d773fe31ddfa6b211031ba81906096a6462385ee868c9cba3fe7ccaac4&)

## [Expected Goals](https://www.kaggle.com/models/owensingh38/rl-expected-goals)

Given information of a frame in which a player initiates a shot attempt (based on in-game shot recording), can we estimate the likelihood the shot attempt will eventually score?

Using the frame that a shot is recorded in replay frame data, the touch initiating the shot is found by looking back and marking the last touch made by the shooting player as the shooting frame, which is where information on the shot is dervied from for the model.  

Because shot and goal events are not necessarily linked in replay data, this must be done heuristically.  These events are linked by looking back 120 frames (four seconds in our 30hz replay frames data) from the frame the goal is recorded and identifying an evaluated shot row in that interval, selecting the most recent shot by the goal scorer.

Model Metrics, Out-of-Sample Test Data:
Model Metrics, Out-of-Sample Test Data:
- Log Loss: 0.278642
- Brier Score: 0.083843
- ROC-AUC: 0.901652

### Example Inference

```python
import joblib
import polars as pl
from pathlib import Path
from tempfile import TemporaryDirectory
from analyzerl_boxcars import parse_replay
from xg.dataset import load_training_frames, prepare_model_data, to_sparse_matrix

REPLAY = "data/frames/game.replay"
artifact = joblib.load("results/xg/standard/xg_model_standard.joblib")

with TemporaryDirectory() as directory:
    parse_replay(REPLAY, event_tagging=True).write_parquet(Path(directory) / "game_frames.parquet")
    shots = load_training_frames(
        directory, model_segments=artifact["model_segment"], include_event_context=False
    )

features = prepare_model_data(shots)
xg = artifact["model"].predict_proba(
    to_sparse_matrix(features, artifact["feature_columns"])
)[:, 1]

scored_shots = shots.with_columns(pl.Series("xG", xg))
```

## References
- [Rocket League Goal Predictor 2025](https://www.kaggle.com/code/dster/rocket-league-goal-predictor-2025-model/notebook)
- [A Framework for the Fine-Grained Evaluation of the Instantaneous Expected Value of Soccer Possessions](https://arxiv.org/abs/2011.09426)
- [GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks](https://arxiv.org/abs/1711.02257)

## License
Apache License 2.0. See [LICENSE](LICENSE).
