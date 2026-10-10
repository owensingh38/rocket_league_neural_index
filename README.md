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

Outputs from replay parsing yield data in 30 frames per second, or 30hz.  For each replay in these seasons, the full replay frames dataset is first downsampled into 10hz (hinged on the frames directly before a goal is scored) and then is reduced into the following "valid" training frames:

- Frame that occurs during an "active" (mid-play) frame state
- Frame occuring in a game-state in which neither team is winning by more than three goals
- Frame occuring in the game-state in which there is at most three players on the field for a single team

Replay Files were collected from the following sources:
- https://www.kaggle.com/datasets/dster/rocket-league-rlcs-season-2021-22-replays
- https://www.kaggle.com/datasets/dster/rocket-league-rlcs-season-2022-23-replays
- https://www.kaggle.com/datasets/dster/rocket-league-rlcs-season-2024-replays
- https://www.kaggle.com/datasets/dster/rocket-league-rlcs-season-2025-replays
- https://ballchasing.com

## [Latent Goal-Probability State](https://www.kaggle.com/models/owensingh38/latent-goal-probability-state)

### Estimating the latent quantification of the current momentum of Rocket League matches by frame.

Given high-resolution information on the position of a Rocket League replay frame in play that occurs in between goals, how accurately can we identify its "latent goal-probability state"; that is, the underlying probabilities of one or neither team scoring within some period of time that describe the current "momentum" of the game.

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

LGPS training now includes future ball and player states at configurable sample-frame
offsets and the next touch player and time. The default offsets `(10, 30, 60, 100)`
correspond to 1, 3, 6, and 10 seconds at the default 10 Hz sample rate. Set
`LGPSModelConfig(sample_rate=..., future_frames=[...])` to choose another rate or
set of offsets, and prepare source replays with the same `target_sample_rate`.
The source Parquet must contain `frame_touch_player_slot`; existing prepared
Parquet files need to be regenerated with `lgps.dataset.prepare_dataset`.
Training combines the goal and auxiliary losses with gradient balancing; validation
and test losses remain goal-only. `infer_lgps(..., include_aux=True)` adds decoded
future ball and player states at each trained offset, plus next-touch player and
timing predictions, while preserving the goal probability columns. The original
PyTorch checkpoint with auxiliary weights is required; TorchScript exports are
goal-only. These are direct horizon predictions from each input frame, not an
autoregressive rollout.

```python
from inference.rlni_inference import infer_lgps

predictions = infer_lgps(
    "data/frames",
    checkpoint="results/lgps/lgps_transformer.pt",
    include_aux=True,
)
```

Auxiliary output columns use `predicted_ball_{offset}_{feature}` and
`predicted_player_{slot}_{offset}_{feature}`. The offset is in sampled frames
(10, 30, 60, or 100 by default); player slots follow the dataset's canonical
blue-then-orange ordering. Next-touch output includes the selected canonical
slot, its probability and predicted time in seconds, and per-slot probability
and time columns. The original head values are retained in companion
`*_head_{feature}` columns; for player binary features, inference also returns
per-feature probability columns. Decoded quaternion outputs are normalized and
binary state columns are thresholded, so use the head and probability columns
when those details matter. Per-player next-touch logits are also included. This
makes the heads available for trajectory experiments; state predictions at an
offset can be inspected or used as inputs to a separate rollout procedure.

```python
import torch
from analyzerl_boxcars import parse_replay
from lgps.dataset import EntityTokenizer, prepare_replay
from lgps.model import LGPSTransformer, load_lgps_weights
from lgps.config import LGPSModelConfig
from lgps.targets import collapse_probabilities

REPLAY = "/kaggle/input/replay/game.replay"
CHECKPOINT = "/kaggle/input/lgps-model/lgps_transformer.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

frames = parse_replay(REPLAY, event_tagging=False)
checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)

model_config = LGPSModelConfig(**checkpoint["model_config"])
frames = prepare_replay(frames, random_augmentation=False,
                        target_sample_rate=model_config.sample_rate, inference=True)
model = LGPSTransformer(config=model_config).to(device).eval()

state_dict = {
    key.removeprefix("_orig_mod."): value
    for key, value in checkpoint["model_state_dict"].items()
}
load_lgps_weights(model, state_dict)

tokenizer = EntityTokenizer(model_entities=model_config.model_entities)
batch = {
    key: value.to(device)
    for key, value in tokenizer(frames).items()
}

with torch.inference_mode():
    logits = model(
        batch["ball"],
        batch["players"],
        batch["goals"],
        batch["entity_types"],
        batch["entity_teams"],
        batch["game_context"],
        batch.get("big_boosts"),
        batch.get("small_boosts"),
    )

probabilities = collapse_probabilities(
    logits.softmax(dim=-1), model_config.targets_per_team
).cpu().numpy()
```

### Example In-Game Probabilities
![LGPS Goal Probabilities](https://cdn.discordapp.com/attachments/1554343458339295272/1554421273189421057/lgps_goal_probabilities.png?ex=6ac0c79f&is=6abf761f&hm=5223f8d773fe31ddfa6b211031ba81906096a6462385ee868c9cba3fe7ccaac4&)

## References
- [Rocket League Goal Predictor 2025](https://www.kaggle.com/code/dster/rocket-league-goal-predictor-2025-model/notebook)
- [A Framework for the Fine-Grained Evaluation of the Instantaneous Expected Value of Soccer Possessions](https://arxiv.org/abs/2011.09426)
- [GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks](https://arxiv.org/abs/1711.02257)

## License
Apache License 2.0. See [LICENSE](LICENSE).
