# Rocket League Neural Index

## [Latent Goal-Probability State](https://www.kaggle.com/models/owensingh38/latent-goal-probability-state)

Given high-resolution information on the position of a Rocket League replay frame in play that occurs in between goals, how accurately can we identify its "latent goal-probability state"; that is, the underlying probabilities of one or neither team scoring within some period of time that describe the current "momentum" of the game.

Training Data: 
Rocket League replays collected from the following seasons and parsed with the AnalyzeRL Boxcars library:
- 2021-22
- 2022-23
- 2024
- 2025

Testing Data:
Rocket League replays collected from the following seasons and parsed with the AnalyzeRL Boxcars library:
- 2026

Model Metrics, Out-of-Sample Test Data:
- Log Loss (Collapsed, Three-Way): 0.407322 (20.1% skill)
- Brier Score: 0.209407 (18.0% skill)
- Macro ROC-AUC: 0.768225 (53.6% skill)
- Macro PR-AUC: 0.569067 (35.4% skill)

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
    8 × LGPSTransformerBlock(
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
      embed_dim=128,
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
)
```

### Example Inference

```python
from analyzerl_boxcars import parse_replay

REPLAY, CHECKPOINT, device = "/kaggle/input/replay/game.replay", "/kaggle/input/lgps-model/lgps_transformer.pt", torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
