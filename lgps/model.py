import torch
import torch.nn as nn
import torch.nn.functional as F
from props import BALL_FEATURES, CONTEXT_FEATURES, GEOMETRY_DIM, GOAL_FEATURES, LGPSModelConfig, PLAYER_FEATURES, normalize_model_entities
from .dataset import LGPSScaler

torch.set_float32_matmul_precision("high")

def _stack_entities(values):
    if isinstance(values, torch.Tensor):
        return values
    return torch.stack(values, dim=1)


class LGPSAgent(nn.Module):
    def __init__(self, input_dim, d_model, *, missing_aware=False):
        super().__init__()
        self.missing_aware = bool(missing_aware)
        projected_dim = input_dim * 2 if self.missing_aware else input_dim

        # Every physical agent is independently projected into the common
        # latent agent space used by the interaction transformer.
        self.network = nn.Sequential(
            nn.Linear(projected_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x):
        if self.missing_aware:
            missing = ~torch.isfinite(x)
            x = torch.cat(
                (
                    torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
                    missing.to(x.dtype),
                ),
                dim=-1,
            )
        return self.network(x)


class PairwiseGeometry(nn.Module):
    def geometry_states(self, ball, players, goals):
        # Geometry only needs position and velocity.
        # Player orientation and other state still remain available in each
        # player's learned stream representation.
        ball = ball[..., :6]
        players = _stack_entities(players)[..., :6]
        goals = _stack_entities(goals)

        # Goals are stationary entities, so their velocity is always zero.
        goal_states = torch.cat(
            (goals[..., :3], torch.zeros_like(goals[..., :3])),
            dim=-1,
        )

        # Shape: [batch, 9 agents, 6 physical geometry features]
        return torch.cat(
            (ball.unsqueeze(1), players, goal_states),
            dim=1,
        )

    def forward(self, ball, players, goals):
        x = self.geometry_states(ball, players, goals)
        valid = torch.isfinite(x).all(dim=-1)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        position = x[..., :3]
        velocity = x[..., 3:6]

        # Broadcasting creates every ordered source -> target pair.
        source_position = position.unsqueeze(2)
        target_position = position.unsqueeze(1)
        source_velocity = velocity.unsqueeze(2)
        target_velocity = velocity.unsqueeze(1)

        relative_position = target_position - source_position
        relative_velocity = target_velocity - source_velocity

        # Scalar pairwise geometry.
        distance = torch.linalg.vector_norm(
            relative_position,
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)

        relative_speed = torch.linalg.vector_norm(
            relative_velocity,
            dim=-1,
            keepdim=True,
        )

        direction = relative_position / distance

        closing_speed = -(
            relative_velocity * direction
        ).sum(
            dim=-1,
            keepdim=True,
        )

        source_speed_toward_target = (
            source_velocity * direction
        ).sum(
            dim=-1,
            keepdim=True,
        )

        target_speed_toward_source = -(
            target_velocity * direction
        ).sum(
            dim=-1,
            keepdim=True,
        )

        velocity_alignment = F.cosine_similarity(
            source_velocity.expand_as(relative_position),
            target_velocity.expand_as(relative_position),
            dim=-1,
            eps=1e-6,
        ).unsqueeze(-1)

        angle_xy = torch.atan2(
            relative_position[..., 1],
            relative_position[..., 0],
        ).unsqueeze(-1)
        angle_xz = torch.atan2(
            relative_position[..., 2],
            relative_position[..., 0],
        ).unsqueeze(-1)
        angle_yz = torch.atan2(
            relative_position[..., 2],
            relative_position[..., 1],
        ).unsqueeze(-1)

        # A demolished player has no physical location. Suppress geometry for
        # every edge incident to that entity; its player stream still carries
        # demoed, on_field, respawn_timer, and per-feature missingness.
        valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)

        # Shape: [batch, 9, 9, 15]
        geometry = torch.cat(
            (
                relative_position,
                relative_velocity,
                distance,
                relative_speed,
                closing_speed,
                source_speed_toward_target,
                target_speed_toward_source,
                velocity_alignment,
                angle_xy,
                angle_xz,
                angle_yz,
            ),
            dim=-1,
        )
        geometry = torch.where(
            valid_pair.unsqueeze(-1), geometry, torch.zeros_like(geometry)
        )

        # Keep physical magnitudes in a numerically stable range for AMP.
        return LGPSScaler.geometry(geometry)


# Extra features are edge-local. Each block has its own input weights in
# geometry_bias; no engineered feature is appended to an entity stream.
RELATION_NAMES = (
    "player_ball", "player_own_goal", "player_opposing_goal",
    "player_teammate", "player_opponent",
    "player_nearest_teammate", "player_nearest_opponent",
)
RELATION_FEATURES = (
    "distance", "center_xy", "center_xz", "center_yz",
    "nose_xy", "nose_xz", "nose_yz",
    "center_xy_sin", "center_xz_sin", "center_yz_sin",
    "center_xy_cos", "center_xz_cos", "center_yz_cos",
    "nose_xy_sin", "nose_xz_sin", "nose_yz_sin",
    "nose_xy_cos", "nose_xz_cos", "nose_yz_cos",
)
# Existing 15 + seven 19-channel relationships + two ball-nearest 4-channel groups.

class RelationalGeometry(PairwiseGeometry):
    def __init__(self, geometry_dim=GEOMETRY_DIM, pad_count=34):
        super().__init__()
        if geometry_dim not in (15, GEOMETRY_DIM):
            raise ValueError("geometry_dim must be 15 or 156")
        self.geometry_dim = geometry_dim
        self.pad_count = pad_count
        self.include_boosts = pad_count > 0

    def forward(self, ball, players, goals, boosts, entity_teams):
        core = super().forward(ball, players, goals)
        batch = ball.shape[0]
        # Do not allocate the 43-token boost layout when boosts are excluded.
        # The selected no-boost model is exactly a nine-token model.
        entity_count = 9 + self.pad_count
        geometry = core.new_zeros(batch, entity_count, entity_count, self.geometry_dim)
        geometry[:, :9, :9, :15] = core
        states = self.geometry_states(ball, players, goals)
        pos = torch.nan_to_num(states[..., :3])
        live = (players[..., PLAYER_FEATURES.index("on_field")] > 0.5)
        live &= ~(players[..., PLAYER_FEATURES.index("demoed")] > 0.5)
        live &= torch.isfinite(players[..., :3]).all(-1)
        valid = torch.cat((torch.ones(batch, 1, device=ball.device, dtype=torch.bool),
                           live, torch.ones(batch, 2, device=ball.device, dtype=torch.bool)), 1)
        geometry[:, :9, :9, :15] *= (valid[:, :, None] & valid[:, None, :])[..., None]
        if self.geometry_dim == 15:
            pad_geometry = core.new_zeros(batch, entity_count, entity_count, 3)
            allowed = torch.zeros(batch, entity_count, entity_count, device=ball.device, dtype=torch.bool)
            allowed[:, :9, :9] = True
            pad_edges = torch.zeros_like(allowed)
            if self.include_boosts:
                pad_delta = boosts[:, None, :, :2] - pos[:, 1:7, None, :2]
                pad_position = torch.cat((pad_delta / 5000.0,
                                          pad_delta.norm(dim=-1, keepdim=True)/5000.0), -1)
                pad_geometry[:, 1:7, 9:] = pad_position
                pad_geometry[:, 9:, 1:7] = pad_position.transpose(1, 2) * core.new_tensor([-1., -1., 1.])
                allowed[:, 1:7, 9:] = live[:, :, None]
                allowed[:, 9:, 1:7] = live[:, None, :]
                pad_edges[:, 1:7, 9:] = live[:, :, None]
                pad_edges[:, 9:, 1:7] = live[:, None, :]
            return geometry, pad_geometry, allowed, pad_edges
        q = torch.nan_to_num(players[..., 9:13])
        q = F.normalize(q, dim=-1)
        x, y, z, w = q.unbind(-1)
        forward = torch.stack((1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)), -1)
        nose = pos[:, 1:7] + 59.0 * forward
        delta = pos[:, None, :, :] - pos[:, 1:7, None, :]
        nose_delta = pos[:, None, :, :] - nose[:, :, None, :]

        def angles(v):
            # Avoid undefined atan2 gradients at coincident points.
            safe = torch.where(v.abs().sum(-1, keepdim=True) > 0, v,
                               v.new_tensor([1.0, 0.0, 0.0]))
            return torch.stack((torch.atan2(safe[..., 1], safe[..., 0]),
                                torch.atan2(safe[..., 2], safe[..., 0]),
                                torch.atan2(safe[..., 2], safe[..., 1] + 1e-8)), -1)

        center_a, nose_a = angles(delta), angles(nose_delta)
        relation = torch.cat((delta.norm(dim=-1, keepdim=True)/6000.0,
                              center_a, nose_a, center_a.sin(), center_a.cos(),
                              nose_a.sin(), nose_a.cos()), -1)
        teams = entity_teams[:, 1:7]
        same = teams[:, :, None] == teams[:, None, :]
        identity = torch.eye(6, device=ball.device, dtype=torch.bool)
        masks = torch.zeros(batch, 7, 6, 9, device=ball.device, dtype=torch.bool)
        masks[:, 0, :, 0] = True
        masks[:, 1, :, 7] = teams == 0
        masks[:, 1, :, 8] = teams == 1
        masks[:, 2, :, 7] = teams == 1
        masks[:, 2, :, 8] = teams == 0
        masks[:, 3, :, 1:7] = same & ~identity
        masks[:, 4, :, 1:7] = ~same & (teams[:, :, None] < 2) & (teams[:, None, :] < 2)
        masks &= live[:, None, :, None] & valid[:, None, None, :]
        distance = delta[..., 1:7, :].norm(dim=-1)
        for group, base in ((5, 3), (6, 4)):
            candidates = masks[:, base, :, 1:7]
            distances = distance.masked_fill(~candidates, float("inf"))
            minimum = distances.amin(-1, keepdim=True)
            # Include tied nearest players to preserve permutation equivariance.
            masks[:, group, :, 1:7] = candidates & (distances == minimum)
        for group in range(7):
            start = 15 + group * 19
            geometry[:, 1:7, :9, start:start+19] = torch.where(
                masks[:, group, :, :, None], relation, 0.0)
        ball_delta = pos[:, 1:7] - pos[:, :1]
        ball_rel = torch.cat((ball_delta.norm(dim=-1, keepdim=True)/6000.0,
                              angles(ball_delta)), -1)
        for team in (0, 1):
            candidates = live & (teams == team)
            distances = ball_delta.norm(dim=-1).masked_fill(~candidates, float("inf"))
            nearest = candidates & (distances == distances.amin(-1, keepdim=True))
            start = 148 + team*4
            geometry[:, 0, 1:7, start:start+4] = torch.where(nearest[..., None], ball_rel, 0.0)

        # Pads have only planar positional edges to live players.
        pad_geometry = core.new_zeros(batch, entity_count, entity_count, 3)
        allowed = torch.zeros(batch, entity_count, entity_count, device=ball.device, dtype=torch.bool)
        allowed[:, :9, :9] = True
        pad_edges = torch.zeros_like(allowed)
        if self.include_boosts:
            pad_delta = boosts[:, None, :, :2] - pos[:, 1:7, None, :2]
            pad_position = torch.cat((pad_delta / 5000.0,
                                      pad_delta.norm(dim=-1, keepdim=True)/5000.0), -1)
            pad_geometry[:, 1:7, 9:] = pad_position
            pad_geometry[:, 9:, 1:7] = pad_position.transpose(1, 2) * core.new_tensor([-1., -1., 1.])
            allowed[:, 1:7, 9:] = live[:, :, None]
            allowed[:, 9:, 1:7] = live[:, None, :]
            pad_edges[:, 1:7, 9:] = live[:, :, None]
            pad_edges[:, 9:, 1:7] = live[:, None, :]
        return geometry, pad_geometry, allowed, pad_edges

class LGPSAttention(nn.Module):
    def __init__(self, d_model, num_heads, geometry_dim, pairwise_rank, dropout, include_boosts=True):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if pairwise_rank <= 0:
            raise ValueError("pairwise_rank must be positive")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.pairwise_rank = pairwise_rank

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        # Geometry remains an explicit physical relationship between entities.
        self.geometry_bias = nn.Sequential(
            nn.Linear(geometry_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_heads),
        )

        # Low-rank pairwise state bias avoids materializing a wide
        # [batch, entities, entities, d_model] activation.
        self.pairwise_source = nn.Linear(
            d_model,
            num_heads * pairwise_rank,
        )
        self.pairwise_target = nn.Linear(
            d_model,
            num_heads * pairwise_rank,
        )
        self.pairwise_source_role = nn.Linear(13, num_heads)
        self.pairwise_target_role = nn.Linear(13, num_heads)
        self.pairwise_scale = nn.Parameter(
            torch.full((num_heads,), 0.1)
        )

        # Geometry is a zero-start residual and contributes exactly zero at
        # initialization while the model learns the geometric correction.
        geometry_output = self.geometry_bias[-1]
        nn.init.zeros_(geometry_output.weight)
        nn.init.zeros_(geometry_output.bias)

        self.pad_geometry_bias = (
            nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, num_heads))
            if include_boosts else None
        )
        if self.pad_geometry_bias is not None:
            nn.init.zeros_(self.pad_geometry_bias[-1].weight)
            nn.init.zeros_(self.pad_geometry_bias[-1].bias)
        self.dropout = nn.Dropout(dropout)

    def pairwise_state_features(
        self,
        streams,
        entity_type_features,
        entity_team_features,
        entity_role_features,
    ):
        # Keep this path in the active AMP dtype. The previous FP32 casts
        # duplicated large pairwise tensors and defeated mixed precision.
        # Pairwise logits must remain defined even when a partially observed
        # entity carries NaN/Inf fields. Entity encoders expose missingness in
        # their stream features; this path treats non-finite learned values as
        # a neutral zero contribution instead of contaminating an attention row.
        streams = torch.nan_to_num(streams, nan=0.0, posinf=0.0, neginf=0.0)
        batch_size, entity_count, _ = streams.shape
        source = self.pairwise_source(streams).view(
            batch_size,
            entity_count,
            self.num_heads,
            self.pairwise_rank,
        )
        target = self.pairwise_target(streams).view(
            batch_size,
            entity_count,
            self.num_heads,
            self.pairwise_rank,
        )
        bilinear = torch.einsum(
            "bnhr,bmhr->bhnm",
            source,
            target,
        ) * (self.pairwise_rank ** -0.5)

        semantic = torch.cat(
            (
                entity_type_features,
                entity_team_features,
                entity_role_features,
            ),
            dim=-1,
        ).reshape(batch_size, entity_count, -1)
        source_role = self.pairwise_source_role(semantic).permute(0, 2, 1)
        target_role = self.pairwise_target_role(semantic).permute(0, 2, 1)
        pairwise = (
            bilinear
            + source_role.unsqueeze(-1)
            + target_role.unsqueeze(-2)
        )
        return torch.nan_to_num(
            pairwise
            * self.pairwise_scale.view(1, self.num_heads, 1, 1)
        , nan=0.0, posinf=0.0, neginf=0.0).permute(0, 2, 3, 1)

    def static_geometry_bias(self, geometry, pad_geometry, core_count):
        # Project frame-static geometry once per shared-weight block.
        #
        #         A transformer block is applied ``loops_per_block`` times, but its
        #         geometry and its geometry-bias weights do not change between those
        #         applications. Caching this projection is algebraically identical to
        #         recomputing it on every loop while avoiding the dominant repeated MLP.
        #         
        geometry = torch.nan_to_num(geometry, nan=0.0, posinf=0.0, neginf=0.0)
        pad_geometry = torch.nan_to_num(pad_geometry, nan=0.0, posinf=0.0, neginf=0.0)
        batch_size, entity_count = geometry.shape[:2]
        core_bias = self.geometry_bias(geometry[:, :core_count, :core_count])
        edge_bias = core_bias.new_zeros(batch_size, entity_count, entity_count, self.num_heads)
        edge_bias[:, :core_count, :core_count] = core_bias
        if entity_count > core_count:
            assert self.pad_geometry_bias is not None
            edge_bias[:, :core_count, core_count:] = self.pad_geometry_bias(
                pad_geometry[:, :core_count, core_count:])
            edge_bias[:, core_count:, :core_count] = self.pad_geometry_bias(
                pad_geometry[:, core_count:, :core_count])
        return edge_bias

    def forward(
        self,
        streams,
        geometry,
        entity_type_features,
        entity_team_features,
        entity_role_features,
        pad_geometry, allowed, pad_edges, core_count, static_geometry_bias=None,
    ):
        streams = torch.nan_to_num(streams, nan=0.0, posinf=0.0, neginf=0.0)
        geometry = torch.nan_to_num(geometry, nan=0.0, posinf=0.0, neginf=0.0)
        pad_geometry = torch.nan_to_num(pad_geometry, nan=0.0, posinf=0.0, neginf=0.0)
        batch_size, entity_count = streams.shape[:2]
        queries = self.q(streams).view(
            batch_size,
            entity_count,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        keys = self.k(streams).view(
            batch_size,
            entity_count,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        values = self.v(streams).view(
            batch_size,
            entity_count,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        pairwise_features = self.pairwise_state_features(
            streams[:, :core_count],
            entity_type_features[:, :core_count],
            entity_team_features[:, :core_count],
            entity_role_features[:, :core_count],
        )

        geometry_bias = static_geometry_bias
        if geometry_bias is None:
            geometry_bias = self.static_geometry_bias(geometry, pad_geometry, core_count)
        if entity_count == core_count:
            edge_bias = geometry_bias + pairwise_features
        else:
            state_bias = geometry_bias.new_zeros(batch_size, entity_count, entity_count, self.num_heads)
            state_bias[:, :core_count, :core_count] = pairwise_features
            edge_bias = geometry_bias + state_bias
        bias = edge_bias.permute(
            0,
            3,
            1,
            2,
        ).to(queries.dtype)

        bias = bias.masked_fill(~allowed[:, None], float("-inf"))
        attended = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=bias,
            dropout_p=(
                self.dropout.p
                if self.training
                else 0.0
            ),
        )

        output = self.out(attended.transpose(1, 2).reshape(batch_size, entity_count, self.d_model))
        return output * allowed.any(-1, keepdim=True)


class LGPSTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.norm1 = nn.LayerNorm(config.d_model)
        self.attention = LGPSAttention(
            config.d_model,
            config.num_heads,
            config.geometry_dim,
            config.pairwise_rank,
            config.dropout,
            include_boosts=("big_boosts" in config.model_entities or "small_boosts" in config.model_entities),
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        streams,
        geometry,
        entity_type_features,
        entity_team_features,
        entity_role_features,
        pad_geometry, allowed, pad_edges, core_count, static_geometry_bias=None,
    ):
        normalized = self.norm1(streams)
        updates = self.attention(
            normalized,
            geometry,
            entity_type_features,
            entity_team_features,
            entity_role_features,
            pad_geometry, allowed, pad_edges, core_count, static_geometry_bias,
        )
        streams = streams + updates
        return streams + self.ffn(self.norm2(streams))

class LearnedQueryPooling(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()

        # This trainable query learns what information must be extracted
        # from the final set of nine contextualized agents.
        self.query = nn.Parameter(
            torch.randn(
                1,
                1,
                d_model,
            ) * 0.02
        )

        self.norm = nn.LayerNorm(d_model)

        self.attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, streams):
        # Pooling is intentionally delayed until after every multi-agent
        # interaction block has completed.
        agents = self.norm(streams)
        query = self.query.expand(
            agents.shape[0],
            -1,
            -1,
        )

        pooled, _ = self.attention(
            query,
            agents,
            agents,
            need_weights=False,
        )

        return pooled[:, 0]


class LGPSTransformer(nn.Module):
    def __init__(self, config: LGPSModelConfig = LGPSModelConfig()):
        super().__init__()
        self.config = config
        self.model_entities = normalize_model_entities(config.model_entities)
        self.entity_indices = torch.tensor(
            ([0] if "ball" in self.model_entities else [])
            + (list(range(1, 7)) if "players" in self.model_entities else [])
            + ([7, 8] if "goals" in self.model_entities else [])
            + (list(range(9, 15)) if "big_boosts" in self.model_entities else [])
            + (list(range(15, 43)) if "small_boosts" in self.model_entities else []),
            dtype=torch.long,
        )
        self.core_count = sum(index < 9 for index in self.entity_indices.tolist())
        self.geometry_indices = torch.tensor(
            [index for index in self.entity_indices.tolist() if index < 9]
            + list(range(9, 9 + len(self.entity_indices) - self.core_count)),
            dtype=torch.long,
        )

        # Ball, player, and goal entities have different raw feature spaces.
        # All six players share one player encoder so player slots do not
        # receive separate learned physics.
        self.ball_stream = (
            LGPSAgent(len(BALL_FEATURES), config.d_model)
            if "ball" in self.model_entities else None
        )
        self.player_stream = (
            LGPSAgent(len(PLAYER_FEATURES), config.d_model, missing_aware=True)
            if "players" in self.model_entities else None
        )
        self.goal_stream = (
            LGPSAgent(len(GOAL_FEATURES), config.d_model)
            if "goals" in self.model_entities else None
        )
        self.big_boost_stream = (
            LGPSAgent(6, config.d_model)
            if "big_boosts" in self.model_entities else None
        )
        self.small_boost_stream = (
            LGPSAgent(6, config.d_model)
            if "small_boosts" in self.model_entities else None
        )
        # Context is encoded separately and is never inserted into the entity axis.
        self.context_token = (
            LGPSAgent(len(CONTEXT_FEATURES), config.d_model)
            if "context" in self.model_entities else None
        )

        # Type embeddings identify ball/player/goal.
        self.type_embedding = nn.Embedding(
            4,
            config.d_model,
        )

        # Team embeddings identify blue/orange/neutral.
        self.team_embedding = nn.Embedding(
            3,
            config.d_model,
        )

        # Explicit pairwise physical relationships are computed once per frame
        # and used as attention biases in every transformer block.
        self.geometry = RelationalGeometry(
            geometry_dim=config.geometry_dim,
            pad_count=len(self.entity_indices) - self.core_count,
        )

        self.blocks = nn.ModuleList(
            [
                LGPSTransformerBlock(config)
                for _ in range(config.num_blocks)
            ]
        )
        self.loops_per_block = config.loops_per_block
        self.loop_embeddings = nn.Parameter(
            torch.zeros(
                config.num_blocks * config.loops_per_block,
                config.d_model,
            )
        )

        # A learned query summarizes the final configured entity state only after
        # all inter-agent communication has taken place.
        self.pool = LearnedQueryPooling(
            config.d_model,
            config.num_heads,
            config.dropout,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(config.d_model * 2),
            nn.Linear(
                config.d_model * 2,
                config.d_model,
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(
                config.d_model,
                config.num_outputs,
            ),
        )


    def encode_streams(self, ball, players, goals, big_boosts, small_boosts, entity_types, entity_teams):
        streams = []
        if "ball" in self.model_entities:
            streams.append(self.ball_stream(LGPSScaler.ball(ball)).unsqueeze(1))
        if "players" in self.model_entities:
            streams.append(self.player_stream(LGPSScaler.player(players)))
        if "goals" in self.model_entities:
            streams.append(self.goal_stream(LGPSScaler.goal(goals)))
        if "big_boosts" in self.model_entities:
            if big_boosts is None:
                raise ValueError("big_boosts must be supplied when selected")
            streams.append(self.big_boost_stream(
                big_boosts / big_boosts.new_tensor([5000., 5000., 1., 1., 1., 10.])))
        if "small_boosts" in self.model_entities:
            if small_boosts is None:
                raise ValueError("small_boosts must be supplied when selected")
            streams.append(self.small_boost_stream(
                small_boosts / small_boosts.new_tensor([5000., 5000., 1., 1., 1., 5.])))
        streams = torch.cat(streams, dim=1)
        return streams + self.type_embedding(entity_types)[None] + self.team_embedding(entity_teams)

    def forward(self, ball, players, goals, entity_types, entity_teams, game_context,
                big_boosts=None, small_boosts=None):
        if entity_types.ndim == 2:
            entity_types = entity_types[0]
        if entity_types.numel() != len(self.entity_indices):
            raise ValueError("entity_types must match model_entities")
        if "big_boosts" in self.model_entities and (big_boosts is None or big_boosts.shape[-2:] != (6, 6)):
            raise ValueError("big_boosts must have shape (..., 6, 6) when selected")
        if "small_boosts" in self.model_entities and (small_boosts is None or small_boosts.shape[-2:] != (28, 6)):
            raise ValueError("small_boosts must have shape (..., 28, 6) when selected")
        if ball.ndim != 2:
            raise ValueError("LGPS accepts one current frame per example; temporal input sequences are not supported")
        batch = ball.shape[0]
        entity_count = len(self.entity_indices)
        if entity_teams.ndim == 1:
            entity_teams = entity_teams.view(1, entity_count).expand(batch, -1)
        elif entity_teams.ndim == 2:
            if entity_teams.shape != (batch, entity_count):
                raise ValueError("entity_teams must describe every configured entity in each frame")
        else:
            raise ValueError("entity_teams must have shape (entities,) or (batch, entities)")
        current = self.encode_streams(ball, players, goals, big_boosts, small_boosts, entity_types, entity_teams)
        current_boosts = None if big_boosts is None and small_boosts is None else torch.cat(
            tuple(value for value in (big_boosts, small_boosts) if value is not None), dim=1)
        with torch.autocast(device_type=ball.device.type, enabled=False):
            full_teams = entity_teams.new_full((batch, 43), 2)
            full_teams[:, self.entity_indices.to(entity_teams.device)] = entity_teams
            full_geometry, full_pad_geometry, full_allowed, full_pad_edges = self.geometry(
                ball.float(), players.float(), goals.float(),
                current_boosts.float() if current_boosts is not None else None, full_teams)
        selected = self.geometry_indices.to(full_geometry.device)
        geometry = full_geometry.index_select(1, selected).index_select(2, selected)
        pad_geometry = full_pad_geometry.index_select(1, selected).index_select(2, selected)
        allowed = full_allowed.index_select(1, selected).index_select(2, selected)
        pad_edges = full_pad_edges.index_select(1, selected).index_select(2, selected)
        geometry, pad_geometry = geometry.to(current.dtype), pad_geometry.to(current.dtype)
        types = F.one_hot(entity_types, num_classes=4).float().expand(batch, -1, -1)
        teams = F.one_hot(entity_teams, num_classes=3).float()
        role_ids = torch.where(entity_types == 0, 0, torch.where(entity_types == 1, 1 + entity_teams,
                   torch.where(entity_types == 2, 3 + entity_teams, 5)))
        roles = F.one_hot(role_ids, num_classes=6).float()
        loop_index = 0
        for block in self.blocks:
            static_geometry_bias = block.attention.static_geometry_bias(geometry, pad_geometry, self.core_count)
            for _ in range(self.loops_per_block):
                state = current + self.loop_embeddings[loop_index].view(1, 1, -1)
                current = block(state, geometry, types, teams, roles, pad_geometry, allowed, pad_edges,
                                self.core_count, static_geometry_bias)
                loop_index += 1
        pooled = self.pool(current[:, :self.core_count])
        context = (self.context_token(LGPSScaler.context(game_context)) if "context" in self.model_entities
                   else current.new_zeros(batch, self.config.d_model))
        return self.head(torch.cat((pooled, context), dim=-1))

# Model construction is deferred to the training cell. This is important
# for in-notebook DDP because CUDA state must not be copied into workers.

def assert_player_permutation_invariant(
    model,
    ball,
    players,
    goals,
    entity_types,
    entity_teams,
    game_context,
):
    model.eval()
    with torch.inference_mode():
        original = model(
            ball,
            players,
            goals,
            entity_types,
            entity_teams,
            game_context,
            torch.zeros(ball.shape[0], 6, 6, device=ball.device),
            torch.zeros(ball.shape[0], 28, 6, device=ball.device),
        )
        permutation = torch.tensor(
            [2, 0, 1, 5, 3, 4],
            device=players.device,
        )
        permuted = model(
            ball,
            players[:, permutation],
            goals,
            entity_types,
            entity_teams,
            game_context,
            torch.zeros(ball.shape[0], 6, 6, device=ball.device),
            torch.zeros(ball.shape[0], 28, 6, device=ball.device),
        )
    if not torch.allclose(original, permuted, atol=1e-5):
        raise AssertionError(
            "Player-slot permutation changed the pooled prediction."
        )

