"""
Environment Configuration Registry for Multi-Environment DDPG.
Maps short environment names to Gymnasium IDs, hyperparameters, and visualization settings.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class EnvConfig:
    """Configuration bundle for a Gymnasium MuJoCo environment."""
    # Gymnasium env ID (with fallback alternatives)
    env_ids: List[str]
    display_name: str

    # Network architecture
    actor_hidden: Tuple[int, ...] = (256, 256)
    critic_hidden: Tuple[int, ...] = (256, 256)

    # Training hyperparameters
    target_reward: float = 5000.0
    max_episodes: int = 450
    batch_size: int = 128
    warmup_steps: int = 300
    buffer_capacity: int = 200_000
    eval_interval: int = 15
    noise_start: float = 0.1
    noise_end: float = 0.015
    noise_decay: float = 0.995
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3

    # Visualization
    supports_pygame_2d: bool = False
    camera_distance: float = 5.0
    camera_elevation: float = -15.0
    camera_azimuth: float = 90.0
    camera_trackbodyid: int = 0
    camera_lookat: Tuple[float, float, float] = (0.0, 0.0, 0.5)

    # Perturbation defaults
    perturb_magnitude: float = 0.45
    perturb_interval: int = 150

    # Env-specific kwargs to pass to gym.make()
    env_kwargs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ENV_CONFIGS = {
    "pendulum": EnvConfig(
        env_ids=["InvertedDoublePendulum-v5", "InvertedDoublePendulum-v4"],
        display_name="Inverted Double Pendulum",
        actor_hidden=(256, 256),
        critic_hidden=(256, 256),
        target_reward=5000.0,
        max_episodes=450,
        batch_size=128,
        warmup_steps=300,
        buffer_capacity=200_000,
        eval_interval=15,
        noise_start=0.1,
        noise_end=0.015,
        noise_decay=0.995,
        actor_lr=3e-4,
        critic_lr=1e-3,
        supports_pygame_2d=True,
        camera_distance=3.8,
        camera_elevation=-12.0,
        camera_azimuth=90.0,
        camera_trackbodyid=1,
        camera_lookat=(0.0, 0.0, 0.6),
        perturb_magnitude=0.45,
        perturb_interval=150,
    ),
    "ant": EnvConfig(
        env_ids=["Ant-v5", "Ant-v4"],
        display_name="Ant (Walking Spider)",
        actor_hidden=(400, 300),
        critic_hidden=(400, 300),
        target_reward=2500.0,
        max_episodes=1500,
        batch_size=256,
        warmup_steps=3000,
        buffer_capacity=500_000,
        eval_interval=15,
        noise_start=0.15,
        noise_end=0.02,
        noise_decay=0.998,
        actor_lr=1e-4,
        critic_lr=3e-4,
        supports_pygame_2d=False,
        camera_distance=6.0,
        camera_elevation=-20.0,
        camera_azimuth=45.0,
        camera_trackbodyid=1,  # Track the torso
        camera_lookat=(0.0, 0.0, 0.5),
        perturb_magnitude=2.0,
        perturb_interval=200,
        env_kwargs={"ctrl_cost_weight": 0.05, "healthy_reward": 2.0},
    ),
}

VALID_ENV_NAMES = list(ENV_CONFIGS.keys())


def get_env_config(name: str) -> EnvConfig:
    """Look up environment config by short name."""
    name = name.lower().strip()
    if name not in ENV_CONFIGS:
        raise ValueError(
            f"Unknown environment '{name}'. Choose from: {VALID_ENV_NAMES}"
        )
    return ENV_CONFIGS[name]
