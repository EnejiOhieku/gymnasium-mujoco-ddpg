"""
Unified Entrypoint for Multi-Environment DDPG with TensorFlow-CPU.

Supported environments:
  - pendulum: InvertedDoublePendulum-v5 (balance a double pendulum on a cart)
  - ant:      Ant-v5 (teach a spider robot to walk)

Usage:
  python main.py --env pendulum --mode train
  python main.py --env ant --mode train
  python main.py --env pendulum --mode visualize --render-mode pygame
  python main.py --env ant --mode visualize --render-mode human --vis-episodes 3
  python main.py --env ant --mode record --gif-path ant_walking.gif
"""

import os
import warnings
import argparse

# Configure MuJoCo to use GLFW backend directly to avoid EGL fallback probe warnings
os.environ.setdefault("MUJOCO_GL", "glfw")
warnings.filterwarnings("ignore", category=UserWarning, module="glfw")
warnings.filterwarnings("ignore", message=".*EGL.*")

from train import train_ddpg
from visualize import visualize_agent
from env_config import get_env_config, VALID_ENV_NAMES


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Environment DDPG with TensorFlow-CPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Environment selection
    parser.add_argument(
        "--env",
        type=str,
        choices=VALID_ENV_NAMES,
        default="pendulum",
        help="Environment to train/visualize.",
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "visualize", "record"],
        default="train",
        help="Execution mode: 'train', 'visualize' (view saved model), or 'record' (export GIF).",
    )

    # Training args
    parser.add_argument("--episodes", type=int, default=None, help="Max episodes to train (default: env-specific).")
    parser.add_argument("--target-reward", type=float, default=None, help="Reward threshold for early stopping (default: env-specific).")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory for model weights (default: checkpoints/<env>).")

    # Visualization args
    parser.add_argument("--vis-episodes", type=int, default=5, help="Number of visualization episodes.")
    parser.add_argument("--gif-path", type=str, default=None, help="Output path for recorded GIF.")
    parser.add_argument("--fps", type=int, default=50, help="Framerate for visualization.")
    parser.add_argument("--render-mode", type=str, default=None, choices=["pygame", "human"],
                        help="Viewer engine: 'pygame' (2D, pendulum only) or 'human' (3D MuJoCo).")

    # Perturbation args
    parser.add_argument("--auto-perturb", action="store_true", help="Inject periodic perturbations during visualization.")
    parser.add_argument("--perturb-interval", type=int, default=150, help="Steps between automatic perturbations.")
    parser.add_argument("--perturb-magnitude", type=float, default=0.45, help="Impulse magnitude for perturbations.")
    parser.add_argument("--no-vis", action="store_true", help="Disable automatic visualization after training.")

    args = parser.parse_args()

    # Resolve defaults based on environment
    cfg = get_env_config(args.env)
    checkpoint_dir = args.checkpoint_dir or f"checkpoints/{args.env}"
    render_mode = args.render_mode or ("pygame" if cfg.supports_pygame_2d else "human")
    gif_path = args.gif_path or f"{args.env}_demo.gif"

    if args.mode == "train":
        train_ddpg(
            env_name=args.env,
            max_episodes=args.episodes,
            target_reward=args.target_reward,
            checkpoint_dir=checkpoint_dir,
            visualize_after_train=not args.no_vis,
            vis_episodes=args.vis_episodes,
            render_mode=render_mode,
            save_gif=None,
        )
    elif args.mode == "visualize":
        visualize_agent(
            env_name=args.env,
            checkpoint_dir=checkpoint_dir,
            prefix="best",
            num_episodes=args.vis_episodes,
            render_mode=render_mode,
            auto_perturb=args.auto_perturb,
            perturb_interval=args.perturb_interval,
            perturb_magnitude=args.perturb_magnitude,
            fps=args.fps,
        )
    elif args.mode == "record":
        visualize_agent(
            env_name=args.env,
            checkpoint_dir=checkpoint_dir,
            prefix="best",
            num_episodes=2,
            render_mode="rgb_array",
            save_gif_path=gif_path,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
