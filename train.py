"""
Training script for DDPG agent on multiple Gymnasium MuJoCo environments.
Trains using TensorFlow-CPU, saves best checkpoints, and optionally launches
visualization upon reaching good rewards.
"""

import os
import time
import warnings
import argparse

# Configure MuJoCo to use GLFW backend directly to avoid EGL fallback probe warnings
os.environ.setdefault("MUJOCO_GL", "glfw")
warnings.filterwarnings("ignore", category=UserWarning, module="glfw")
warnings.filterwarnings("ignore", message=".*EGL.*")

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from ddpg_agent import DDPGAgent
from visualize import visualize_agent, make_environment
from env_config import get_env_config, EnvConfig, VALID_ENV_NAMES


def evaluate_policy(env, agent: DDPGAgent, num_episodes: int = 3):
    """Run deterministic evaluation episodes without exploration noise."""
    eval_rewards = []
    eval_steps = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = agent.select_action(obs, noise=0.0)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            obs = next_obs
            ep_reward += reward
            steps += 1

        eval_rewards.append(ep_reward)
        eval_steps.append(steps)

    return float(np.mean(eval_rewards)), float(np.mean(eval_steps))


def save_plot(eval_episodes, eval_rewards, train_rewards, env_name: str, output_path="learning_curve.png"):
    """Plot and save training and evaluation reward curves."""
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(train_rewards, label="Training Episode Reward", alpha=0.35, color="gray")

        # Smooth training curve
        if len(train_rewards) >= 10:
            window = 10
            smoothed = np.convolve(train_rewards, np.ones(window) / window, mode="valid")
            plt.plot(range(window - 1, len(train_rewards)), smoothed, label=f"Train Moving Avg ({window})", color="blue", linewidth=1.8)

        if eval_episodes and eval_rewards:
            plt.plot(eval_episodes, eval_rewards, label="Deterministic Evaluation Reward", color="green", marker="o", linewidth=2.0)

        plt.xlabel("Episode")
        plt.ylabel("Reward")

        cfg = get_env_config(env_name)
        plt.title(f"DDPG Learning Curve: {cfg.display_name} (TF-CPU)")

        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"[Plot] Saved training curve to: {output_path}")
    except Exception as e:
        print(f"[Warning] Could not generate plot: {e}")


def train_ddpg(
    env_name: str = "pendulum",
    max_episodes: int = None,
    target_reward: float = None,
    batch_size: int = None,
    warmup_steps: int = None,
    eval_interval: int = None,
    checkpoint_dir: str = None,
    visualize_after_train: bool = True,
    vis_episodes: int = 5,
    render_mode: str = None,
    save_gif: str = None,
):
    """
    Train DDPG agent on the selected environment until max_episodes or
    target_reward is reached, then optionally visualize balancing/locomotion.
    """
    cfg = get_env_config(env_name)

    # Use env-specific defaults for any parameter not explicitly set
    max_episodes = max_episodes or cfg.max_episodes
    target_reward = target_reward or cfg.target_reward
    batch_size = batch_size or cfg.batch_size
    warmup_steps = warmup_steps or cfg.warmup_steps
    eval_interval = eval_interval or cfg.eval_interval
    checkpoint_dir = checkpoint_dir or f"checkpoints/{env_name}"
    render_mode = render_mode or ("pygame" if cfg.supports_pygame_2d else "human")

    env, env_id = make_environment(cfg, render_mode=None)
    eval_env, _ = make_environment(cfg, render_mode=None)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print("=" * 70)
    print(f"  DDPG TRAINING: {cfg.display_name} ({env_id}) on TensorFlow-CPU")
    print(f"  State Dim: {state_dim} | Action Dim: {action_dim} | Max Action: {max_action}")
    print(f"  Network: Actor {cfg.actor_hidden} | Critic {cfg.critic_hidden}")
    print(f"  Target Reward: {target_reward} | Max Episodes: {max_episodes}")
    print(f"  Batch Size: {batch_size} | Warmup Steps: {warmup_steps}")
    print("=" * 70)

    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        buffer_capacity=cfg.buffer_capacity,
        actor_hidden=cfg.actor_hidden,
        critic_hidden=cfg.critic_hidden,
    )

    total_steps = 0
    start_time = time.time()
    noise_std = cfg.noise_start
    best_eval_reward = -float("inf")

    train_rewards = []
    eval_episodes = []
    eval_rewards = []

    os.makedirs(checkpoint_dir, exist_ok=True)
    solved = False

    try:
        for episode in range(1, max_episodes + 1):
            obs, _ = env.reset()
            ep_reward = 0.0
            ep_steps = 0
            done = False

            while not done:
                total_steps += 1
                ep_steps += 1

                # Initial warmup exploration with gentle random actions
                if total_steps < warmup_steps:
                    action = env.action_space.sample() * 0.3
                else:
                    action = agent.select_action(obs, noise=noise_std)

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Critical RL bootstrapping detail:
                # Terminal state only if agent failed (terminated), NOT if max episode steps hit (truncated)
                done_learning = terminated and not truncated
                agent.buffer.add(obs, action, reward, next_obs, float(done_learning))

                # Gradient updates once buffer contains enough samples
                if total_steps >= batch_size * 2:
                    agent.train(batch_size=batch_size)

                obs = next_obs
                ep_reward += reward

            train_rewards.append(ep_reward)

            # Exponential decay of exploration noise towards minimum
            noise_std = max(cfg.noise_end, noise_std * cfg.noise_decay)

            if episode % 5 == 0:
                print(
                    f"  [Train] Ep {episode:4d}/{max_episodes} | Steps: {ep_steps:4d} | "
                    f"Reward: {ep_reward:7.1f} | Noise: {noise_std:.4f} | Total: {total_steps:6d}",
                    flush=True
                )

            # Periodic deterministic evaluation
            if episode % eval_interval == 0:
                mean_eval_rew, mean_eval_steps = evaluate_policy(eval_env, agent, num_episodes=3)
                eval_episodes.append(episode)
                eval_rewards.append(mean_eval_rew)
                elapsed = time.time() - start_time

                print(
                    f"\n>>> EVAL Ep {episode:4d} | Noise: {noise_std:.4f} | "
                    f"Eval Reward: {mean_eval_rew:8.1f} | Eval Steps: {mean_eval_steps:4.0f} | "
                    f"Total Steps: {total_steps:7d} | Elapsed: {elapsed:6.1f}s",
                    flush=True
                )

                # Always update latest weights
                agent.save_weights(checkpoint_dir=checkpoint_dir, prefix="latest")

                # Save best model
                if mean_eval_rew > best_eval_reward:
                    best_eval_reward = mean_eval_rew
                    agent.save_weights(checkpoint_dir=checkpoint_dir, prefix="best")
                    print(f"  >>> [Checkpoint] New best model saved with evaluation reward {best_eval_reward:.1f}!\n", flush=True)

                # Early stopping if target good reward threshold reached
                if mean_eval_rew >= target_reward:
                    print("\n" + "*" * 70, flush=True)
                    print(f"  SUCCESS: TARGET REWARD OF {target_reward} REACHED AT EPISODE {episode}!", flush=True)
                    print(f"  Final Evaluation Reward: {mean_eval_rew:.1f} (Balanced for {mean_eval_steps:.0f} steps)", flush=True)
                    print("*" * 70 + "\n", flush=True)
                    solved = True
                    break

    except KeyboardInterrupt:
        print("\n[Interrupted] Training stopped early by user. Saving current weights...")
        agent.save_weights(checkpoint_dir=checkpoint_dir, prefix="latest")

    finally:
        env.close()
        eval_env.close()

    # Always save latest weights as well
    agent.save_weights(checkpoint_dir=checkpoint_dir, prefix="latest")
    save_plot(eval_episodes, eval_rewards, train_rewards, env_name,
              output_path=os.path.join(checkpoint_dir, "learning_curve.png"))

    if not solved:
        print(f"\n[Completed] Finished training. Best evaluation reward achieved: {best_eval_reward:.1f}")

    # Launch visualization
    if visualize_after_train:
        print("\n" + "=" * 70)
        print(f"  LAUNCHING VISUALIZATION: {cfg.display_name.upper()}")
        print("=" * 70)
        visualize_agent(
            env_name=env_name,
            checkpoint_dir=checkpoint_dir,
            prefix="best",
            num_episodes=vis_episodes,
            render_mode=render_mode,
            save_gif_path=save_gif,
        )


def main():
    parser = argparse.ArgumentParser(description="Train DDPG agent on Gymnasium MuJoCo environments with TensorFlow-CPU.")
    parser.add_argument("--env", type=str, default="pendulum", choices=VALID_ENV_NAMES, help="Environment to train on.")
    parser.add_argument("--episodes", type=int, default=None, help="Max episodes to train (default: env-specific).")
    parser.add_argument("--target-reward", type=float, default=None, help="Target reward for early stopping (default: env-specific).")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for DDPG updates (default: env-specific).")
    parser.add_argument("--eval-interval", type=int, default=None, help="Episodes between evaluations (default: env-specific).")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to save weights (default: checkpoints/<env>).")
    parser.add_argument("--no-vis", action="store_true", help="Disable automatic visualization after training.")
    parser.add_argument("--vis-episodes", type=int, default=5, help="Number of visualization episodes to run (default: 5).")
    parser.add_argument("--render-mode", type=str, default=None, choices=["pygame", "human", "rgb_array"], help="Viewer engine.")
    parser.add_argument("--save-gif", type=str, default=None, help="Save an animated GIF of balancing.")

    args = parser.parse_args()
    train_ddpg(
        env_name=args.env,
        max_episodes=args.episodes,
        target_reward=args.target_reward,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        checkpoint_dir=args.checkpoint_dir,
        visualize_after_train=not args.no_vis,
        vis_episodes=args.vis_episodes,
        render_mode=args.render_mode,
        save_gif=args.save_gif,
    )


if __name__ == "__main__":
    main()
