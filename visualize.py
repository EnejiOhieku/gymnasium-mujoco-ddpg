"""
Visualization module for the DDPG agent across multiple Gymnasium MuJoCo environments.
Provides:
  - Interactive perturbations via keyboard (Left/Right push, Up/Down tilt, Random kick).
  - Automated periodic perturbations (--auto-perturb, --perturb-interval, --perturb-magnitude).
  - 2D Interactive Pygame viewer (render_mode='pygame') with live telemetry HUD & force arrows.
  - 3D Native MuJoCo viewer (render_mode='human') with camera tracking.
  - GIF / Video exporter (render_mode='rgb_array').
"""

import os
import sys
import time
import warnings
import argparse

# Direct MuJoCo to GLFW backend and suppress harmless GLFW EGL probe warnings
os.environ.setdefault("MUJOCO_GL", "glfw")
warnings.filterwarnings("ignore", category=UserWarning, module="glfw")
warnings.filterwarnings("ignore", message=".*EGL.*")

import numpy as np
import gymnasium as gym
import mujoco

# Compatibility patch for Gymnasium 1.2.2 + MuJoCo 3.13+ mouse callback signature
_original_mjv_moveCamera = mujoco.mjv_moveCamera
def _safe_mjv_moveCamera(*args, **kwargs):
    if len(args) == 6:
        m, action, reldx, reldy, scn, cam = args
        return _original_mjv_moveCamera(m, action, reldx, reldy, cam)
    return _original_mjv_moveCamera(*args, **kwargs)
mujoco.mjv_moveCamera = _safe_mjv_moveCamera

from ddpg_agent import DDPGAgent
from env_config import get_env_config, EnvConfig, VALID_ENV_NAMES


def make_environment(env_config: EnvConfig, render_mode: str = "human"):
    """Create Gymnasium environment from config with version fallback."""
    for env_id in env_config.env_ids:
        try:
            env = gym.make(env_id, render_mode=render_mode, **env_config.env_kwargs)
            return env, env_id
        except Exception:
            continue
    raise RuntimeError(
        f"Could not create any of {env_config.env_ids}. "
        "Check that gymnasium[mujoco] is installed."
    )


def configure_camera(env, env_config: EnvConfig, mode: str = "human"):
    """Tune MuJoCo camera using environment-specific parameters."""
    try:
        viewer = env.unwrapped.mujoco_renderer._get_viewer(mode)
        if hasattr(viewer, "cam"):
            viewer.cam.trackbodyid = env_config.camera_trackbodyid
            viewer.cam.distance = env_config.camera_distance
            viewer.cam.elevation = env_config.camera_elevation
            viewer.cam.azimuth = env_config.camera_azimuth
            viewer.cam.lookat[0] = env_config.camera_lookat[0]
            viewer.cam.lookat[1] = env_config.camera_lookat[1]
            viewer.cam.lookat[2] = env_config.camera_lookat[2]
    except Exception:
        pass


class Pygame2DVisualizer:
    """Crisp 2D Vector Visualizer with Real-time HUD Telemetry & Interactive Perturbations.
    
    Note: This viewer is specifically designed for the InvertedDoublePendulum environment.
    Other environments should use the MuJoCo 'human' render mode.
    """

    def __init__(self, width: int = 680, height: int = 720, perturb_strength: float = 0.45):
        import pygame
        pygame.init()
        pygame.font.init()
        self.width = width
        self.height = height
        self.perturb_strength = perturb_strength
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Gymnasium Double Pendulum - DDPG (TF-CPU)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("DejaVu Sans, Arial, sans-serif", 15)
        self.font_bold = pygame.font.SysFont("DejaVu Sans, Arial, sans-serif", 17, bold=True)
        self.paused = False

        # Perturbation state
        self.pending_perturbation = None
        self.perturb_indicator_timer = 0
        self.perturb_indicator_msg = ""
        self.perturb_direction = 0  # -1 left, +1 right, 0 none

    def handle_events(self):
        """Handle keyboard & window events, including interactive perturbation hotkeys."""
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                # Interactive perturbations
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    self.pending_perturbation = ("cart", -self.perturb_strength)
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    self.pending_perturbation = ("cart", +self.perturb_strength)
                elif event.key in (pygame.K_UP, pygame.K_w):
                    self.pending_perturbation = ("pole", +0.35)
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    self.pending_perturbation = ("pole", -0.35)
                elif event.key == pygame.K_p:
                    choice = np.random.choice(["cart_left", "cart_right", "pole_cw", "pole_ccw"])
                    if choice == "cart_left":
                        self.pending_perturbation = ("cart", -self.perturb_strength)
                    elif choice == "cart_right":
                        self.pending_perturbation = ("cart", +self.perturb_strength)
                    elif choice == "pole_cw":
                        self.pending_perturbation = ("pole", +0.35)
                    else:
                        self.pending_perturbation = ("pole", -0.35)
        return True

    def consume_perturbation(self):
        """Consume and return any user-triggered perturbation."""
        perturb = self.pending_perturbation
        self.pending_perturbation = None
        return perturb

    def trigger_indicator(self, msg: str, direction: int = 0):
        """Activate visual perturbation badge & arrow for ~30 frames."""
        self.perturb_indicator_timer = 30
        self.perturb_indicator_msg = msg
        self.perturb_direction = direction

    def render_frame(
        self,
        obs: np.ndarray,
        ep: int,
        total_eps: int,
        step: int,
        reward: float,
        action: float,
        tip_y: float,
        fps: int = 50,
    ):
        """Draw 2D kinematic simulation, perturbation indicators, and overlay telemetry HUD."""
        import pygame
        if not self.handle_events():
            return False

        while self.paused:
            if not self.handle_events():
                return False
            self.clock.tick(15)

        cart_x = float(obs[0])
        sin1, sin2 = float(obs[1]), float(obs[2])
        cos1, cos2 = float(obs[3]), float(obs[4])

        render_y_start = 105
        render_h = self.height - render_y_start

        # Background
        sim_rect = pygame.Rect(0, render_y_start, self.width, render_h)
        pygame.draw.rect(self.screen, (20, 24, 34), sim_rect)

        # Coordinate transforms
        center_x = self.width // 2
        track_y = render_y_start + render_h - 150
        scale = 170.0  # pixels per meter

        # 1. Track rail
        rail_w = int(2.0 * scale)
        pygame.draw.line(self.screen, (70, 80, 110), (center_x - rail_w, track_y), (center_x + rail_w, track_y), 4)

        # Distance tick marks
        for m in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
            tx = int(center_x + m * scale)
            pygame.draw.line(self.screen, (90, 100, 130), (tx, track_y - 6), (tx, track_y + 6), 2)

        # 2. Cart
        cart_w, cart_h = int(0.26 * scale), int(0.12 * scale)
        cart_center_x = int(center_x + cart_x * scale)
        cart_rect = pygame.Rect(cart_center_x - cart_w // 2, track_y - cart_h // 2, cart_w, cart_h)
        pygame.draw.rect(self.screen, (240, 195, 45), cart_rect, border_radius=6)
        pygame.draw.rect(self.screen, (255, 235, 130), cart_rect, width=2, border_radius=6)

        # 3. Double Pendulum Arm Kinematics
        L1 = 0.6 * scale
        p1_x = cart_center_x
        p1_y = track_y
        p2_x = int(p1_x + L1 * sin1)
        p2_y = int(p1_y - L1 * cos1)

        L2 = 0.6 * scale
        p3_x = int(p2_x + L2 * sin2)
        p3_y = int(p2_y - L2 * cos2)

        # Link 1 (inner arm - cyan)
        pygame.draw.line(self.screen, (30, 200, 240), (p1_x, p1_y), (p2_x, p2_y), 10)
        pygame.draw.circle(self.screen, (255, 255, 255), (p1_x, p1_y), 6)

        # Link 2 (outer arm - lime green)
        pygame.draw.line(self.screen, (60, 230, 120), (p2_x, p2_y), (p3_x, p3_y), 8)
        pygame.draw.circle(self.screen, (255, 140, 40), (p2_x, p2_y), 7)
        pygame.draw.circle(self.screen, (255, 60, 60), (p3_x, p3_y), 7)

        # Draw Perturbation Impulse Arrow if active
        if self.perturb_indicator_timer > 0 and self.perturb_direction != 0:
            arrow_len = 50 * self.perturb_direction
            arrow_color = (255, 180, 20)
            start_pt = (cart_center_x, track_y - 25)
            end_pt = (cart_center_x + arrow_len, track_y - 25)
            pygame.draw.line(self.screen, arrow_color, start_pt, end_pt, 4)
            # Arrow head
            head_dir = 1 if self.perturb_direction > 0 else -1
            head_pts = [
                end_pt,
                (end_pt[0] - 12 * head_dir, end_pt[1] - 8),
                (end_pt[0] - 12 * head_dir, end_pt[1] + 8),
            ]
            pygame.draw.polygon(self.screen, arrow_color, head_pts)

        # Floor reference
        pygame.draw.line(self.screen, (40, 46, 60), (0, track_y + 40), (self.width, track_y + 40), 2)

        # 4. HUD Top Banner
        hud_rect = pygame.Rect(0, 0, self.width, render_y_start)
        pygame.draw.rect(self.screen, (15, 18, 24), hud_rect)
        pygame.draw.line(self.screen, (55, 65, 85), (0, render_y_start - 1), (self.width, render_y_start - 1), 2)

        title_surf = self.font_bold.render("DOUBLE PENDULUM BALANCING (DDPG / TF-CPU)", True, (80, 180, 255))
        self.screen.blit(title_surf, (15, 10))

        # Status badge or Perturbation indicator
        if self.perturb_indicator_timer > 0:
            badge_surf = self.font_bold.render(f"⚡ [{self.perturb_indicator_msg}]", True, (255, 190, 40))
            self.perturb_indicator_timer -= 1
        else:
            is_balanced = tip_y > 1.2
            status_text = "BALANCING" if is_balanced else "FALLING"
            status_color = (60, 230, 110) if is_balanced else (255, 75, 75)
            badge_surf = self.font_bold.render(f"[{status_text}]", True, status_color)
        self.screen.blit(badge_surf, (self.width - 230, 10))

        line1 = f"Episode: {ep}/{total_eps}    Step: {step:4d}/1000    Reward: {reward:7.1f}"
        l1_surf = self.font.render(line1, True, (225, 230, 240))
        self.screen.blit(l1_surf, (15, 38))

        line2 = f"Cart X: {cart_x:+.2f}m    Tip Y: {tip_y:.2f}m    Policy Force: {action:+.2f}N"
        l2_surf = self.font.render(line2, True, (170, 185, 205))
        self.screen.blit(l2_surf, (15, 60))

        controls_surf = self.font.render("[A/D/Arrows] Push Cart   [W/S] Tilt Pole   [P] Kick   [SPACE] Pause", True, (130, 150, 175))
        self.screen.blit(controls_surf, (15, 82))

        pygame.display.flip()
        self.clock.tick(fps)
        return True

    def close(self):
        import pygame
        pygame.quit()


def _apply_perturbation(env, env_name: str, p_type: str, p_val: float):
    """Apply a perturbation impulse to the environment's physical state.
    
    For pendulum: modifies cart velocity (qvel[0]) or pole angular velocity (qvel[1]).
    For ant: modifies torso velocity (qvel[0:3]) or applies angular perturbation (qvel[3:6]).
    """
    if env_name == "pendulum":
        if p_type == "cart":
            env.unwrapped.data.qvel[0] += p_val
        else:  # pole
            env.unwrapped.data.qvel[1] += p_val
    elif env_name == "ant":
        if p_type == "push":
            # Push the torso in x or y direction
            direction = np.random.choice([0, 1])  # 0=x, 1=y
            env.unwrapped.data.qvel[direction] += p_val
        else:  # angular kick
            axis = np.random.choice([3, 4, 5])  # roll/pitch/yaw
            env.unwrapped.data.qvel[axis] += p_val * 0.5


def _get_perturbation_msg(env_name: str, p_type: str, p_val: float) -> tuple:
    """Return (message_string, direction_indicator) for the perturbation."""
    if env_name == "pendulum":
        if p_type == "cart":
            return f"PUSH CART {p_val:+.2f} m/s", (1 if p_val > 0 else -1)
        else:
            return f"TILT POLE {p_val:+.2f} rad/s", 0
    else:  # ant
        if p_type == "push":
            return f"PUSH TORSO {p_val:+.2f} m/s", (1 if p_val > 0 else -1)
        else:
            return f"ANGULAR KICK {p_val:+.2f} rad/s", 0


def _get_tip_y(env, env_name: str) -> float:
    """Get the relevant height metric for the environment."""
    try:
        if env_name == "pendulum":
            return float(env.unwrapped.data.site_xpos[0][2])
        elif env_name == "ant":
            # Torso z-position (height above ground)
            return float(env.unwrapped.data.body("torso").xpos[2])
    except Exception:
        pass
    return 1.0


def visualize_agent(
    env_name: str = "pendulum",
    checkpoint_dir: str = "checkpoints",
    prefix: str = "best",
    num_episodes: int = 5,
    render_mode: str = "human",
    auto_perturb: bool = False,
    perturb_interval: int = 150,
    perturb_magnitude: float = 0.45,
    save_gif_path: str = None,
    fps: int = 50,
):
    """
    Run evaluation episodes and render the agent in the selected environment.
    - Supports interactive keyboard perturbations (pendulum: Left/Right push, Up/Down tilt, P kick).
    - Supports automated periodic perturbations via auto_perturb.
    """
    cfg = get_env_config(env_name)
    has_display = "DISPLAY" in os.environ and bool(os.environ["DISPLAY"])

    if not has_display and not save_gif_path:
        print("[WARNING] No active DISPLAY found. Switching to background GIF recording.")
        save_gif_path = "balancing_demo.gif"

    # Pygame 2D viewer only supported for pendulum
    use_pygame = (render_mode == "pygame" and cfg.supports_pygame_2d and not save_gif_path)
    if render_mode == "pygame" and not cfg.supports_pygame_2d:
        print(f"[INFO] Pygame 2D viewer not available for '{env_name}'. Using MuJoCo 3D viewer.")
        render_mode = "human"

    if save_gif_path:
        env_render_mode = "rgb_array"
    elif use_pygame:
        env_render_mode = None  # Pygame renders from obs, no OpenGL needed from MuJoCo
    else:
        env_render_mode = "human"

    env, env_id = make_environment(cfg, render_mode=env_render_mode)
    print(f"\n[Environment] Loaded {env_id} (Viewer Engine: '{render_mode}')")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        actor_hidden=cfg.actor_hidden,
        critic_hidden=cfg.critic_hidden,
    )

    try:
        actor_path, _ = agent.load_weights(checkpoint_dir=checkpoint_dir, prefix=prefix)
        print(f"[Model] Successfully loaded actor weights from: {actor_path}")
    except FileNotFoundError as err:
        print(f"[Error] {err}")
        print("Please train the agent first using: python main.py --mode train --env " + env_name)
        env.close()
        return

    # Configure camera for MuJoCo 3D renderers
    if env_render_mode in ("human", "rgb_array"):
        configure_camera(env, cfg, mode=env_render_mode)

    pyg_viewer = None
    if use_pygame:
        pyg_viewer = Pygame2DVisualizer(width=680, height=720, perturb_strength=perturb_magnitude)

    # Use env-specific defaults if not overridden
    if perturb_magnitude == 0.45:
        perturb_magnitude = cfg.perturb_magnitude
    if perturb_interval == 150:
        perturb_interval = cfg.perturb_interval

    gif_frames = []
    episode_rewards = []
    episode_steps = []
    max_steps = env.spec.max_episode_steps or 1000

    print("\n" + "=" * 70)
    print(f"  VISUALIZING DDPG ON {cfg.display_name.upper()} ACROSS {num_episodes} EPISODES")
    print(f"  Mode: {render_mode.upper()}")
    if cfg.supports_pygame_2d and use_pygame:
        print("  Controls:")
        print("    [A] / [LEFT]     : Push Cart Left")
        print("    [D] / [RIGHT]    : Push Cart Right")
        print("    [W] / [UP]       : Tilt Pole Clockwise")
        print("    [S] / [DOWN]     : Tilt Pole Counter-Clockwise")
        print("    [P]              : Apply Random Disturbance Impulse")
        print("    [SPACE]          : Pause / Resume Simulation")
        print("    [Q] / [ESC]      : Quit Viewer")
    if auto_perturb:
        print(f"  ⚡ Auto-Perturbation: Active every {perturb_interval} steps (magnitude: {perturb_magnitude})")
    print("=" * 70)

    delay = 1.0 / fps if fps > 0 else 0.02
    keep_running = True

    try:
        for ep in range(1, num_episodes + 1):
            if not keep_running:
                break

            obs, info = env.reset()
            if env_render_mode in ("human", "rgb_array"):
                configure_camera(env, cfg, mode=env_render_mode)

            ep_reward = 0.0
            steps = 0
            done = False

            print(f"\n>>> Starting Episode {ep}/{num_episodes} ...")

            while not done and keep_running:
                # 1. Check for interactive keyboard perturbation
                perturb = None
                if pyg_viewer is not None:
                    perturb = pyg_viewer.consume_perturbation()

                # 2. Check for automated periodic perturbation
                if perturb is None and auto_perturb and (steps > 50) and (steps % perturb_interval == 0):
                    if env_name == "pendulum":
                        p_type = np.random.choice(["cart", "pole"])
                        p_sign = np.random.choice([-1.0, 1.0])
                        p_val = p_sign * (perturb_magnitude if p_type == "cart" else 0.35)
                    else:  # ant
                        p_type = np.random.choice(["push", "angular"])
                        p_sign = np.random.choice([-1.0, 1.0])
                        p_val = p_sign * perturb_magnitude
                    perturb = (p_type, p_val)

                # 3. Apply perturbation to MuJoCo physical state
                if perturb is not None:
                    p_type, p_val = perturb
                    _apply_perturbation(env, env_name, p_type, p_val)
                    p_msg, p_dir = _get_perturbation_msg(env_name, p_type, p_val)
                    print(f"  ⚡ [Step {steps:4d}] Perturbation Applied: {p_msg}")
                    if pyg_viewer is not None:
                        pyg_viewer.trigger_indicator(p_msg, direction=p_dir)

                # 4. Agent takes action
                action = agent.select_action(obs, noise=0.0)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                obs = next_obs
                ep_reward += reward
                steps += 1

                tip_y = _get_tip_y(env, env_name)

                if pyg_viewer is not None:
                    keep_running = pyg_viewer.render_frame(
                        obs, ep, num_episodes, steps, ep_reward,
                        float(action[0]) if action_dim == 1 else float(np.linalg.norm(action)),
                        tip_y, fps=fps
                    )
                elif render_mode == "human":
                    time.sleep(delay)
                elif save_gif_path:
                    frame = env.render()
                    if frame is not None:
                        gif_frames.append(frame)

                if steps % 100 == 0:
                    if env_name == "pendulum":
                        cart_x = float(obs[0])
                        print(f"  Ep {ep} | Step {steps:4d}/{max_steps} | Cart X: {cart_x:+.2f} | Tip Y: {tip_y:.2f} | Reward: {ep_reward:.1f}")
                    else:
                        x_pos = info.get("x_position", 0.0)
                        x_vel = info.get("x_velocity", 0.0)
                        print(f"  Ep {ep} | Step {steps:4d}/{max_steps} | X: {x_pos:+.2f} | Height: {tip_y:.2f} | V: {x_vel:+.2f} | Reward: {ep_reward:.1f}")

            episode_rewards.append(ep_reward)
            episode_steps.append(steps)
            outcome = "TIMELIMIT (Max Steps)" if truncated else ("Terminated" if terminated else "Done")
            print(f"  Episode {ep} finished: Steps = {steps}, Total Reward = {ep_reward:.1f} [{outcome}]")

    except KeyboardInterrupt:
        print("\n[Interrupted] Visualization stopped by user.")
    finally:
        if pyg_viewer is not None:
            pyg_viewer.close()
        env.close()

    print("\n" + "=" * 70)
    print("  VISUALIZATION SUMMARY")
    print(f"  Environment: {cfg.display_name}")
    print(f"  Episodes Completed: {len(episode_rewards)}")
    if episode_rewards:
        print(f"  Average Reward:    {np.mean(episode_rewards):.1f} (Max: {np.max(episode_rewards):.1f})")
        print(f"  Average Steps:     {np.mean(episode_steps):.1f} (Max: {np.max(episode_steps):.1f})")
    print("=" * 70 + "\n")

    if save_gif_path and len(gif_frames) > 0:
        try:
            import imageio
            print(f"[GIF] Saving animation ({len(gif_frames)} frames) to {save_gif_path} ...")
            step_skip = max(1, len(gif_frames) // 300)
            imageio.mimsave(save_gif_path, gif_frames[::step_skip], fps=max(15, fps // step_skip))
            print(f"[GIF] Successfully saved GIF: {save_gif_path}")
        except Exception as e:
            print(f"[Warning] Could not save GIF: {e}")


def main():
    parser = argparse.ArgumentParser(description="Visualize trained DDPG agent with perturbations.")
    parser.add_argument("--env", type=str, default="pendulum", choices=VALID_ENV_NAMES, help="Environment to visualize.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Path to checkpoint directory.")
    parser.add_argument("--prefix", type=str, default="best", help="Checkpoint file prefix (default: best).")
    parser.add_argument("--episodes", type=int, default=5, help="Number of visualization episodes.")
    parser.add_argument("--fps", type=int, default=50, help="Frame rate for visualization (default: 50).")
    parser.add_argument("--render-mode", type=str, default=None, choices=["pygame", "human"], help="Renderer: 'pygame' (2D GUI, pendulum only) or 'human' (Native 3D MuJoCo).")
    parser.add_argument("--auto-perturb", action="store_true", help="Inject periodic perturbations automatically.")
    parser.add_argument("--perturb-interval", type=int, default=150, help="Steps between automatic perturbations.")
    parser.add_argument("--perturb-magnitude", type=float, default=0.45, help="Impulse magnitude.")
    parser.add_argument("--save-gif", type=str, default=None, help="Optional path to save an animated GIF.")

    args = parser.parse_args()

    # Default checkpoint dir based on env name
    checkpoint_dir = args.checkpoint_dir or f"checkpoints/{args.env}"

    # Default render mode: pygame for pendulum, human for everything else
    cfg = get_env_config(args.env)
    render_mode = args.render_mode or ("pygame" if cfg.supports_pygame_2d else "human")

    visualize_agent(
        env_name=args.env,
        checkpoint_dir=checkpoint_dir,
        prefix=args.prefix,
        num_episodes=args.episodes,
        render_mode=render_mode,
        auto_perturb=args.auto_perturb,
        perturb_interval=args.perturb_interval,
        perturb_magnitude=args.perturb_magnitude,
        save_gif_path=args.save_gif,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
