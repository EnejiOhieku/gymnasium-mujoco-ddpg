"""
Deep Deterministic Policy Gradient (DDPG) Agent using TensorFlow-CPU.
Designed for continuous control environments in Gymnasium (e.g., InvertedDoublePendulum-v5).
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Force TensorFlow to execute on CPU
tf.config.set_visible_devices([], 'GPU')
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class ReplayBuffer:
    """Experience Replay Buffer for DDPG."""

    def __init__(self, capacity: int = 200000, state_dim: int = 9, action_dim: int = 1):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: float):
        """Add transition to replay buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int = 128):
        """Sample mini-batch as TensorFlow tensors."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            tf.convert_to_tensor(self.states[indices], dtype=tf.float32),
            tf.convert_to_tensor(self.actions[indices], dtype=tf.float32),
            tf.convert_to_tensor(self.rewards[indices], dtype=tf.float32),
            tf.convert_to_tensor(self.next_states[indices], dtype=tf.float32),
            tf.convert_to_tensor(self.dones[indices], dtype=tf.float32),
        )

    def __len__(self):
        return self.size


def build_actor(state_dim: int, action_dim: int, max_action: float = 1.0,
                hidden_sizes: tuple = (256, 256)) -> tf.keras.Model:
    """Actor network mapping state -> deterministic continuous action."""
    inputs = layers.Input(shape=(state_dim,), name="actor_state_input")
    x = inputs
    for i, units in enumerate(hidden_sizes):
        x = layers.Dense(units, activation="relu", name=f"actor_hidden_{i}")(x)
    # Small initial weights for final layer to prevent large initial saturations
    init = tf.keras.initializers.RandomUniform(minval=-3e-3, maxval=3e-3)
    outputs = layers.Dense(action_dim, activation="tanh", kernel_initializer=init, name="actor_output")(x)
    scaled_outputs = layers.Lambda(lambda act: act * max_action)(outputs)
    return tf.keras.Model(inputs=inputs, outputs=scaled_outputs, name="Actor")


def build_critic(state_dim: int, action_dim: int,
                 hidden_sizes: tuple = (256, 256)) -> tf.keras.Model:
    """Critic network mapping (state, action) -> Q(state, action)."""
    state_input = layers.Input(shape=(state_dim,), name="critic_state_input")
    action_input = layers.Input(shape=(action_dim,), name="critic_action_input")
    concat = layers.Concatenate()([state_input, action_input])
    x = concat
    for i, units in enumerate(hidden_sizes):
        x = layers.Dense(units, activation="relu", name=f"critic_hidden_{i}")(x)
    init = tf.keras.initializers.RandomUniform(minval=-3e-3, maxval=3e-3)
    outputs = layers.Dense(1, kernel_initializer=init, name="q_value_output")(x)
    return tf.keras.Model(inputs=[state_input, action_input], outputs=outputs, name="Critic")


class DDPGAgent:
    """DDPG Agent optimized for TensorFlow-CPU."""

    def __init__(
        self,
        state_dim: int = 9,
        action_dim: int = 1,
        max_action: float = 1.0,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        buffer_capacity: int = 200000,
        actor_hidden: tuple = (256, 256),
        critic_hidden: tuple = (256, 256),
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau

        with tf.device("/CPU:0"):
            # Main networks
            self.actor = build_actor(state_dim, action_dim, max_action, hidden_sizes=actor_hidden)
            self.critic = build_critic(state_dim, action_dim, hidden_sizes=critic_hidden)

            # Target networks
            self.actor_target = build_actor(state_dim, action_dim, max_action, hidden_sizes=actor_hidden)
            self.critic_target = build_critic(state_dim, action_dim, hidden_sizes=critic_hidden)

            # Initialize target networks with exact same weights
            self.actor_target.set_weights(self.actor.get_weights())
            self.critic_target.set_weights(self.critic.get_weights())

            # Optimizers with gradient clipping for numerical stability
            self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr, clipnorm=1.0)
            self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr, clipnorm=1.0)

        # Experience replay buffer
        self.buffer = ReplayBuffer(capacity=buffer_capacity, state_dim=state_dim, action_dim=action_dim)

    def select_action(self, state: np.ndarray, noise: float = 0.0) -> np.ndarray:
        """
        Select continuous action for a given state.
        If noise > 0, adds exploration noise (used during training).
        If noise == 0, returns deterministic action (used during evaluation/visualization).
        """
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        action = self.actor(state_tensor, training=False).numpy()[0]

        if noise > 0.0:
            noise_sample = np.random.normal(0.0, noise, size=self.action_dim)
            action = action + noise_sample
            action = np.clip(action, -self.max_action, self.max_action)

        return action.astype(np.float32)

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        """Compiled TensorFlow function executing a single DDPG gradient step on CPU."""
        with tf.device("/CPU:0"):
            # 1. Update Critic
            with tf.GradientTape() as tape:
                # Target Q = r + gamma * (1 - done) * Q_target(s', mu_target(s'))
                target_actions = self.actor_target(next_states, training=False)
                target_q = self.critic_target([next_states, target_actions], training=False)
                y_target = rewards + self.gamma * (1.0 - dones) * target_q

                current_q = self.critic([states, actions], training=True)
                critic_loss = tf.reduce_mean(tf.square(y_target - current_q))

            critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

            # 2. Update Actor
            with tf.GradientTape() as tape:
                predicted_actions = self.actor(states, training=True)
                # DDPG policy objective: maximize Q(s, mu(s)) -> minimize -Q(s, mu(s))
                actor_loss = -tf.reduce_mean(self.critic([states, predicted_actions], training=False))

            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

            # 3. Soft update target networks (Polyak averaging)
            for var, target_var in zip(self.critic.variables, self.critic_target.variables):
                target_var.assign(self.tau * var + (1.0 - self.tau) * target_var)
            for var, target_var in zip(self.actor.variables, self.actor_target.variables):
                target_var.assign(self.tau * var + (1.0 - self.tau) * target_var)

        return critic_loss, actor_loss

    def train(self, batch_size: int = 128):
        """Sample mini-batch and execute training step."""
        if len(self.buffer) < batch_size:
            return None, None
        batch = self.buffer.sample(batch_size)
        return self.train_step(*batch)

    def save_weights(self, checkpoint_dir: str, prefix: str = "best"):
        """Save actor and critic weights."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        actor_path = os.path.join(checkpoint_dir, f"{prefix}_actor.weights.h5")
        critic_path = os.path.join(checkpoint_dir, f"{prefix}_critic.weights.h5")
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        return actor_path, critic_path

    def load_weights(self, checkpoint_dir: str, prefix: str = "best"):
        """Load actor and critic weights."""
        actor_path = os.path.join(checkpoint_dir, f"{prefix}_actor.weights.h5")
        critic_path = os.path.join(checkpoint_dir, f"{prefix}_critic.weights.h5")

        if os.path.exists(actor_path):
            self.actor.load_weights(actor_path)
            self.actor_target.set_weights(self.actor.get_weights())
        else:
            raise FileNotFoundError(f"Actor checkpoint not found at {actor_path}")

        if os.path.exists(critic_path):
            self.critic.load_weights(critic_path)
            self.critic_target.set_weights(self.critic.get_weights())
        return actor_path, critic_path
