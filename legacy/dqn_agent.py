import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

import random
from collections import deque

import numpy as np
import tensorflow as tf
from google.protobuf import __version__ as PROTOBUF_VERSION

from shared import config


def _parse_major_minor(version: str):
    parts = []
    for token in version.split('.'):
        digits = []
        for char in token:
            if char.isdigit():
                digits.append(char)
            else:
                break
        if not digits:
            break
        parts.append(int(''.join(digits)))
        if len(parts) == 2:
            break
    while len(parts) < 2:
        parts.append(0)
    return tuple(parts[:2])


def _ensure_runtime_compatibility():
    if _parse_major_minor(tf.__version__) < (2, 11) and _parse_major_minor(PROTOBUF_VERSION)[0] >= 4:
        raise RuntimeError(
            f"Detected incompatible TensorFlow/protobuf versions: tensorflow {tf.__version__}, "
            f"protobuf {PROTOBUF_VERSION}. For TensorFlow < 2.11, please install protobuf==3.20.*."
        )


class PrioritizedReplayBuffer:
    """Proportional prioritized replay with lightweight numpy sampling."""

    def __init__(self, capacity, alpha=0.6, epsilon=1e-5):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.buffer = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, experience, priority=None):
        if priority is None:
            current = self.priorities[:len(self.buffer)]
            priority = float(np.max(current)) if len(current) > 0 else 1.0
        priority = max(float(priority), self.epsilon)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta):
        size = len(self.buffer)
        priorities = self.priorities[:size]
        scaled = np.power(np.maximum(priorities, self.epsilon), self.alpha)
        probs = scaled / np.sum(scaled)

        indices = np.random.choice(size, batch_size, replace=size < batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        weights = np.power(size * probs[indices], -beta)
        weights = weights / np.max(weights)
        return samples, indices, weights.astype(np.float32)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = max(float(priority), self.epsilon)


class DQNAgent:
    def __init__(self, state_dim=None, action_dim=None):
        _ensure_runtime_compatibility()
        self.state_dim = state_dim if state_dim is not None else config.INPUT_DIM
        self.action_dim = action_dim if action_dim is not None else config.NODE_NUM

        self.use_per = getattr(config, 'PER_ENABLED', True)
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(
                config.MEMORY_CAPACITY,
                alpha=getattr(config, 'PER_ALPHA', 0.6),
                epsilon=getattr(config, 'PER_EPSILON', 1e-5)
            )
        else:
            self.memory = deque(maxlen=config.MEMORY_CAPACITY)

        self.epsilon = config.EPSILON_START
        self.replay_steps = 0

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        inputs = tf.keras.Input(shape=(self.state_dim,), name='state')
        x = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform')(inputs)
        x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform')(x)

        value = tf.keras.layers.Dense(
            getattr(config, 'DUELING_HIDDEN_DIM', 256),
            activation='relu',
            kernel_initializer='he_uniform'
        )(x)
        value = tf.keras.layers.Dense(1, activation='linear', name='state_value')(value)

        advantage = tf.keras.layers.Dense(
            getattr(config, 'DUELING_HIDDEN_DIM', 256),
            activation='relu',
            kernel_initializer='he_uniform'
        )(x)
        advantage = tf.keras.layers.Dense(self.action_dim, activation='linear', name='action_advantage')(advantage)

        centered_advantage = tf.keras.layers.Lambda(
            lambda a: a - tf.reduce_mean(a, axis=1, keepdims=True),
            name='centered_advantage'
        )(advantage)
        outputs = tf.keras.layers.Add(name='dueling_q')([value, centered_advantage])

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='DuelingDoubleDQN')
        model.compile(
            loss=tf.keras.losses.Huber(),
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config.LEARNING_RATE,
                clipnorm=1.0
            )
        )
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state, valid_actions=None):
        if np.random.rand() <= self.epsilon:
            if valid_actions is not None and len(valid_actions) > 0:
                return random.choice(valid_actions)
            return random.randint(0, self.action_dim - 1)

        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        q_values = self.model(state_tensor, training=False).numpy()[0]
        q_values = self._mask_q_values(q_values[None, :], [valid_actions])[0]
        return int(np.argmax(q_values))

    def _initial_priority(self, reward, info):
        priority = abs(float(reward)) + 1.0
        if not isinstance(info, dict):
            return priority

        status = info.get('status')
        if status not in (None, 'Success', 'Deferred'):
            priority += getattr(config, 'PER_FAILURE_PRIORITY_BOOST', 4.0)

        constraint_costs = info.get('constraint_costs', {})
        if constraint_costs:
            priority += getattr(config, 'PER_CONSTRAINT_PRIORITY_BOOST', 3.0) * sum(
                float(v) for v in constraint_costs.values()
            )

        components = info.get('reward_components', {})
        if components:
            cec_proxy = np.mean([
                max(0.0, float(components.get('R_green', 0.0))),
                max(0.0, float(components.get('R_cost', 0.0))),
                max(0.0, float(components.get('R_balance', 0.0))),
                max(0.0, float(components.get('R_success', 0.0))),
            ])
            if cec_proxy < 0.25:
                priority += getattr(config, 'PER_LOW_CECI_PRIORITY_BOOST', 2.0)

        return priority

    def remember(self, state, action, reward, next_state, done,
                 valid_actions=None, next_valid_actions=None, info=None):
        experience = (
            np.asarray(state, dtype=np.float32).copy(),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32).copy(),
            bool(done),
            list(valid_actions) if valid_actions is not None else None,
            list(next_valid_actions) if next_valid_actions is not None else None,
        )

        if self.use_per:
            self.memory.add(experience, priority=self._initial_priority(reward, info))
        else:
            self.memory.append(experience)

    @staticmethod
    def _mask_q_values(q_values, valid_actions_batch):
        masked = np.array(q_values, copy=True)
        if valid_actions_batch is None:
            return masked

        for row, valid_actions in enumerate(valid_actions_batch):
            if valid_actions is None:
                continue
            if len(valid_actions) == 0:
                masked[row, :] = -np.inf
                continue
            mask = np.ones(masked.shape[1], dtype=bool)
            mask[np.asarray(valid_actions, dtype=np.int32)] = False
            masked[row, mask] = -np.inf
        return masked

    def _current_beta(self):
        beta_start = getattr(config, 'PER_BETA_START', 0.4)
        beta_frames = max(1, getattr(config, 'PER_BETA_FRAMES', 120000))
        progress = min(1.0, self.replay_steps / beta_frames)
        return beta_start + progress * (1.0 - beta_start)

    def replay(self):
        if len(self.memory) < config.BATCH_SIZE:
            return None

        self.replay_steps += 1
        if self.use_per:
            minibatch, sample_indices, sample_weights = self.memory.sample(
                config.BATCH_SIZE,
                beta=self._current_beta()
            )
        else:
            minibatch = random.sample(self.memory, config.BATCH_SIZE)
            sample_indices = None
            sample_weights = np.ones(config.BATCH_SIZE, dtype=np.float32)

        states = np.asarray([item[0] for item in minibatch], dtype=np.float32)
        actions = np.asarray([item[1] for item in minibatch], dtype=np.int32)
        rewards = np.asarray([item[2] for item in minibatch], dtype=np.float32)
        next_states = np.asarray([item[3] for item in minibatch], dtype=np.float32)
        dones = np.asarray([item[4] for item in minibatch], dtype=np.float32)
        next_valid_actions = [item[6] for item in minibatch]

        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)

        targets = self.model(states_tensor, training=False).numpy()

        next_q_current = self.model(next_states_tensor, training=False).numpy()
        next_q_current = self._mask_q_values(next_q_current, next_valid_actions)
        best_next_actions = np.argmax(next_q_current, axis=1)

        next_q_target = self.target_model(next_states_tensor, training=False).numpy()
        batch_indices = np.arange(config.BATCH_SIZE)
        max_next_q = next_q_target[batch_indices, best_next_actions]
        td_targets = rewards + (1.0 - dones) * config.GAMMA * max_next_q

        td_errors = td_targets - targets[batch_indices, actions]
        targets[batch_indices, actions] = td_targets

        loss = self.model.train_on_batch(states, targets, sample_weight=sample_weights)

        if self.use_per:
            self.memory.update_priorities(
                sample_indices,
                np.abs(td_errors) + getattr(config, 'PER_EPSILON', 1e-5)
            )

        if self.epsilon > config.EPSILON_MIN:
            self.epsilon = max(config.EPSILON_MIN, self.epsilon * config.EPSILON_DECAY)

        return loss[0] if isinstance(loss, list) else float(loss)

    def save(self, filepath):
        self.model.save(filepath)
        print(f"Model saved to: {filepath}")

    def load(self, filepath):
        if os.path.exists(filepath):
            try:
                self.model = tf.keras.models.load_model(filepath, compile=False)
                self.model.compile(
                    loss=tf.keras.losses.Huber(),
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=config.LEARNING_RATE,
                        clipnorm=1.0
                    )
                )
                self.target_model = tf.keras.models.load_model(filepath, compile=False)
                self.target_model.compile(
                    loss=tf.keras.losses.Huber(),
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=config.LEARNING_RATE,
                        clipnorm=1.0
                    )
                )
                self.update_target_model()
                self.epsilon = 0.0
                print(f"Model loaded from: {filepath}")
            except Exception as exc:
                raise RuntimeError(f"Failed to load model: {filepath}") from exc
        else:
            print(f"Error: model file not found {filepath}")
