import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

import random
from collections import deque

import numpy as np
import tensorflow as tf

from shared import config
from legacy.dqn_agent import _ensure_runtime_compatibility


class GraphQNetwork(tf.keras.Model):
    """Lightweight GCN + dueling Q head for graph-based scheduling."""

    def __init__(self, node_feature_dim, compute_node_indices, action_dim):
        super().__init__()
        self.compute_node_indices = tf.constant(compute_node_indices, dtype=tf.int32)
        self.action_dim = action_dim

        self.input_proj = tf.keras.layers.Dense(config.GNN_HIDDEN_DIM, activation='relu',
                                                kernel_initializer='he_uniform')
        self.gcn_1 = tf.keras.layers.Dense(config.GNN_HIDDEN_DIM, activation='relu',
                                           kernel_initializer='he_uniform')
        self.gcn_2 = tf.keras.layers.Dense(config.GNN_EMBED_DIM, activation='relu',
                                           kernel_initializer='he_uniform')

        self.value_head = tf.keras.Sequential([
            tf.keras.layers.Dense(config.GNN_DUELING_DIM, activation='relu', kernel_initializer='he_uniform'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        self.node_adv_head = tf.keras.Sequential([
            tf.keras.layers.Dense(config.GNN_DUELING_DIM, activation='relu', kernel_initializer='he_uniform'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        self.wait_adv_head = tf.keras.Sequential([
            tf.keras.layers.Dense(config.GNN_DUELING_DIM, activation='relu', kernel_initializer='he_uniform'),
            tf.keras.layers.Dense(1, activation='linear')
        ])

        self._node_feature_dim = node_feature_dim

    def call(self, inputs, training=False):
        node_features = inputs["node_features"]
        adjacency = inputs["adjacency"]

        h = self.input_proj(node_features)
        h = tf.matmul(adjacency, h)
        h = self.gcn_1(h)
        h = tf.matmul(adjacency, h)
        h = self.gcn_2(h)

        graph_embedding = tf.reduce_mean(h, axis=1)
        value = self.value_head(graph_embedding)

        compute_embeddings = tf.gather(h, self.compute_node_indices, axis=1)
        node_advantages = tf.squeeze(self.node_adv_head(compute_embeddings), axis=-1)
        wait_advantage = self.wait_adv_head(graph_embedding)
        advantages = tf.concat([node_advantages, wait_advantage], axis=1)

        return value + (advantages - tf.reduce_mean(advantages, axis=1, keepdims=True))


class GNNAgent:
    """GNN-based Double DQN agent with action masking.

    The environment still owns the action semantics:
    action 0..N-1 maps to compute nodes, action N maps to WAIT.
    """

    def __init__(self, graph_state_template, action_dim=None):
        _ensure_runtime_compatibility()
        self.action_dim = action_dim if action_dim is not None else config.NODE_NUM
        self.node_feature_dim = graph_state_template["node_features"].shape[1]
        self.node_count = graph_state_template["node_features"].shape[0]
        self.compute_node_indices = np.asarray(graph_state_template["compute_node_indices"], dtype=np.int32)
        self.memory = deque(maxlen=config.MEMORY_CAPACITY)
        self.epsilon = config.EPSILON_START

        self.model = GraphQNetwork(self.node_feature_dim, self.compute_node_indices, self.action_dim)
        self.target_model = GraphQNetwork(self.node_feature_dim, self.compute_node_indices, self.action_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE, clipnorm=1.0)
        self.loss_fn = tf.keras.losses.Huber()
        self._build_networks(graph_state_template)
        self.update_target_model()

    def _build_networks(self, graph_state_template):
        batch = self._state_batch([graph_state_template])
        self.model(batch, training=False)
        self.target_model(batch, training=False)

    @staticmethod
    def _copy_state(state):
        return {
            "node_features": np.asarray(state["node_features"], dtype=np.float32).copy(),
            "adjacency": np.asarray(state["adjacency"], dtype=np.float32).copy(),
            "compute_node_indices": np.asarray(state["compute_node_indices"], dtype=np.int32).copy(),
            "wait_action_index": int(state.get("wait_action_index", len(state["compute_node_indices"])))
        }

    @staticmethod
    def _state_batch(states):
        return {
            "node_features": tf.convert_to_tensor(
                np.stack([s["node_features"] for s in states]), dtype=tf.float32
            ),
            "adjacency": tf.convert_to_tensor(
                np.stack([s["adjacency"] for s in states]), dtype=tf.float32
            )
        }

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state, valid_actions=None):
        if np.random.rand() <= self.epsilon:
            if valid_actions is not None and len(valid_actions) > 0:
                return random.choice(valid_actions)
            return random.randint(0, self.action_dim - 1)

        q_values = self.model(self._state_batch([state]), training=False).numpy()[0]
        if valid_actions is not None and len(valid_actions) > 0:
            mask = np.ones(self.action_dim, dtype=bool)
            mask[valid_actions] = False
            q_values[mask] = -np.inf
        return int(np.argmax(q_values))

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

    def remember(self, state, action, reward, next_state, done,
                 valid_actions=None, next_valid_actions=None, info=None):
        self.memory.append((
            self._copy_state(state),
            int(action),
            float(reward),
            self._copy_state(next_state),
            bool(done),
            list(valid_actions) if valid_actions is not None else None,
            list(next_valid_actions) if next_valid_actions is not None else None
        ))

    def replay(self):
        if len(self.memory) < config.BATCH_SIZE:
            return None

        minibatch = random.sample(self.memory, config.BATCH_SIZE)
        states = [item[0] for item in minibatch]
        actions = np.asarray([item[1] for item in minibatch], dtype=np.int32)
        rewards = np.asarray([item[2] for item in minibatch], dtype=np.float32)
        next_states = [item[3] for item in minibatch]
        dones = np.asarray([item[4] for item in minibatch], dtype=np.float32)
        next_valid_actions = [item[6] for item in minibatch]

        state_batch = self._state_batch(states)
        next_state_batch = self._state_batch(next_states)

        next_q_current = self.model(next_state_batch, training=False).numpy()
        next_q_current = self._mask_q_values(next_q_current, next_valid_actions)
        best_next_actions = np.argmax(next_q_current, axis=1)
        next_q_target = self.target_model(next_state_batch, training=False).numpy()
        batch_indices = np.arange(config.BATCH_SIZE)
        td_targets = rewards + (1.0 - dones) * config.GAMMA * next_q_target[batch_indices, best_next_actions]

        with tf.GradientTape() as tape:
            q_values = self.model(state_batch, training=True)
            action_indices = tf.stack([
                tf.range(config.BATCH_SIZE, dtype=tf.int32),
                tf.convert_to_tensor(actions, dtype=tf.int32)
            ], axis=1)
            chosen_q = tf.gather_nd(q_values, action_indices)
            loss = self.loss_fn(tf.convert_to_tensor(td_targets, dtype=tf.float32), chosen_q)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        if self.epsilon > config.EPSILON_MIN:
            self.epsilon *= config.EPSILON_DECAY

        return float(loss.numpy())

    def save(self, filepath):
        weights_path = filepath if filepath.endswith(".weights.h5") else f"{filepath}.weights.h5"
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        self.model.save_weights(weights_path)
        print(f"GNN 模型权重已保存至: {weights_path}")

    def load(self, filepath):
        candidates = [filepath, f"{filepath}.weights.h5"]
        for candidate in candidates:
            if os.path.exists(candidate):
                self.model.load_weights(candidate)
                self.update_target_model()
                self.epsilon = 0.0
                print(f"GNN 模型权重已加载: {candidate}")
                return
        print(f"错误：找不到 GNN 模型权重 {filepath}")
