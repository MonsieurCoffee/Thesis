from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    BOARD_SIZE,
    TABLE_DIR,
    dataframe_to_markdown,
    ensure_output_dirs,
    ensure_project_imports,
    quiet_tensorflow_logs,
)


INPUT_SHAPE = (BOARD_SIZE, BOARD_SIZE, 10)
NUM_ACTIONS = BOARD_SIZE * BOARD_SIZE
CONV_UNITS = 32
DENSE_UNITS = 128
BATCH_SIZE = 128
MEMORY_SIZE = 300_000


def bytes_to_mib(value: float) -> float:
    return value / (1024**2)


def theoretical_counts() -> dict[str, float]:
    conv1 = (3 * 3 * 10 + 1) * CONV_UNITS
    conv2 = (3 * 3 * CONV_UNITS + 1) * CONV_UNITS
    conv3 = conv2
    flatten_units = BOARD_SIZE * BOARD_SIZE * CONV_UNITS
    dense1 = (flatten_units + 1) * DENSE_UNITS
    dense2 = (DENSE_UNITS + 1) * DENSE_UNITS
    q_output = (DENSE_UNITS + 1) * NUM_ACTIONS
    params = conv1 + conv2 + conv3 + dense1 + dense2 + q_output

    conv1_macs = BOARD_SIZE * BOARD_SIZE * CONV_UNITS * (3 * 3 * 10)
    conv2_macs = BOARD_SIZE * BOARD_SIZE * CONV_UNITS * (3 * 3 * CONV_UNITS)
    conv3_macs = conv2_macs
    dense1_macs = flatten_units * DENSE_UNITS
    dense2_macs = DENSE_UNITS * DENSE_UNITS
    q_output_macs = DENSE_UNITS * NUM_ACTIONS
    forward_macs = (
        conv1_macs
        + conv2_macs
        + conv3_macs
        + dense1_macs
        + dense2_macs
        + q_output_macs
    )

    state_bytes = np.prod(INPUT_SHAPE) * np.dtype(np.float32).itemsize
    transition_bytes = (2 * state_bytes) + 4 + 4 + 1
    replay_bytes = transition_bytes * MEMORY_SIZE
    model_weight_bytes = params * np.dtype(np.float32).itemsize
    cae_matrix_bytes = 2 * DENSE_UNITS * DENSE_UNITS * np.dtype(np.float64).itemsize

    return {
        "conv1_params": conv1,
        "conv2_params": conv2,
        "conv3_params": conv3,
        "dense1_params": dense1,
        "dense2_or_embedding_params": dense2,
        "q_output_params": q_output,
        "total_params": params,
        "forward_macs": forward_macs,
        "state_bytes": state_bytes,
        "transition_bytes_raw": transition_bytes,
        "replay_buffer_raw_mib": bytes_to_mib(replay_bytes),
        "single_model_weight_mib": bytes_to_mib(model_weight_bytes),
        "main_plus_target_weight_mib": bytes_to_mib(2 * model_weight_bytes),
        "cae_A_and_A_inv_mib_float64": bytes_to_mib(cae_matrix_bytes),
        "cae_ucb_matrix_vector_ops": DENSE_UNITS * DENSE_UNITS,
        "cae_sherman_morrison_outer_ops": DENSE_UNITS * DENSE_UNITS,
    }


def try_tensorflow_benchmarks(iterations: int, train_iterations: int) -> dict[str, float | str]:
    quiet_tensorflow_logs()
    ensure_project_imports("dqn", "caedqn")
    try:
        import tensorflow as tf
        from dqn_model import create_dqn
        from caedqn_model import create_caedqn_model
    except Exception as exc:  # pragma: no cover - depends on local env
        return {"tensorflow_status": f"unavailable: {exc}"}

    tf.random.set_seed(123)
    np.random.seed(123)

    dqn = create_dqn(0.001, INPUT_SHAPE, NUM_ACTIONS)
    cae = create_caedqn_model(0.001, INPUT_SHAPE, NUM_ACTIONS)
    dummy_state = np.zeros((1, *INPUT_SHAPE), dtype=np.float32)
    dummy_batch = np.random.random((BATCH_SIZE, *INPUT_SHAPE)).astype(np.float32)
    next_batch = np.random.random((BATCH_SIZE, *INPUT_SHAPE)).astype(np.float32)
    actions = tf.convert_to_tensor(
        np.random.randint(0, NUM_ACTIONS, size=BATCH_SIZE), dtype=tf.int32
    )
    rewards = tf.convert_to_tensor(
        np.random.normal(size=BATCH_SIZE).astype(np.float32), dtype=tf.float32
    )
    dones = tf.convert_to_tensor(
        np.random.randint(0, 2, size=BATCH_SIZE).astype(bool), dtype=tf.bool
    )
    states_t = tf.convert_to_tensor(dummy_batch, dtype=tf.float32)
    next_t = tf.convert_to_tensor(next_batch, dtype=tf.float32)

    for _ in range(10):
        dqn.predict_on_batch(dummy_state)
        cae.predict_on_batch(dummy_state)

    dqn_forward_ms = _time_predict(dqn, dummy_state, iterations)
    cae_forward_ms = _time_predict(cae, dummy_state, iterations)

    dqn_train_ms = _time_train_step_dqn(dqn, states_t, actions, rewards, next_t, dones, train_iterations)
    cae_train_ms = _time_train_step_cae(cae, states_t, actions, rewards, next_t, dones, train_iterations)

    return {
        "tensorflow_status": "available",
        "dqn_keras_params": int(dqn.count_params()),
        "caedqn_keras_params": int(cae.count_params()),
        "dqn_forward_ms_per_state": dqn_forward_ms,
        "caedqn_forward_ms_per_state": cae_forward_ms,
        "dqn_train_step_ms_per_batch": dqn_train_ms,
        "caedqn_train_step_ms_per_batch": cae_train_ms,
        "cae_bonus_update_ms_per_action": _time_cae_bonus_update(max(5000, iterations * 10)),
    }


def _time_predict(model, state: np.ndarray, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        model.predict_on_batch(state)
    elapsed = time.perf_counter() - start
    return (elapsed / iterations) * 1000.0


def _time_train_step_dqn(model, states, actions, rewards, next_states, dones, iterations: int) -> float:
    import tensorflow as tf

    optimizer = model.optimizer
    gamma = 0.9
    for _ in range(3):
        _single_train_step_dqn(model, optimizer, states, actions, rewards, next_states, dones, gamma)
    start = time.perf_counter()
    for _ in range(iterations):
        _single_train_step_dqn(model, optimizer, states, actions, rewards, next_states, dones, gamma)
    elapsed = time.perf_counter() - start
    return (elapsed / iterations) * 1000.0


def _time_train_step_cae(model, states, actions, rewards, next_states, dones, iterations: int) -> float:
    import tensorflow as tf

    optimizer = model.optimizer
    gamma = 0.9
    for _ in range(3):
        _single_train_step_cae(model, optimizer, states, actions, rewards, next_states, dones, gamma)
    start = time.perf_counter()
    for _ in range(iterations):
        _single_train_step_cae(model, optimizer, states, actions, rewards, next_states, dones, gamma)
    elapsed = time.perf_counter() - start
    return (elapsed / iterations) * 1000.0


def _time_cae_bonus_update(iterations: int) -> float:
    rng = np.random.default_rng(123)
    phis = rng.normal(size=(iterations, DENSE_UNITS)).astype(np.float64)
    A_inv = np.eye(DENSE_UNITS, dtype=np.float64)
    start = time.perf_counter()
    for phi in phis:
        A_inv_phi = np.dot(A_inv, phi)
        _ = np.sqrt(np.dot(phi, A_inv_phi))
        denom = 1.0 + np.dot(phi, A_inv_phi)
        A_inv -= np.outer(A_inv_phi, A_inv_phi) / denom
    elapsed = time.perf_counter() - start
    return (elapsed / iterations) * 1000.0


def _single_train_step_dqn(model, optimizer, states, actions, rewards, next_states, dones, gamma: float):
    import tensorflow as tf

    future_qs = model(next_states, training=False)
    max_future_q = tf.reduce_max(future_qs, axis=1)
    targets = rewards + gamma * max_future_q * (1.0 - tf.cast(dones, tf.float32))
    with tf.GradientTape() as tape:
        q_values = model(states, training=True)
        actions_one_hot = tf.one_hot(actions, depth=NUM_ACTIONS)
        selected_q = tf.reduce_sum(q_values * actions_one_hot, axis=1)
        loss = tf.reduce_mean(tf.square(targets - selected_q))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def _single_train_step_cae(model, optimizer, states, actions, rewards, next_states, dones, gamma: float):
    import tensorflow as tf

    future_qs, _ = model(next_states, training=False)
    max_future_q = tf.reduce_max(future_qs, axis=1)
    targets = rewards + gamma * max_future_q * (1.0 - tf.cast(dones, tf.float32))
    with tf.GradientTape() as tape:
        q_values, _ = model(states, training=True)
        actions_one_hot = tf.one_hot(actions, depth=NUM_ACTIONS)
        selected_q = tf.reduce_sum(q_values * actions_one_hot, axis=1)
        loss = tf.reduce_mean(tf.square(targets - selected_q))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def build_tables(benchmarks: dict[str, float | str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = theoretical_counts()

    complexity_rows = [
        {
            "algorithm": "DQN",
            "trainable_params": counts["total_params"],
            "forward_macs_per_state": counts["forward_macs"],
            "extra_cae_ops_per_action": 0,
            "single_model_weight_mib": counts["single_model_weight_mib"],
            "main_plus_target_weight_mib": counts["main_plus_target_weight_mib"],
            "extra_cae_matrix_memory_mib": 0.0,
            "network_forward_ms_per_state": benchmarks.get("dqn_forward_ms_per_state", pd.NA),
            "network_train_step_ms_per_batch": benchmarks.get("dqn_train_step_ms_per_batch", pd.NA),
            "cae_bonus_update_ms_per_action": 0.0,
        },
        {
            "algorithm": "CAE-DQN",
            "trainable_params": counts["total_params"],
            "forward_macs_per_state": counts["forward_macs"],
            "extra_cae_ops_per_action": counts["cae_ucb_matrix_vector_ops"]
            + counts["cae_sherman_morrison_outer_ops"],
            "single_model_weight_mib": counts["single_model_weight_mib"],
            "main_plus_target_weight_mib": counts["main_plus_target_weight_mib"],
            "extra_cae_matrix_memory_mib": counts["cae_A_and_A_inv_mib_float64"],
            "network_forward_ms_per_state": benchmarks.get("caedqn_forward_ms_per_state", pd.NA),
            "network_train_step_ms_per_batch": benchmarks.get("caedqn_train_step_ms_per_batch", pd.NA),
            "cae_bonus_update_ms_per_action": benchmarks.get("cae_bonus_update_ms_per_action", pd.NA),
        },
    ]

    memory_rows = [
        {
            "item": "single_state_float32",
            "value": counts["state_bytes"],
            "unit": "bytes",
            "notes": "6x6x10 one-hot observation",
        },
        {
            "item": "single_transition_raw",
            "value": counts["transition_bytes_raw"],
            "unit": "bytes",
            "notes": "state, next_state, action, reward, done; excludes Python overhead",
        },
        {
            "item": "replay_buffer_raw",
            "value": counts["replay_buffer_raw_mib"],
            "unit": "MiB",
            "notes": f"{MEMORY_SIZE:,} transitions, excludes deque/object overhead",
        },
        {
            "item": "cae_A_and_A_inv",
            "value": counts["cae_A_and_A_inv_mib_float64"],
            "unit": "MiB",
            "notes": "two 128x128 float64 matrices",
        },
        {
            "item": "tensorflow_status",
            "value": benchmarks.get("tensorflow_status", "not checked"),
            "unit": "",
            "notes": "Timing columns are filled only when TensorFlow is available",
        },
    ]

    complexity_df = pd.DataFrame(complexity_rows)
    memory_df = pd.DataFrame(memory_rows)
    for column in [
        "single_model_weight_mib",
        "main_plus_target_weight_mib",
        "extra_cae_matrix_memory_mib",
        "network_forward_ms_per_state",
        "network_train_step_ms_per_batch",
        "cae_bonus_update_ms_per_action",
    ]:
        if column in complexity_df.columns:
            try:
                complexity_df[column] = pd.to_numeric(complexity_df[column])
            except Exception:
                pass
    complexity_df = complexity_df.round(
        {
            "single_model_weight_mib": 4,
            "main_plus_target_weight_mib": 4,
            "extra_cae_matrix_memory_mib": 4,
            "network_forward_ms_per_state": 4,
            "network_train_step_ms_per_batch": 4,
            "cae_bonus_update_ms_per_action": 4,
        }
    )
    memory_df["value"] = memory_df["value"].map(
        lambda value: round(value, 4) if isinstance(value, (int, float)) else value
    )
    return complexity_df, memory_df


def write_markdown(path: Path, complexity_df: pd.DataFrame, memory_df: pd.DataFrame) -> None:
    lines = [
        "# Complexity Analysis",
        "",
        "Both DQN and CAE-DQN use the same CNN Q-network. CAE-DQN reuses the 128-dimensional embedding layer for uncertainty estimation, so it does not add trainable neural-network parameters. Its additional cost comes from the Gram matrix and UCB bonus computation.",
        "",
        "## Model Complexity",
        "",
        dataframe_to_markdown(complexity_df),
        "",
        "## Memory Estimate",
        "",
        dataframe_to_markdown(memory_df),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute DQN/CAE-DQN complexity tables.")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--train-iterations", type=int, default=30)
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument(
        "--complexity-output",
        type=Path,
        default=TABLE_DIR / "complexity_analysis.csv",
    )
    parser.add_argument(
        "--memory-output",
        type=Path,
        default=TABLE_DIR / "complexity_memory_estimate.csv",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=TABLE_DIR / "complexity_analysis.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    benchmarks = (
        {"tensorflow_status": "skipped by --skip-timing"}
        if args.skip_timing
        else try_tensorflow_benchmarks(args.iterations, args.train_iterations)
    )
    complexity_df, memory_df = build_tables(benchmarks)
    complexity_df.to_csv(args.complexity_output, index=False)
    memory_df.to_csv(args.memory_output, index=False)
    write_markdown(args.markdown_output, complexity_df, memory_df)

    print(f"Saved complexity table to {args.complexity_output}")
    print(f"Saved memory estimate to {args.memory_output}")
    print(f"Saved Markdown notes to {args.markdown_output}")


if __name__ == "__main__":
    main()
