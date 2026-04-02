# coding=utf-8
# Copyright 2023 The Google Research Authors.
# Modifications copyright 2024 Kevin Ahrendt.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import platform
import contextlib
import random
import multiprocessing as mp
import json

from absl import logging

import numpy as np
import tensorflow as tf

import microwakeword.data as data_lib


@contextlib.contextmanager
def swap_attribute(obj, attr, temp_value):
    """Temporarily swap an attribute of an object."""
    original_value = getattr(obj, attr)
    setattr(obj, attr, temp_value)

    try:
        yield
    finally:
        setattr(obj, attr, original_value)


def _has_visible_gpu() -> bool:
    try:
        return bool(tf.config.list_logical_devices("GPU"))
    except Exception:
        return False


def nonstreaming_eval_batch_size(
    config=None,
    data_set: str = "",
    truncation_strategy: str = "",
) -> int:
    raw_value = os.environ.get("MICRO_NONSTREAMING_EVAL_BATCH_SIZE", "auto").strip()
    value = raw_value.lower()
    if value in {"", "auto"}:
        train_batch_size = 0
        if isinstance(config, dict):
            try:
                train_batch_size = max(0, int(config.get("batch_size", 0)))
            except (TypeError, ValueError):
                train_batch_size = 0
        if _has_visible_gpu():
            auto_cap = env_int(
                "MICRO_NONSTREAMING_EVAL_BATCH_SIZE_AUTO_MAX",
                65536,
                minimum=1024,
            )
            auto_batch = max(4096, train_batch_size * 256 if train_batch_size > 0 else 16384)
            if truncation_strategy == "split" or data_set.endswith("_ambient"):
                auto_batch = max(
                    auto_batch,
                    env_int(
                        "MICRO_NONSTREAMING_EVAL_SPLIT_BATCH_SIZE_AUTO",
                        65536,
                        minimum=4096,
                    ),
                )
            return min(auto_batch, auto_cap)
        return 1024
    try:
        return max(1, int(raw_value))
    except ValueError:
        logging.warning(
            "Invalid MICRO_NONSTREAMING_EVAL_BATCH_SIZE=%r, falling back to auto",
            raw_value,
        )
        return nonstreaming_eval_batch_size(config, data_set=data_set, truncation_strategy=truncation_strategy)


def env_int(name: str, default: int, minimum: int = 0) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return max(minimum, int(value))
    except ValueError:
        logging.warning("Invalid %s=%r, falling back to %d", name, value, default)
        return default


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def env_float(name: str, default: float, minimum: float = 0.0) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return max(minimum, float(value))
    except ValueError:
        logging.warning("Invalid %s=%r, falling back to %.6f", name, value, default)
        return default


def apply_optional_device_prefetch(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Optionally copy batches to a device before consumption by the model."""
    prefetch_device = os.environ.get("MICRO_TRAIN_PREFETCH_DEVICE", "off").strip()
    prefetch_mode = prefetch_device.lower()
    if prefetch_mode in {"", "off", "none", "0", "false"}:
        return dataset

    target_device = None
    if prefetch_mode in {"auto", "gpu"}:
        gpus = tf.config.list_logical_devices("GPU")
        if gpus:
            target_device = gpus[0].name
        elif prefetch_mode == "gpu":
            logging.warning(
                "MICRO_TRAIN_PREFETCH_DEVICE=%s requested but no GPU found",
                prefetch_device,
            )
            return dataset
    elif prefetch_mode == "cpu":
        target_device = "/CPU:0"
    else:
        target_device = prefetch_device

    if not target_device:
        return dataset

    try:
        dataset = dataset.apply(tf.data.experimental.copy_to_device(target_device))
        dataset = dataset.prefetch(1)
        logging.info("Training device prefetch enabled: %s", target_device)
        return dataset
    except Exception as exc:
        logging.warning(
            "Failed to enable training device prefetch on %s: %s",
            target_device,
            exc,
        )
        return dataset


def make_streaming_eval_dataset(
    config,
    data_processor,
    data_set: str,
    truncation_strategy: str,
    max_samples: int = 0,
    repeat: bool = False,
):
    feature_shape = tuple(config["training_input_shape"])

    def sample_generator():
        if max_samples > 0:
            providers = list(data_processor.feature_providers)
            random.shuffle(providers)
            iterators = [
                (
                    provider,
                    iter(
                        provider.get_feature_generator(
                            data_set,
                            features_length=config["spectrogram_length"],
                            truncation_strategy=truncation_strategy,
                        )
                    ),
                )
                for provider in providers
            ]
            emitted = 0
            while iterators and emitted < max_samples:
                next_iterators = []
                for provider, generator in iterators:
                    try:
                        spectrogram = next(generator)
                    except StopIteration:
                        continue
                    yield (
                        np.asarray(spectrogram, dtype=np.float32),
                        np.asarray([provider.label], dtype=np.float32),
                    )
                    emitted += 1
                    if emitted >= max_samples:
                        break
                    next_iterators.append((provider, generator))
                iterators = next_iterators
        else:
            for provider in data_processor.feature_providers:
                generator = provider.get_feature_generator(
                    data_set,
                    features_length=config["spectrogram_length"],
                    truncation_strategy=truncation_strategy,
                )
                for spectrogram in generator:
                    yield (
                        np.asarray(spectrogram, dtype=np.float32),
                        np.asarray([provider.label], dtype=np.float32),
                    )

    dataset = tf.data.Dataset.from_generator(
        sample_generator,
        output_signature=(
            tf.TensorSpec(shape=feature_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
        ),
    )
    dataset = dataset.batch(nonstreaming_eval_batch_size())
    if repeat:
        dataset = dataset.repeat()
    return dataset.prefetch(tf.data.AUTOTUNE)


def eval_steps_for_mode(data_processor, data_set: str, max_samples: int = 0) -> int:
    """Compute deterministic evaluate() steps for finite datasets."""
    total_samples = int(data_processor.get_mode_size(data_set))
    if max_samples > 0:
        total_samples = min(total_samples, int(max_samples))
    total_samples = max(1, total_samples)
    return int(np.ceil(total_samples / float(nonstreaming_eval_batch_size())))


def full_validation_is_better(
    current_metrics: dict,
    target_minimization: float,
    minimization_metric: str | None,
    maximization_metric: str,
    best_minimization_quantity: float,
    best_maximization_quantity: float,
) -> tuple[bool, float, float, float]:
    current_minimization_quantity = 0.0
    if minimization_metric is not None:
        current_minimization_quantity = float(current_metrics[minimization_metric])
    current_maximization_quantity = float(current_metrics[maximization_metric])
    current_no_faph_cutoff = float(current_metrics["cutoff_for_no_faph"])

    improved = (
        (
            (current_minimization_quantity <= target_minimization)
            and (
                (current_maximization_quantity > best_maximization_quantity)
                or (best_minimization_quantity > target_minimization)
            )
        )
        or (
            (current_minimization_quantity > target_minimization)
            and (current_minimization_quantity < best_minimization_quantity)
        )
        or (
            (current_minimization_quantity == best_minimization_quantity)
            and (current_maximization_quantity > best_maximization_quantity)
        )
    )
    return (
        improved,
        current_minimization_quantity,
        current_maximization_quantity,
        current_no_faph_cutoff,
    )


def _iter_nonstreaming_eval_batches_for_providers(
    selected_providers,
    data_set: str,
    feature_shape,
    features_length: int,
    truncation_strategy: str,
    batch_size: int,
    max_samples: int = 0,
):
    batch_x = []
    batch_y = []
    if max_samples > 0:
        providers = list(selected_providers)
        random.shuffle(providers)
        iterators = [
            (
                provider,
                iter(
                    provider.get_feature_generator(
                        data_set,
                        features_length=features_length,
                        truncation_strategy=truncation_strategy,
                    )
                ),
            )
            for provider in providers
        ]
        emitted = 0
        while iterators and emitted < max_samples:
            next_iterators = []
            for provider, generator in iterators:
                try:
                    spectrogram = next(generator)
                except StopIteration:
                    continue
                spectrogram_np = np.asarray(spectrogram, dtype=np.float32)
                if spectrogram_np.shape != feature_shape:
                    spectrogram_np = np.asarray(
                        data_lib.fixed_length_spectrogram(
                            spectrogram_np,
                            features_length,
                            truncation_strategy,
                            0,
                        ),
                        dtype=np.float32,
                    )
                batch_x.append(np.array(spectrogram_np, dtype=np.float32, copy=True))
                batch_y.append(float(provider.label))
                emitted += 1
                if len(batch_x) == batch_size:
                    yield (
                        np.stack(batch_x, axis=0),
                        np.asarray(batch_y, dtype=np.float32).reshape(-1, 1),
                    )
                    batch_x = []
                    batch_y = []
                if emitted >= max_samples:
                    break
                next_iterators.append((provider, generator))
            iterators = next_iterators
    else:
        for provider in selected_providers:
            generator = provider.get_feature_generator(
                data_set,
                features_length=features_length,
                truncation_strategy=truncation_strategy,
            )
            for spectrogram in generator:
                spectrogram_np = np.asarray(spectrogram, dtype=np.float32)
                if spectrogram_np.shape != feature_shape:
                    spectrogram_np = np.asarray(
                        data_lib.fixed_length_spectrogram(
                            spectrogram_np,
                            features_length,
                            truncation_strategy,
                            0,
                        ),
                        dtype=np.float32,
                    )
                batch_x.append(np.array(spectrogram_np, dtype=np.float32, copy=True))
                batch_y.append(float(provider.label))
                if len(batch_x) == batch_size:
                    yield (
                        np.stack(batch_x, axis=0),
                        np.asarray(batch_y, dtype=np.float32).reshape(-1, 1),
                    )
                    batch_x = []
                    batch_y = []

    if batch_x:
        yield (
            np.stack(batch_x, axis=0),
            np.asarray(batch_y, dtype=np.float32).reshape(-1, 1),
        )


def _nonstreaming_eval_worker_loop(
    result_q,
    provider_chunk,
    data_set: str,
    feature_shape,
    features_length: int,
    truncation_strategy: str,
    batch_size: int,
    max_samples: int,
):
    try:
        for batch in _iter_nonstreaming_eval_batches_for_providers(
            provider_chunk,
            data_set=data_set,
            feature_shape=feature_shape,
            features_length=features_length,
            truncation_strategy=truncation_strategy,
            batch_size=batch_size,
            max_samples=max_samples,
        ):
            result_q.put(batch)
    except Exception as exc:
        result_q.put(("__validation_worker_error__", repr(exc)))
    finally:
        result_q.put(None)


def iter_nonstreaming_eval_batches(
    config,
    data_processor,
    data_set: str,
    truncation_strategy: str,
    max_samples: int = 0,
):
    """Yield copied NumPy batches for nonstreaming eval without tf.data/from_generator."""
    feature_shape = tuple(config["training_input_shape"])
    features_length = int(config["spectrogram_length"])
    batch_size = nonstreaming_eval_batch_size(
        config,
        data_set=data_set,
        truncation_strategy=truncation_strategy,
    )
    providers = [
        provider
        for provider in data_processor.feature_providers
        if provider.get_mode_size(data_set)
    ]

    def iter_samples_for_providers(selected_providers):
        for provider in selected_providers:
            generator = provider.get_feature_generator(
                data_set,
                features_length=features_length,
                truncation_strategy=truncation_strategy,
            )
            for spectrogram in generator:
                yield provider, spectrogram

    default_workers = os.cpu_count() or 1
    eval_workers = env_int(
        "MICRO_TRAIN_VALIDATION_WORKERS",
        default_workers,
        minimum=0,
    )
    if max_samples > 0:
        eval_workers = min(eval_workers, max_samples)
    if eval_workers <= 1 or len(providers) <= 1:
        if eval_workers <= 1:
            logging.info(
                "Validation multiprocessing disabled: workers=%d dataset=%s",
                eval_workers,
                data_set,
            )
        for batch in _iter_nonstreaming_eval_batches_for_providers(
            providers,
            data_set=data_set,
            feature_shape=feature_shape,
            features_length=features_length,
            truncation_strategy=truncation_strategy,
            batch_size=batch_size,
            max_samples=max_samples,
        ):
            yield batch
        return

    queue_factor = env_int("MICRO_TRAIN_VALIDATION_QUEUE_FACTOR", 2, minimum=1)
    start_method = os.environ.get(
        "MICRO_TRAIN_VALIDATION_START_METHOD",
        os.environ.get("MICRO_TRAIN_DATA_START_METHOD", "fork"),
    ).strip().lower()
    if start_method not in {"fork", "spawn", "forkserver"}:
        logging.warning(
            "Invalid MICRO_TRAIN_VALIDATION_START_METHOD=%r, falling back to 'fork'",
            start_method,
        )
        start_method = "fork"

    worker_count = max(1, min(eval_workers, len(providers)))
    provider_chunks = [[] for _ in range(worker_count)]
    chunk_sizes = [0 for _ in range(worker_count)]
    for provider in sorted(
        providers,
        key=lambda current: int(current.get_mode_size(data_set)),
        reverse=True,
    ):
        idx = min(range(worker_count), key=lambda current: chunk_sizes[current])
        provider_chunks[idx].append(provider)
        chunk_sizes[idx] += max(1, int(provider.get_mode_size(data_set)))
    provider_chunks = [chunk for chunk in provider_chunks if chunk]
    worker_count = len(provider_chunks)
    worker_limits = [0 for _ in range(worker_count)]
    if max_samples > 0 and worker_count > 0:
        base_limit = max_samples // worker_count
        remainder = max_samples % worker_count
        worker_limits = [
            base_limit + (1 if index < remainder else 0)
            for index in range(worker_count)
        ]
        active = [
            (chunk, limit)
            for chunk, limit in zip(provider_chunks, worker_limits)
            if limit > 0
        ]
        provider_chunks = [chunk for chunk, _ in active]
        worker_limits = [limit for _, limit in active]
        worker_count = len(provider_chunks)
    if worker_count <= 1:
        for batch in _iter_nonstreaming_eval_batches_for_providers(
            providers,
            data_set=data_set,
            feature_shape=feature_shape,
            features_length=features_length,
            truncation_strategy=truncation_strategy,
            batch_size=batch_size,
            max_samples=max_samples,
        ):
            yield batch
        return

    queue_depth = max(worker_count * queue_factor, worker_count + 1)
    ctx = mp.get_context(start_method)
    result_q = ctx.Queue(maxsize=queue_depth)
    workers = []
    logging.info(
        "Validation multiprocessing enabled: workers=%d dataset=%s start_method=%s queue_depth=%d",
        worker_count,
        data_set,
        start_method,
        queue_depth,
    )
    try:
        for provider_chunk, worker_limit in zip(
            provider_chunks,
            worker_limits if max_samples > 0 else [0] * worker_count,
        ):
            proc = ctx.Process(
                target=_nonstreaming_eval_worker_loop,
                args=(
                    result_q,
                    provider_chunk,
                    data_set,
                    feature_shape,
                    features_length,
                    truncation_strategy,
                    batch_size,
                    worker_limit,
                ),
                daemon=True,
            )
            proc.start()
            workers.append(proc)

        finished_workers = 0
        while finished_workers < worker_count:
            batch = result_q.get()
            if batch is None:
                finished_workers += 1
                continue
            if (
                isinstance(batch, tuple)
                and len(batch) == 2
                and isinstance(batch[0], str)
                and batch[0] == "__validation_worker_error__"
            ):
                raise RuntimeError(f"Validation worker failed: {batch[1]}")
            yield batch
    finally:
        for proc in workers:
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.terminate()
        result_q.close()


def _compute_binary_metrics_from_outputs(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute binary classification metrics and threshold curves from flat arrays."""
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    if y_true.size == 0 or y_pred.size == 0:
        return {
            "accuracy": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "auc": 0.0,
            "loss": 0.0,
            "tp": np.zeros(101, dtype=np.float32),
            "fp": np.zeros(101, dtype=np.float32),
            "tn": np.zeros(101, dtype=np.float32),
            "fn": np.zeros(101, dtype=np.float32),
        }

    clipped_pred = np.clip(y_pred, 1e-7, 1.0 - 1e-7)
    positive_mask = y_true >= 0.5
    predicted_positive = clipped_pred >= 0.5

    tp_05 = float(np.sum(predicted_positive & positive_mask))
    fp_05 = float(np.sum(predicted_positive & ~positive_mask))
    tn_05 = float(np.sum((~predicted_positive) & (~positive_mask)))
    fn_05 = float(np.sum((~predicted_positive) & positive_mask))
    total = max(1.0, tp_05 + fp_05 + tn_05 + fn_05)

    auc_metric = tf.keras.metrics.AUC()
    auc_metric.update_state(
        tf.convert_to_tensor(y_true.reshape(-1, 1), dtype=tf.float32),
        tf.convert_to_tensor(clipped_pred.reshape(-1, 1), dtype=tf.float32),
    )

    thresholds = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    thresholded_positive = clipped_pred[:, None] >= thresholds[None, :]
    positive_by_threshold = positive_mask[:, None]
    tp = np.sum(thresholded_positive & positive_by_threshold, axis=0).astype(np.float32)
    fp = np.sum(thresholded_positive & ~positive_by_threshold, axis=0).astype(np.float32)
    tn = np.sum((~thresholded_positive) & ~positive_by_threshold, axis=0).astype(np.float32)
    fn = np.sum((~thresholded_positive) & positive_by_threshold, axis=0).astype(np.float32)

    return {
        "accuracy": (tp_05 + tn_05) / total,
        "recall": tp_05 / max(1.0, tp_05 + fn_05),
        "precision": tp_05 / max(1.0, tp_05 + fp_05),
        "auc": float(auc_metric.result().numpy()),
        "loss": float(
            np.mean(
                -(
                    y_true * np.log(clipped_pred)
                    + (1.0 - y_true) * np.log(1.0 - clipped_pred)
                )
            )
        ),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def run_nonstreaming_numpy_eval(
    config,
    data_processor,
    model,
    data_set: str,
    truncation_strategy: str,
    max_samples: int = 0,
):
    """Run nonstreaming eval in eager NumPy batches, bypassing tf.data generator ops."""
    all_truth = []
    all_pred = []
    for batch_x, batch_y in iter_nonstreaming_eval_batches(
        config,
        data_processor,
        data_set,
        truncation_strategy=truncation_strategy,
        max_samples=max_samples,
    ):
        pred_batch = model(tf.convert_to_tensor(batch_x, dtype=tf.float32), training=False)
        all_truth.append(np.asarray(batch_y, dtype=np.float32).reshape(-1))
        all_pred.append(np.asarray(pred_batch, dtype=np.float32).reshape(-1))

    if not all_truth:
        return {"y_true": np.zeros(0, dtype=np.float32), "y_pred": np.zeros(0, dtype=np.float32), **_compute_binary_metrics_from_outputs(np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32))}

    y_true = np.concatenate(all_truth, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        **_compute_binary_metrics_from_outputs(y_true, y_pred),
    }


def build_training_dataset(config, data_processor, policy_ref):
    """Create a repeating tf.data pipeline for training samples."""
    feature_shape = tuple(config["training_input_shape"])
    features_length = int(config["spectrogram_length"])
    providers = [
        provider
        for provider in data_processor.feature_providers
        if provider.get_mode_size("training")
    ]
    if not providers:
        raise ValueError("No training feature providers available.")

    provider_weights = np.asarray(
        [max(0.0, float(provider.sampling_weight)) for provider in providers],
        dtype=np.float64,
    )
    if float(np.sum(provider_weights)) <= 0.0:
        provider_weights = np.ones(len(providers), dtype=np.float64)
    provider_probs = provider_weights / np.sum(provider_weights)

    batch_size = int(config["batch_size"])
    default_workers = max(1, (os.cpu_count() or 1) - 1)
    data_workers = env_int(
        "MICRO_TRAIN_DATA_WORKERS",
        default_workers,
        minimum=0,
    )
    queue_factor = env_int("MICRO_TRAIN_DATA_QUEUE_FACTOR", 3, minimum=1)
    start_method = os.environ.get("MICRO_TRAIN_DATA_START_METHOD", "fork").strip().lower()
    if start_method not in {"fork", "spawn", "forkserver"}:
        logging.warning(
            "Invalid MICRO_TRAIN_DATA_START_METHOD=%r, falling back to 'fork'",
            start_method,
        )
        start_method = "fork"

    def build_numpy_batch(policy):
        x = np.empty((batch_size, *feature_shape), dtype=np.float32)
        y = np.empty((batch_size, 1), dtype=np.float32)
        w = np.empty((batch_size, 1), dtype=np.float32)
        s = np.empty((batch_size, 1), dtype=np.float32)
        for i in range(batch_size):
            provider_idx = int(np.random.choice(len(providers), p=provider_probs))
            provider = providers[provider_idx]
            soft_label = np.nan
            if hasattr(provider, "get_random_example"):
                (
                    spectrogram,
                    sampled_label,
                    sampled_weight,
                    soft_label,
                ) = provider.get_random_example(
                    "training", features_length, "default"
                )
            else:
                spectrogram = provider.get_random_spectrogram(
                    "training", features_length, "default"
                )
                sampled_label = float(provider.label)
                sampled_weight = float(provider.penalty_weight)

            spectrogram = data_lib.spec_augment(
                spectrogram,
                int(policy["time_mask_max_size"]),
                int(policy["time_mask_count"]),
                int(policy["freq_mask_max_size"]),
                int(policy["freq_mask_count"]),
            )
            x[i] = np.asarray(spectrogram, dtype=np.float32)
            y[i, 0] = np.float32(sampled_label)
            w[i, 0] = np.float32(sampled_weight)
            s[i, 0] = np.float32(soft_label) if np.isfinite(soft_label) else np.float32(np.nan)
        return x, y, w, s

    def sample_generator_single():
        while True:
            yield build_numpy_batch(policy_ref["value"])

    def _worker_loop(request_q, result_q):
        while True:
            policy = request_q.get()
            if policy is None:
                break
            result_q.put(build_numpy_batch(policy))

    def sample_generator_mp():
        worker_count = max(2, data_workers)
        queue_depth = max(worker_count * queue_factor, worker_count + 1)
        ctx = mp.get_context(start_method)
        request_q = ctx.Queue(maxsize=queue_depth)
        result_q = ctx.Queue(maxsize=queue_depth)
        workers = []
        logging.info(
            "Training tf.data multiprocessing enabled: workers=%d start_method=%s queue_depth=%d",
            worker_count,
            start_method,
            queue_depth,
        )
        for _ in range(worker_count):
            proc = ctx.Process(target=_worker_loop, args=(request_q, result_q), daemon=True)
            proc.start()
            workers.append(proc)

        for _ in workers:
            request_q.put(policy_ref["value"])

        try:
            while True:
                batch = result_q.get()
                request_q.put(policy_ref["value"])
                yield batch
        finally:
            for _ in workers:
                try:
                    request_q.put_nowait(None)
                except Exception:
                    pass
            for proc in workers:
                proc.join(timeout=1.0)
                if proc.is_alive():
                    proc.terminate()
            request_q.close()
            result_q.close()

    dataset = tf.data.Dataset.from_generator(
        sample_generator_mp if data_workers > 1 else sample_generator_single,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, *feature_shape), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, 1), dtype=tf.float32),
        ),
    )
    if data_workers <= 1:
        logging.info("Training tf.data multiprocessing disabled: workers=%d", data_workers)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = apply_optional_device_prefetch(dataset)
    return dataset


def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value: float) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(value)]))


def build_tfrecord_training_dataset(config, data_processor):
    """Create (or reuse) TFRecord training cache and build tf.data pipeline from it."""
    feature_shape = tuple(config["training_input_shape"])
    features_length = int(config["spectrogram_length"])
    batch_size = int(config["batch_size"])
    default_examples = max(8192, batch_size * 64)
    record_count = env_int("MICRO_TRAIN_TFRECORD_EXAMPLES", default_examples, minimum=batch_size)
    shuffle_buffer = env_int(
        "MICRO_TRAIN_TFRECORD_SHUFFLE_BUFFER",
        min(16384, max(2048, record_count // 2)),
        minimum=batch_size,
    )
    rebuild_cache = os.environ.get("MICRO_TRAIN_TFRECORD_REBUILD", "0").strip() == "1"
    cache_mode = os.environ.get("MICRO_TRAIN_TFRECORD_CACHE_MODE", "none").strip().lower()
    if cache_mode not in {"none", "auto", "ram", "file"}:
        logging.warning(
            "Invalid MICRO_TRAIN_TFRECORD_CACHE_MODE=%r, falling back to 'none'",
            cache_mode,
        )
        cache_mode = "none"
    cache_ram_mb = env_int("MICRO_TRAIN_TFRECORD_CACHE_RAM_MB", 4096, minimum=256)
    cache_file_path = os.environ.get("MICRO_TRAIN_TFRECORD_CACHE_FILE", "").strip()
    cache_dir = os.environ.get(
        "MICRO_TRAIN_TFRECORD_CACHE_DIR",
        os.path.join(config["train_dir"], "input_tfrecord"),
    ).strip()
    if not cache_dir:
        cache_dir = os.path.join(config["train_dir"], "input_tfrecord")
    os.makedirs(cache_dir, exist_ok=True)
    tfrecord_path = os.path.join(cache_dir, f"train_{record_count}.tfrecord")
    metadata_path = os.path.join(cache_dir, f"train_{record_count}.meta.json")
    build_batch_size = env_int(
        "MICRO_TRAIN_TFRECORD_BUILD_BATCH",
        min(record_count, max(batch_size, 512)),
        minimum=1,
    )
    build_batch_size = min(record_count, max(1, build_batch_size))

    providers = [
        provider
        for provider in data_processor.feature_providers
        if provider.get_mode_size("training")
    ]
    if not providers:
        raise ValueError("No training feature providers available.")
    provider_weights = np.asarray(
        [max(0.0, float(provider.sampling_weight)) for provider in providers],
        dtype=np.float64,
    )
    if float(np.sum(provider_weights)) <= 0.0:
        provider_weights = np.ones(len(providers), dtype=np.float64)
    provider_probs = provider_weights / np.sum(provider_weights)

    if rebuild_cache or (not os.path.isfile(tfrecord_path)):
        logging.info(
            "Building TFRecord training cache: %s (examples=%d, build_batch=%d)",
            tfrecord_path,
            record_count,
            build_batch_size,
        )
        writer = tf.io.TFRecordWriter(tfrecord_path)
        try:
            written = 0
            while written < record_count:
                current_batch = min(build_batch_size, record_count - written)
                batch_x, batch_y, batch_w, batch_s = data_processor.get_data(
                    "training",
                    current_batch,
                    features_length,
                    "default",
                )
                batch_x = np.asarray(batch_x, dtype=np.float32)
                batch_y = np.asarray(batch_y, dtype=np.float32).reshape(-1)
                batch_w = np.asarray(batch_w, dtype=np.float32).reshape(-1)
                batch_s = np.asarray(batch_s, dtype=np.float32).reshape(-1)

                for batch_idx in range(current_batch):
                    spectrogram = batch_x[batch_idx]
                    if spectrogram.shape != feature_shape:
                        spectrogram = data_lib.fixed_length_spectrogram(
                            spectrogram, features_length, "truncate_start", 0
                        )
                        spectrogram = np.asarray(spectrogram, dtype=np.float32)
                    spectrogram_fp16 = spectrogram.astype(np.float16, copy=False)
                    soft_label = float(batch_s[batch_idx])
                    soft = soft_label if np.isfinite(soft_label) else float("nan")
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                "x": _bytes_feature(spectrogram_fp16.tobytes()),
                                "y": _float_feature(float(batch_y[batch_idx])),
                                "w": _float_feature(float(batch_w[batch_idx])),
                                "s": _float_feature(soft),
                            }
                        )
                    )
                    writer.write(example.SerializeToString())

                written += current_batch
                if written % 5000 == 0 or written == record_count:
                    logging.info(
                        "TFRecord cache progress: %d/%d examples",
                        written,
                        record_count,
                    )
        finally:
            writer.close()

        with open(metadata_path, "w", encoding="utf-8") as meta_out:
            json.dump(
                {
                    "record_count": record_count,
                    "feature_shape": list(feature_shape),
                    "features_length": features_length,
                    "provider_count": len(providers),
                    "build_batch_size": build_batch_size,
                },
                meta_out,
                indent=2,
            )
    else:
        logging.info("Reusing TFRecord training cache: %s", tfrecord_path)

    flat_size = int(np.prod(feature_shape))
    feature_spec = {
        "x": tf.io.FixedLenFeature([], tf.string),
        "y": tf.io.FixedLenFeature([], tf.float32),
        "w": tf.io.FixedLenFeature([], tf.float32),
        "s": tf.io.FixedLenFeature([], tf.float32),
    }

    def _parse_record(serialized):
        parsed = tf.io.parse_single_example(serialized, feature_spec)
        x = tf.io.decode_raw(parsed["x"], tf.float16)
        x = tf.reshape(x, (flat_size,))
        x = tf.cast(x, tf.float32)
        x = tf.reshape(x, feature_shape)
        y = tf.reshape(parsed["y"], (1,))
        w = tf.reshape(parsed["w"], (1,))
        s = tf.reshape(parsed["s"], (1,))
        return x, y, w, s

    dataset = tf.data.TFRecordDataset(
        tfrecord_path, num_parallel_reads=tf.data.AUTOTUNE
    )
    dataset = dataset.map(_parse_record, num_parallel_calls=tf.data.AUTOTUNE)

    # Optional dataset cache layer before shuffle/repeat to reduce disk IO.
    # Default "auto": cache in RAM when estimated record size fits configured budget.
    chosen_cache_mode = "none"
    if cache_mode == "ram":
        dataset = dataset.cache()
        chosen_cache_mode = "ram"
    elif cache_mode == "file":
        if not cache_file_path:
            cache_file_path = os.path.join(cache_dir, f"train_{record_count}.dataset_cache")
        dataset = dataset.cache(cache_file_path)
        chosen_cache_mode = f"file:{cache_file_path}"
    elif cache_mode == "auto":
        estimated_bytes = int(record_count * (flat_size * 2 + 16))
        if estimated_bytes <= (int(cache_ram_mb) * 1024 * 1024):
            dataset = dataset.cache()
            chosen_cache_mode = "ram(auto)"

    dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = apply_optional_device_prefetch(dataset)
    logging.info(
        "Training TFRecord pipeline enabled: records=%d, batch=%d, shuffle_buffer=%d, cache=%s",
        record_count,
        batch_size,
        shuffle_buffer,
        chosen_cache_mode,
    )
    if cache_mode == "auto":
        logging.info(
            "TFRecord cache auto decision: estimated=%.1fMB threshold=%dMB",
            estimated_bytes / (1024.0 * 1024.0),
            cache_ram_mb,
        )
    return dataset


def tf_spec_augment_batch(
    batch: tf.Tensor,
    time_mask_max_size: int,
    time_mask_count: int,
    freq_mask_max_size: int,
    freq_mask_count: int,
) -> tf.Tensor:
    """Apply SpecAugment with TensorFlow ops on a [B, T, F] batch."""
    if (
        time_mask_max_size <= 0
        or time_mask_count <= 0
    ) and (
        freq_mask_max_size <= 0
        or freq_mask_count <= 0
    ):
        return batch

    def _apply_single(spec):
        out = spec
        time_frames = tf.shape(out)[0]
        freq_bins = tf.shape(out)[1]

        for _ in range(max(0, int(time_mask_count))):
            max_t = tf.minimum(tf.cast(time_mask_max_size, tf.int32), time_frames)
            t = tf.random.uniform([], minval=0, maxval=max_t + 1, dtype=tf.int32)
            max_start = tf.maximum(time_frames - t + 1, 1)
            t0 = tf.random.uniform([], minval=0, maxval=max_start, dtype=tf.int32)
            mask = tf.concat(
                [
                    tf.ones((t0,), dtype=out.dtype),
                    tf.zeros((t,), dtype=out.dtype),
                    tf.ones((time_frames - t0 - t,), dtype=out.dtype),
                ],
                axis=0,
            )
            out = out * tf.expand_dims(mask, axis=1)

        for _ in range(max(0, int(freq_mask_count))):
            max_f = tf.minimum(tf.cast(freq_mask_max_size, tf.int32), freq_bins)
            f = tf.random.uniform([], minval=0, maxval=max_f + 1, dtype=tf.int32)
            max_start = tf.maximum(freq_bins - f + 1, 1)
            f0 = tf.random.uniform([], minval=0, maxval=max_start, dtype=tf.int32)
            mask = tf.concat(
                [
                    tf.ones((f0,), dtype=out.dtype),
                    tf.zeros((f,), dtype=out.dtype),
                    tf.ones((freq_bins - f0 - f,), dtype=out.dtype),
                ],
                axis=0,
            )
            out = out * tf.expand_dims(mask, axis=0)

        return out

    augmented = tf.map_fn(_apply_single, batch, fn_output_signature=tf.float32)
    # Keep static shape information for downstream Keras layers (Conv2D).
    augmented.set_shape(batch.shape)
    return augmented


def validate_nonstreaming(config, data_processor, model, test_set):
    test_eval = run_nonstreaming_numpy_eval(
        config,
        data_processor,
        model,
        test_set,
        truncation_strategy="truncate_start",
        max_samples=0,
    )

    metrics = {}
    metrics["accuracy"] = test_eval["accuracy"]
    metrics["recall"] = test_eval["recall"]
    metrics["precision"] = test_eval["precision"]

    metrics["auc"] = test_eval["auc"]
    metrics["loss"] = test_eval["loss"]
    metrics["recall_at_no_faph"] = 0
    metrics["cutoff_for_no_faph"] = 0
    metrics["ambient_false_positives"] = 0
    metrics["ambient_false_positives_per_hour"] = 0
    metrics["average_viable_recall"] = 0

    ambient_mode = test_set + "_ambient"
    if data_processor.get_mode_size(ambient_mode) > 0:
        ambient_eval = run_nonstreaming_numpy_eval(
            config,
            data_processor,
            model,
            ambient_mode,
            truncation_strategy="split",
            max_samples=0,
        )

        duration_of_ambient_set = data_processor.get_mode_duration(ambient_mode) / 3600.0
        duration_of_ambient_set = max(duration_of_ambient_set, 1e-8)

        combined_eval = _compute_binary_metrics_from_outputs(
            np.concatenate([test_eval["y_true"], ambient_eval["y_true"]], axis=0),
            np.concatenate([test_eval["y_pred"], ambient_eval["y_pred"]], axis=0),
        )
        all_true_positives = combined_eval["tp"]
        ambient_false_positives = ambient_eval["fp"]
        all_false_negatives = combined_eval["fn"]

        metrics["auc"] = combined_eval["auc"]
        metrics["loss"] = combined_eval["loss"]
        metrics["accuracy"] = combined_eval["accuracy"]
        metrics["recall"] = combined_eval["recall"]
        metrics["precision"] = combined_eval["precision"]

        recall_at_cutoffs = (
            all_true_positives / (all_true_positives + all_false_negatives)
        )
        faph_at_cutoffs = ambient_false_positives / duration_of_ambient_set

        target_faph_cutoff_probability = 1.0
        for index, cutoff in enumerate(np.linspace(0.0, 1.0, 101)):
            if faph_at_cutoffs[index] == 0:
                target_faph_cutoff_probability = cutoff
                recall_at_no_faph = recall_at_cutoffs[index]
                break

        if faph_at_cutoffs[0] > 2:
            # Use linear interpolation to estimate recall at 2 faph

            # Increase index until we find a faph less than 2
            index_of_first_viable = 1
            while faph_at_cutoffs[index_of_first_viable] > 2:
                index_of_first_viable += 1

            x0 = faph_at_cutoffs[index_of_first_viable - 1]
            y0 = recall_at_cutoffs[index_of_first_viable - 1]
            x1 = faph_at_cutoffs[index_of_first_viable]
            y1 = recall_at_cutoffs[index_of_first_viable]

            recall_at_2faph = (y0 * (x1 - 2.0) + y1 * (2.0 - x0)) / (x1 - x0)
        else:
            # Lowest faph is already under 2, assume the recall is constant before this
            index_of_first_viable = 0
            recall_at_2faph = recall_at_cutoffs[0]

        x_coordinates = [2.0]
        y_coordinates = [recall_at_2faph]

        for index in range(index_of_first_viable, len(recall_at_cutoffs)):
            if faph_at_cutoffs[index] != x_coordinates[-1]:
                # Only add a point if it is a new faph
                # This ensures if a faph rate is repeated, we use the highest recall
                x_coordinates.append(faph_at_cutoffs[index])
                y_coordinates.append(recall_at_cutoffs[index])

        # Use trapezoid rule to estimate the area under the curve, then divide by 2.0 to get the average recall
        average_viable_recall = (
            np.trapz(np.flip(y_coordinates), np.flip(x_coordinates)) / 2.0
        )

        metrics["recall_at_no_faph"] = recall_at_no_faph
        metrics["cutoff_for_no_faph"] = target_faph_cutoff_probability
        metrics["ambient_false_positives"] = ambient_false_positives[50]
        metrics["ambient_false_positives_per_hour"] = faph_at_cutoffs[50]
        metrics["average_viable_recall"] = average_viable_recall

    return metrics


def validate_nonstreaming_with_policy(config, data_processor, model, test_set, mode):
    max_samples = 0
    include_ambient = True
    if mode == "fast":
        max_samples = env_int("MICRO_FAST_VALIDATION_MAX_SAMPLES", 512, minimum=1)
        include_ambient = False

    metrics = {
        "validation_mode": mode,
        "is_full_validation": mode == "full",
    }

    test_eval = run_nonstreaming_numpy_eval(
        config,
        data_processor,
        model,
        test_set,
        truncation_strategy="truncate_start",
        max_samples=max_samples,
    )

    metrics["accuracy"] = test_eval["accuracy"]
    metrics["recall"] = test_eval["recall"]
    metrics["precision"] = test_eval["precision"]
    metrics["auc"] = test_eval["auc"]
    metrics["loss"] = test_eval["loss"]
    metrics["recall_at_no_faph"] = 0
    metrics["cutoff_for_no_faph"] = 0
    metrics["ambient_false_positives"] = 0
    metrics["ambient_false_positives_per_hour"] = 0
    metrics["average_viable_recall"] = 0
    metrics["ambient_metrics_available"] = False
    metrics["ambient_reporting_cutoff"] = 0.5

    ambient_mode = test_set + "_ambient"
    if not include_ambient or data_processor.get_mode_size(ambient_mode) <= 0:
        return metrics

    ambient_eval = run_nonstreaming_numpy_eval(
        config,
        data_processor,
        model,
        ambient_mode,
        truncation_strategy="split",
        max_samples=0,
    )

    duration_of_ambient_set = data_processor.get_mode_duration(ambient_mode) / 3600.0
    duration_of_ambient_set = max(duration_of_ambient_set, 1e-8)
    combined_eval = _compute_binary_metrics_from_outputs(
        np.concatenate([test_eval["y_true"], ambient_eval["y_true"]], axis=0),
        np.concatenate([test_eval["y_pred"], ambient_eval["y_pred"]], axis=0),
    )
    all_true_positives = combined_eval["tp"]
    ambient_false_positives = ambient_eval["fp"]
    all_false_negatives = combined_eval["fn"]

    metrics["auc"] = combined_eval["auc"]
    metrics["loss"] = combined_eval["loss"]
    metrics["accuracy"] = combined_eval["accuracy"]
    metrics["recall"] = combined_eval["recall"]
    metrics["precision"] = combined_eval["precision"]

    recall_at_cutoffs = all_true_positives / (all_true_positives + all_false_negatives)
    faph_at_cutoffs = ambient_false_positives / duration_of_ambient_set

    target_faph_cutoff_probability = 1.0
    recall_at_no_faph = 0.0
    for index, cutoff in enumerate(np.linspace(0.0, 1.0, 101)):
        if faph_at_cutoffs[index] == 0:
            target_faph_cutoff_probability = cutoff
            recall_at_no_faph = recall_at_cutoffs[index]
            break

    if faph_at_cutoffs[0] > 2:
        index_of_first_viable = 1
        while faph_at_cutoffs[index_of_first_viable] > 2:
            index_of_first_viable += 1

        x0 = faph_at_cutoffs[index_of_first_viable - 1]
        y0 = recall_at_cutoffs[index_of_first_viable - 1]
        x1 = faph_at_cutoffs[index_of_first_viable]
        y1 = recall_at_cutoffs[index_of_first_viable]
        recall_at_2faph = (y0 * (x1 - 2.0) + y1 * (2.0 - x0)) / (x1 - x0)
    else:
        index_of_first_viable = 0
        recall_at_2faph = recall_at_cutoffs[0]

    x_coordinates = [2.0]
    y_coordinates = [recall_at_2faph]
    for index in range(index_of_first_viable, len(recall_at_cutoffs)):
        if faph_at_cutoffs[index] != x_coordinates[-1]:
            x_coordinates.append(faph_at_cutoffs[index])
            y_coordinates.append(recall_at_cutoffs[index])

    metrics["recall_at_no_faph"] = recall_at_no_faph
    metrics["cutoff_for_no_faph"] = target_faph_cutoff_probability
    metrics["ambient_false_positives"] = ambient_false_positives[50]
    metrics["ambient_false_positives_per_hour"] = faph_at_cutoffs[50]
    metrics["average_viable_recall"] = (
        np.trapz(np.flip(y_coordinates), np.flip(x_coordinates)) / 2.0
    )
    metrics["ambient_metrics_available"] = True
    return metrics


def format_nonstreaming_validation_log(training_step, nonstreaming_metrics):
    validation_mode = nonstreaming_metrics["validation_mode"]
    prefix = (
        f"Step {training_step} (nonstreaming/{validation_mode}): Validation: "
        f"accuracy = {nonstreaming_metrics['accuracy'] * 100:.2f}%, "
        f"recall = {nonstreaming_metrics['recall'] * 100:.2f}%, "
        f"precision = {nonstreaming_metrics['precision'] * 100:.2f}%, "
        f"loss = {nonstreaming_metrics['loss']:.5f}, "
        f"auc = {nonstreaming_metrics['auc']:.5f}"
    )
    if not nonstreaming_metrics.get("ambient_metrics_available", False):
        return (
            prefix
            + ", recall at no faph = n/a, cutoff = n/a, "
            + "ambient false positives = n/a, estimated false positives per hour = n/a, "
            + "average viable recall = n/a"
        )

    ambient_cutoff = nonstreaming_metrics.get("ambient_reporting_cutoff", 0.5)
    return (
        prefix
        + f", recall at no faph = {nonstreaming_metrics['recall_at_no_faph'] * 100:.3f}"
        + f" with cutoff {nonstreaming_metrics['cutoff_for_no_faph']:.2f}, "
        + f"ambient false positives @ cutoff {ambient_cutoff:.2f} = "
        + f"{int(nonstreaming_metrics['ambient_false_positives'])}, "
        + "estimated false positives per hour "
        + f"@ cutoff {ambient_cutoff:.2f} = "
        + f"{nonstreaming_metrics['ambient_false_positives_per_hour']:.5f}, "
        + f"average viable recall = {nonstreaming_metrics['average_viable_recall']:.9f}"
    )


def train(model, config, data_processor):
    skip_nonstreaming_validation = (
        os.environ.get("MICRO_BENCH_SKIP_VALIDATION", "0") == "1"
    )
    full_validation_every = env_int("MICRO_FULL_VALIDATION_EVERY", 4, minimum=1)
    # Assign default training settings if not set in the configuration yaml
    if not (training_steps_list := config.get("training_steps")):
        training_steps_list = [20000]
    if not (learning_rates_list := config.get("learning_rates")):
        learning_rates_list = [0.001]
    if not (mix_up_prob_list := config.get("mix_up_augmentation_prob")):
        mix_up_prob_list = [0.0]
    if not (freq_mix_prob_list := config.get("freq_mix_augmentation_prob")):
        freq_mix_prob_list = [0.0]
    if not (time_mask_max_size_list := config.get("time_mask_max_size")):
        time_mask_max_size_list = [5]
    if not (time_mask_count_list := config.get("time_mask_count")):
        time_mask_count_list = [2]
    if not (freq_mask_max_size_list := config.get("freq_mask_max_size")):
        freq_mask_max_size_list = [5]
    if not (freq_mask_count_list := config.get("freq_mask_count")):
        freq_mask_count_list = [2]
    if not (positive_class_weight_list := config.get("positive_class_weight")):
        positive_class_weight_list = [1.0]
    if not (negative_class_weight_list := config.get("negative_class_weight")):
        negative_class_weight_list = [1.0]
    distillation_config = config.get("distillation", {}) or {}
    distillation_enabled = bool(distillation_config.get("enabled", False))
    distillation_alpha = float(distillation_config.get("alpha", 0.7))
    distillation_beta = float(distillation_config.get("beta", 0.3))
    distillation_temperature = max(
        1e-3, float(distillation_config.get("temperature", 2.0))
    )

    # Ensure all training setting lists are as long as the training step iterations
    def pad_list_with_last_entry(list_to_pad, desired_length):
        while len(list_to_pad) < desired_length:
            last_entry = list_to_pad[-1]
            list_to_pad.append(last_entry)

    training_step_iterations = len(training_steps_list)
    pad_list_with_last_entry(learning_rates_list, training_step_iterations)
    pad_list_with_last_entry(mix_up_prob_list, training_step_iterations)
    pad_list_with_last_entry(freq_mix_prob_list, training_step_iterations)
    pad_list_with_last_entry(time_mask_max_size_list, training_step_iterations)
    pad_list_with_last_entry(time_mask_count_list, training_step_iterations)
    pad_list_with_last_entry(freq_mask_max_size_list, training_step_iterations)
    pad_list_with_last_entry(freq_mask_count_list, training_step_iterations)
    pad_list_with_last_entry(positive_class_weight_list, training_step_iterations)
    pad_list_with_last_entry(negative_class_weight_list, training_step_iterations)

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam()

    cutoffs = np.linspace(0.0, 1.0, 101).tolist()

    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.TruePositives(name="tp", thresholds=cutoffs),
        tf.keras.metrics.FalsePositives(name="fp", thresholds=cutoffs),
        tf.keras.metrics.TrueNegatives(name="tn", thresholds=cutoffs),
        tf.keras.metrics.FalseNegatives(name="fn", thresholds=cutoffs),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.BinaryCrossentropy(name="loss"),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    graph_mode = env_bool("MICRO_TRAIN_GRAPH_MODE", True)
    jit_compile = graph_mode and env_bool("MICRO_TRAIN_JIT_COMPILE", False)
    progress_interval = env_int("MICRO_TRAIN_PROGRESS_INTERVAL", 25, minimum=1)
    logging.info(
        "Training execution mode: %s (jit_compile=%s, progress_interval=%d)",
        "graph" if graph_mode else "eager",
        jit_compile,
        progress_interval,
    )

    def _train_step_impl(
        x_batch,
        y_batch,
        base_weights_batch,
        soft_batch,
        positive_class_weight,
        negative_class_weight,
        time_mask_max_size,
        time_mask_count,
        freq_mask_max_size,
        freq_mask_count,
    ):
        x_batch = tf_spec_augment_batch(
            x_batch,
            time_mask_max_size,
            time_mask_count,
            freq_mask_max_size,
            freq_mask_count,
        )

        class_weight_vector = tf.where(
            y_batch >= 0.5,
            tf.cast(positive_class_weight, tf.float32),
            tf.cast(negative_class_weight, tf.float32),
        )
        w_batch = base_weights_batch * class_weight_vector

        finite_soft_mask = tf.cast(tf.math.is_finite(soft_batch), tf.float32)
        has_distillation_targets = tf.reduce_any(finite_soft_mask > 0.0)

        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)

            hard_loss_per_sample = tf.keras.backend.binary_crossentropy(y_batch, y_pred)
            hard_loss_per_sample = tf.reshape(hard_loss_per_sample, (-1, 1))
            hard_weight_sum = tf.reduce_sum(w_batch) + 1e-8
            hard_loss = tf.reduce_sum(hard_loss_per_sample * w_batch) / hard_weight_sum

            clipped_pred = tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6)
            safe_teacher = tf.where(
                tf.math.is_finite(soft_batch),
                soft_batch,
                tf.fill(tf.shape(soft_batch), 0.5),
            )
            clipped_teacher = tf.clip_by_value(safe_teacher, 1e-6, 1.0 - 1e-6)
            student_logits = tf.math.log(clipped_pred / (1.0 - clipped_pred))
            teacher_logits = tf.math.log(clipped_teacher / (1.0 - clipped_teacher))
            student_temp = tf.sigmoid(student_logits / distillation_temperature)
            teacher_temp = tf.sigmoid(teacher_logits / distillation_temperature)

            distillation_loss_per_sample = tf.square(student_temp - teacher_temp)
            distillation_weights = w_batch * finite_soft_mask
            distillation_weight_sum = tf.reduce_sum(distillation_weights)
            distillation_loss = tf.where(
                distillation_weight_sum > 0.0,
                tf.reduce_sum(distillation_loss_per_sample * distillation_weights)
                / (distillation_weight_sum + 1e-8),
                tf.zeros((), dtype=tf.float32),
            )

            effective_alpha = tf.constant(0.0, dtype=tf.float32)
            effective_beta = tf.constant(1.0, dtype=tf.float32)
            if distillation_enabled:
                effective_alpha = tf.where(
                    has_distillation_targets,
                    tf.cast(distillation_alpha, tf.float32),
                    tf.constant(0.0, dtype=tf.float32),
                )
                effective_beta = tf.where(
                    has_distillation_targets,
                    tf.cast(distillation_beta, tf.float32),
                    tf.constant(1.0, dtype=tf.float32),
                )

            total_loss = effective_beta * hard_loss + effective_alpha * distillation_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        y_true_bin = y_batch >= 0.5
        y_pred_bin = y_pred >= 0.5
        tp = tf.reduce_sum(
            tf.cast(tf.logical_and(y_true_bin, y_pred_bin), tf.float32)
        )
        tn = tf.reduce_sum(
            tf.cast(tf.logical_and(tf.logical_not(y_true_bin), tf.logical_not(y_pred_bin)), tf.float32)
        )
        fp = tf.reduce_sum(
            tf.cast(tf.logical_and(tf.logical_not(y_true_bin), y_pred_bin), tf.float32)
        )
        fn = tf.reduce_sum(
            tf.cast(tf.logical_and(y_true_bin, tf.logical_not(y_pred_bin)), tf.float32)
        )
        total = tf.maximum(1.0, tp + tn + fp + fn)

        return {
            "accuracy": (tp + tn) / total,
            "recall": tf.math.divide_no_nan(tp, tp + fn),
            "precision": tf.math.divide_no_nan(tp, tp + fp),
            "hard_loss": hard_loss,
            "distill_loss": distillation_loss,
            "total_loss": total_loss,
            "distillation_active": has_distillation_targets,
            "auc": tf.constant(float("nan"), dtype=tf.float32),
        }

    if graph_mode:
        train_step_runner = tf.function(
            _train_step_impl,
            reduce_retracing=True,
            jit_compile=jit_compile,
        )
    else:
        train_step_runner = _train_step_impl

    # Configure checkpointer and restore if available
    checkpoint_directory = os.path.join(config["train_dir"], "restore/")
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

    # Configure TensorBoard summaries
    train_writer = tf.summary.create_file_writer(
        os.path.join(config["summaries_dir"], "train")
    )
    validation_writer = tf.summary.create_file_writer(
        os.path.join(config["summaries_dir"], "validation")
    )

    training_steps_max = np.sum(training_steps_list)

    best_minimization_quantity = 10000
    best_maximization_quantity = 0.0
    best_no_faph_cutoff = 1.0
    full_validation_lr_patience = env_int("MICRO_TRAIN_FULL_LR_PATIENCE", 2, minimum=1)
    full_validation_early_stop_patience = env_int(
        "MICRO_TRAIN_FULL_EARLY_STOP_PATIENCE", 6, minimum=1
    )
    full_validation_max_lr_reductions = env_int(
        "MICRO_TRAIN_FULL_MAX_LR_REDUCTIONS", 3, minimum=1
    )
    full_validation_lr_factor = env_float("MICRO_TRAIN_FULL_LR_FACTOR", 0.5, minimum=0.0)
    full_validation_min_lr = env_float("MICRO_TRAIN_FULL_MIN_LR", 1e-5, minimum=0.0)
    learning_rate_decay_multiplier = 1.0
    full_validation_no_improve_count = 0
    full_validation_no_improve_since_lr_drop = 0
    full_validation_lr_reductions = 0
    stop_requested = False
    current_policy_ref = {
        "value": {
            "mix_up_prob": 0.0,
            "freq_mix_prob": 0.0,
            "time_mask_max_size": 0,
            "time_mask_count": 0,
            "freq_mask_max_size": 0,
            "freq_mask_count": 0,
        }
    }
    input_mode = os.environ.get("MICRO_TRAIN_INPUT_MODE", "python").strip().lower()
    if input_mode == "tfrecord":
        training_dataset = build_tfrecord_training_dataset(config, data_processor)
    else:
        input_mode = "python"
        training_dataset = build_training_dataset(config, data_processor, current_policy_ref)
    logging.info("Training input mode: %s", input_mode)
    training_iterator = iter(training_dataset)

    for training_step in range(1, training_steps_max + 1):
        training_steps_sum = 0
        for i in range(len(training_steps_list)):
            training_steps_sum += training_steps_list[i]
            if training_step <= training_steps_sum:
                scheduled_learning_rate = learning_rates_list[i]
                learning_rate = scheduled_learning_rate * learning_rate_decay_multiplier
                mix_up_prob = mix_up_prob_list[i]
                freq_mix_prob = freq_mix_prob_list[i]
                time_mask_max_size = time_mask_max_size_list[i]
                time_mask_count = time_mask_count_list[i]
                freq_mask_max_size = freq_mask_max_size_list[i]
                freq_mask_count = freq_mask_count_list[i]
                positive_class_weight = positive_class_weight_list[i]
                negative_class_weight = negative_class_weight_list[i]
                break

        model.optimizer.learning_rate.assign(learning_rate)

        augmentation_policy = {
            "mix_up_prob": mix_up_prob,
            "freq_mix_prob": freq_mix_prob,
            "time_mask_max_size": time_mask_max_size,
            "time_mask_count": time_mask_count,
            "freq_mask_max_size": freq_mask_max_size,
            "freq_mask_count": freq_mask_count,
        }
        current_policy_ref["value"] = augmentation_policy

        (
            train_fingerprints,
            train_ground_truth,
            train_sample_weights,
            train_soft_labels,
        ) = next(training_iterator)

        x_batch = tf.convert_to_tensor(train_fingerprints, dtype=tf.float32)
        y_batch = tf.reshape(
            tf.convert_to_tensor(train_ground_truth, dtype=tf.float32), (-1, 1)
        )
        base_weights_batch = tf.reshape(
            tf.convert_to_tensor(train_sample_weights, dtype=tf.float32), (-1, 1)
        )
        soft_batch = tf.reshape(
            tf.convert_to_tensor(train_soft_labels, dtype=tf.float32), (-1, 1)
        )

        train_result = train_step_runner(
            x_batch,
            y_batch,
            base_weights_batch,
            soft_batch,
            float(positive_class_weight),
            float(negative_class_weight),
            int(time_mask_max_size),
            int(time_mask_count),
            int(freq_mask_max_size),
            int(freq_mask_count),
        )

        is_last_step = training_step == training_steps_max
        progress_due = (training_step % progress_interval) == 0 or is_last_step
        eval_due = (training_step % config["eval_step_interval"]) == 0 or is_last_step
        if progress_due or eval_due:
            train_accuracy = float(train_result["accuracy"])
            train_recall = float(train_result["recall"])
            train_precision = float(train_result["precision"])
            train_auc = float(train_result["auc"])
            train_hard_loss = float(train_result["hard_loss"])
            train_distill_loss = float(train_result["distill_loss"])
            train_total_loss = float(train_result["total_loss"])
            distillation_active = bool(train_result["distillation_active"])
            print(
                "Validation Batch #{:d}: Accuracy = {:.3f}; Recall = {:.3f}; Precision = {:.3f}; Loss = {:.4f}; Mini-Batch #{:d}".format(
                    (training_step // config["eval_step_interval"] + 1),
                    train_accuracy,
                    train_recall,
                    train_precision,
                    train_total_loss,
                    (training_step % config["eval_step_interval"]),
                ),
                end="\r",
            )

        if eval_due:
            progress_pct = (training_step / training_steps_max) * 100.0
            current_eval_batch = int(np.ceil(training_step / config["eval_step_interval"]))
            total_eval_batches = int(np.ceil(training_steps_max / config["eval_step_interval"]))
            logging.info(
                "Step #%d: rate %f, accuracy %.2f%%, recall %.2f%%, precision %.2f%%, cross entropy %f",
                *(
                    training_step,
                    learning_rate,
                    train_accuracy * 100,
                    train_recall * 100,
                    train_precision * 100,
                    train_total_loss,
                ),
            )
            if distillation_active:
                logging.info(
                    "Step #%d (distill): hard_loss=%f distill_loss=%f alpha=%f beta=%f temperature=%f",
                    training_step,
                    train_hard_loss,
                    train_distill_loss,
                    distillation_alpha,
                    distillation_beta,
                    distillation_temperature,
                )
            logging.info(
                "Progress: %.1f%% (%d/%d steps, eval batch %d/%d)",
                progress_pct,
                training_step,
                training_steps_max,
                current_eval_batch,
                total_eval_batches,
            )

            with train_writer.as_default():
                tf.summary.scalar("loss", train_total_loss, step=training_step)
                tf.summary.scalar("loss_hard", train_hard_loss, step=training_step)
                tf.summary.scalar(
                    "loss_distill", train_distill_loss, step=training_step
                )
                tf.summary.scalar("accuracy", train_accuracy, step=training_step)
                tf.summary.scalar("recall", train_recall, step=training_step)
                tf.summary.scalar("precision", train_precision, step=training_step)
                if np.isfinite(train_auc):
                    tf.summary.scalar("auc", train_auc, step=training_step)
                train_writer.flush()

            model.save_weights(
                os.path.join(config["train_dir"], "last_weights.weights.h5")
            )

            if skip_nonstreaming_validation:
                logging.info(
                    "Skipping nonstreaming validation (MICRO_BENCH_SKIP_VALIDATION=1)"
                )
                continue

            validation_mode = "full"
            if not is_last_step and (current_eval_batch % full_validation_every) != 0:
                validation_mode = "fast"

            nonstreaming_metrics = validate_nonstreaming_with_policy(
                config, data_processor, model, "validation", validation_mode
            )
            model.reset_metrics()  # reset metrics for next validation epoch of training
            logging.info(
                format_nonstreaming_validation_log(
                    training_step, nonstreaming_metrics
                )
            )

            with validation_writer.as_default():
                tf.summary.scalar(
                    "loss", nonstreaming_metrics["loss"], step=training_step
                )
                tf.summary.scalar(
                    "accuracy", nonstreaming_metrics["accuracy"], step=training_step
                )
                tf.summary.scalar(
                    "recall", nonstreaming_metrics["recall"], step=training_step
                )
                tf.summary.scalar(
                    "precision", nonstreaming_metrics["precision"], step=training_step
                )
                tf.summary.scalar(
                    "recall_at_no_faph",
                    nonstreaming_metrics["recall_at_no_faph"],
                    step=training_step,
                )
                tf.summary.scalar(
                    "auc",
                    nonstreaming_metrics["auc"],
                    step=training_step,
                )
                tf.summary.scalar(
                    "average_viable_recall",
                    nonstreaming_metrics["average_viable_recall"],
                    step=training_step,
                )
                tf.summary.scalar(
                    "is_full_validation",
                    float(nonstreaming_metrics["is_full_validation"]),
                    step=training_step,
                )
                validation_writer.flush()

            os.makedirs(os.path.join(config["train_dir"], "train"), exist_ok=True)

            model.save_weights(
                os.path.join(
                    config["train_dir"],
                    "train",
                    f"{int(best_minimization_quantity * 10000)}_weights_{training_step}.weights.h5",
                )
            )

            if nonstreaming_metrics["is_full_validation"]:
                (
                    full_validation_improved,
                    current_minimization_quantity,
                    current_maximization_quantity,
                    current_no_faph_cutoff,
                ) = full_validation_is_better(
                    nonstreaming_metrics,
                    float(config["target_minimization"]),
                    config["minimization_metric"],
                    config["maximization_metric"],
                    best_minimization_quantity,
                    best_maximization_quantity,
                )

                # Save model weights if this is a new best model
                if full_validation_improved:
                    best_minimization_quantity = current_minimization_quantity
                    best_maximization_quantity = current_maximization_quantity
                    best_no_faph_cutoff = current_no_faph_cutoff
                    full_validation_no_improve_count = 0
                    full_validation_no_improve_since_lr_drop = 0

                    # overwrite the best model weights
                    model.save_weights(
                        os.path.join(config["train_dir"], "best_weights.weights.h5")
                    )
                    checkpoint.save(file_prefix=checkpoint_prefix)
                else:
                    full_validation_no_improve_count += 1
                    full_validation_no_improve_since_lr_drop += 1
                    if (
                        full_validation_no_improve_since_lr_drop
                        >= full_validation_lr_patience
                        and full_validation_lr_reductions
                        < full_validation_max_lr_reductions
                    ):
                        next_learning_rate = (
                            scheduled_learning_rate * learning_rate_decay_multiplier * full_validation_lr_factor
                        )
                        if next_learning_rate >= full_validation_min_lr:
                            learning_rate_decay_multiplier *= full_validation_lr_factor
                            full_validation_lr_reductions += 1
                            full_validation_no_improve_since_lr_drop = 0
                            model.optimizer.learning_rate.assign(
                                scheduled_learning_rate * learning_rate_decay_multiplier
                            )
                            logging.info(
                                "Full validation plateau: reducing learning rate to %f (scheduled=%f, multiplier=%f, reductions=%d/%d)",
                                float(scheduled_learning_rate * learning_rate_decay_multiplier),
                                float(scheduled_learning_rate),
                                float(learning_rate_decay_multiplier),
                                full_validation_lr_reductions,
                                full_validation_max_lr_reductions,
                            )
                        else:
                            logging.info(
                                "Full validation plateau reached but next learning rate %f would fall below floor %f; keeping current rate",
                                float(next_learning_rate),
                                float(full_validation_min_lr),
                            )

                    if (
                        full_validation_no_improve_count
                        >= full_validation_early_stop_patience
                    ):
                        stop_requested = True
                        logging.info(
                            "Full validation early stop triggered after %d consecutive full validations without improvement",
                            full_validation_no_improve_count,
                        )

            logging.info(
                "So far the best minimization quantity is %.3f with best maximization quantity of %.5f%%; no faph cutoff is %.2f",
                best_minimization_quantity,
                (best_maximization_quantity * 100),
                best_no_faph_cutoff,
            )

            if stop_requested:
                break

    if stop_requested:
        logging.info(
            "Training stopped early after full-validation feedback exhausted the patience budget."
        )

    # Save checkpoint after training
    checkpoint.save(file_prefix=checkpoint_prefix)
    model.save_weights(os.path.join(config["train_dir"], "last_weights.weights.h5"))
