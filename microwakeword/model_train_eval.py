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

import argparse
import contextlib
import io
import os
import random
import sys
import yaml
import platform
from absl import logging

import numpy as np
import tensorflow as tf

# Disable GPU by default on ARM Macs, it's slower than just using the CPU
if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" or (
    sys.platform == "darwin"
    and platform.processor() == "arm"
    and "CUDA_VISIBLE_DEVICES" not in os.environ
):
    tf.config.set_visible_devices([], "GPU")

import microwakeword.data as input_data
import microwakeword.train as train
import microwakeword.test as test
import microwakeword.utils as utils

import microwakeword.inception as inception
import microwakeword.mixednet as mixednet

from microwakeword.layers import modes

_MINIMAL_LOG_ROUTING_INSTALLED = False
_SUPPRESSED_MINIMAL_INFO_PREFIXES = (
    "Sharding callback duration:",
)
_SUPPRESSED_MINIMAL_WARNINGS = (
    "ring_buffer_size_in_time_dim overwritten by the passed-in value",
    "Function `call` contains input name(s) resource with unsupported characters",
)


def _minimal_logs_enabled() -> bool:
    raw = os.environ.get("WAKEWORD_MINIMAL_LOGS", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _format_log_message(message, *args) -> str:
    text = str(message)
    if not args:
        return text
    try:
        return text % args
    except Exception:
        return " ".join([text, *[str(arg) for arg in args]])


def _emit_plain(prefix: str, message: str) -> None:
    print(f"{prefix}{message}", flush=True)


@contextlib.contextmanager
def _suppress_stdio_if_minimal():
    if not _minimal_logs_enabled():
        yield
        return
    saved_fds = []
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        for stream in (sys.__stdout__, sys.__stderr__):
            try:
                fd = stream.fileno()
            except (AttributeError, io.UnsupportedOperation):
                continue
            stream.flush()
            saved_fds.append((fd, os.dup(fd)))
            os.dup2(devnull_fd, fd)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        for fd, saved_fd in reversed(saved_fds):
            os.dup2(saved_fd, fd)
            os.close(saved_fd)
        os.close(devnull_fd)


def _install_minimal_log_routing() -> None:
    global _MINIMAL_LOG_ROUTING_INSTALLED
    if _MINIMAL_LOG_ROUTING_INSTALLED or not _minimal_logs_enabled():
        return

    def _info(message, *args, **kwargs):
        del kwargs
        text = _format_log_message(message, *args)
        if any(text.startswith(prefix) for prefix in _SUPPRESSED_MINIMAL_INFO_PREFIXES):
            return
        _emit_plain("", text)

    def _warning(message, *args, **kwargs):
        del kwargs
        text = _format_log_message(message, *args)
        if any(pattern in text for pattern in _SUPPRESSED_MINIMAL_WARNINGS):
            return
        _emit_plain("WARNING: ", text)

    def _error(message, *args, **kwargs):
        del kwargs
        _emit_plain("ERROR: ", _format_log_message(message, *args))

    logging.info = _info
    logging.warning = _warning
    logging.error = _error
    _MINIMAL_LOG_ROUTING_INSTALLED = True


def _log_model_summary(model, label: str) -> None:
    if _minimal_logs_enabled():
        logging.info(
            "%s: params=%d trainable=%d non_trainable=%d input_shape=%s output_shape=%s",
            label,
            int(model.count_params()),
            len(model.trainable_weights),
            len(model.non_trainable_weights),
            model.input_shape,
            model.output_shape,
        )
        return
    model.summary(print_fn=lambda line: logging.info(line))


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def configure_reproducibility(seed: int, deterministic_ops: bool):
    """Configures process-global reproducibility knobs."""
    if seed >= 0:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)
        logging.info("Using global random seed: %d", seed)

    if deterministic_ops:
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        try:
            tf.config.experimental.enable_op_determinism()
            logging.info("Enabled TensorFlow deterministic ops")
        except Exception as exc:
            logging.warning("Could not enable deterministic ops: %s", exc)


def load_config(flags, model_module):
    """Loads the training configuration from the specified yaml file.

    Args:
        flags (argparse.Namespace): command line flags
        model_module (module): python module for loading the model

    Returns:
        dict: dictionary containing training configuration
    """
    config_filename = flags.training_config
    config = yaml.load(open(config_filename, "r").read(), yaml.Loader)

    config["summaries_dir"] = os.path.join(config["train_dir"], "logs/")

    config["stride"] = flags.__dict__.get("stride", 1)
    config["window_step_ms"] = config.get("window_step_ms", 20)

    # Default preprocessor settings
    preprocessor_sample_rate = 16000  # Hz
    preprocessor_window_size = 30  # ms
    preprocessor_window_step = config["window_step_ms"]  # ms

    desired_samples = int(preprocessor_sample_rate * config["clip_duration_ms"] / 1000)

    window_size_samples = int(
        preprocessor_sample_rate * preprocessor_window_size / 1000
    )
    window_step_samples = int(
        config["stride"] * preprocessor_sample_rate * preprocessor_window_step / 1000
    )

    length_minus_window = desired_samples - window_size_samples

    if length_minus_window < 0:
        config["spectrogram_length_final_layer"] = 0
    else:
        config["spectrogram_length_final_layer"] = 1 + int(
            length_minus_window / window_step_samples
        )

    config["spectrogram_length"] = config[
        "spectrogram_length_final_layer"
    ] + model_module.spectrogram_slices_dropped(flags)

    config["flags"] = flags.__dict__

    config["training_input_shape"] = modes.get_input_data_shape(
        config, modes.Modes.TRAINING
    )

    return config


def train_model(config, model, data_processor, restore_checkpoint):
    """Trains a model.

    Args:
        config (dict): dictionary containing training configuration
        model (Keras model): model architecture to train
        data_processor (FeatureHandler): feature handler that loads spectrogram data
        restore_checkpoint (bool): Whether to restore from checkpoint if model exists

    Raises:
        ValueError: If the model exists but the training flag isn't set
    """
    try:
        os.makedirs(config["train_dir"])
        os.mkdir(config["summaries_dir"])
    except OSError:
        if restore_checkpoint:
            pass
        else:
            raise ValueError(
                "model already exists in folder %s" % config["train_dir"]
            ) from None
    config_fname = os.path.join(config["train_dir"], "training_config.yaml")

    with open(config_fname, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    utils.save_model_summary(model, config["train_dir"])

    train.train(model, config, data_processor)


def evaluate_model(
    config,
    model,
    data_processor,
    test_tf_nonstreaming,
    test_tflite_nonstreaming,
    test_tflite_nonstreaming_quantized,
    test_tflite_streaming,
    test_tflite_streaming_quantized,
):
    """Evaluates a model on test data.

    Saves the nonstreaming model or streaming model in SavedModel format,
    then converts it to TFLite as specified.

    Args:
        config (dict): dictionary containing training configuration
        model (Keras model): model (with loaded weights) to test
        data_processor (FeatureHandler): feature handler that loads spectrogram data
        test_tf_nonstreaming (bool): Evaluate the nonstreaming SavedModel
        test_tflite_nonstreaming_quantized (bool): Convert and evaluate quantized nonstreaming TFLite model
        test_tflite_nonstreaming (bool): Convert and evaluate nonstreaming TFLite model
        test_tflite_streaming (bool): Convert and evaluate streaming TFLite model
        test_tflite_streaming_quantized (bool): Convert and evaluate quantized streaming TFLite model
    """

    if (
        test_tf_nonstreaming
        or test_tflite_nonstreaming
        or test_tflite_nonstreaming_quantized
    ):
        # Save the nonstreaming model to disk
        logging.info("Saving nonstreaming model")

        utils.convert_model_saved(
            model,
            config,
            folder="non_stream",
            mode=modes.Modes.NON_STREAM_INFERENCE,
        )

    if test_tflite_streaming or test_tflite_streaming_quantized:
        # Save the internal streaming model to disk
        logging.info("Saving streaming model")

        utils.convert_model_saved(
            model,
            config,
            folder="stream_state_internal",
            mode=modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
        )

    if test_tf_nonstreaming:
        logging.info("Testing nonstreaming model")

        folder_name = "non_stream"
        test.tf_model_accuracy(
            config,
            folder_name,
            data_processor,
            data_set="testing",
            accuracy_name="testing_set_metrics.txt",
        )

    tflite_configs = []

    if test_tflite_nonstreaming:
        tflite_configs.append(
            {
                "log_string": "nonstreaming model",
                "source_folder": "non_stream",
                "output_folder": "tflite_non_stream",
                "filename": "non_stream.tflite",
                "testing_dataset": "testing",
                "testing_ambient_dataset": "testing_ambient",
                "quantize": False,
            }
        )

    if test_tflite_nonstreaming_quantized:
        tflite_configs.append(
            {
                "log_string": "quantized nonstreaming model",
                "source_folder": "non_stream",
                "output_folder": "tflite_non_stream_quant",
                "filename": "non_stream_quant.tflite",
                "testing_dataset": "testing",
                "testing_ambient_dataset": "testing_ambient",
                "quantize": True,
            }
        )

    if test_tflite_streaming:
        tflite_configs.append(
            {
                "log_string": "streaming model",
                "source_folder": "stream_state_internal",
                "output_folder": "tflite_stream_state_internal",
                "filename": "stream_state_internal.tflite",
                "testing_dataset": "testing",
                "testing_ambient_dataset": "testing_ambient",
                "quantize": False,
            }
        )

    if test_tflite_streaming_quantized:
        tflite_configs.append(
            {
                "log_string": "quantized streaming model",
                "source_folder": "stream_state_internal",
                "output_folder": "tflite_stream_state_internal_quant",
                "filename": "stream_state_internal_quant.tflite",
                "testing_dataset": "testing",
                "testing_ambient_dataset": "testing_ambient",
                "quantize": True,
            }
        )

    for tflite_config in tflite_configs:
        logging.info("Converting %s to TFLite", tflite_config["log_string"])

        utils.convert_saved_model_to_tflite(
            config,
            audio_processor=data_processor,
            path_to_model=os.path.join(config["train_dir"], tflite_config["source_folder"]),
            folder=os.path.join(config["train_dir"], tflite_config["output_folder"]),
            fname=tflite_config["filename"],
            quantize=tflite_config["quantize"],
        )

        logging.info(
            "Testing the TFLite %s false accept per hour and false rejection rates at various cutoffs.",
            tflite_config["log_string"],
        )

        test.tflite_streaming_model_roc(
            config,
            tflite_config["output_folder"],
            data_processor,
            data_set=tflite_config["testing_dataset"],
            ambient_set=tflite_config["testing_ambient_dataset"],
            tflite_model_name=tflite_config["filename"],
            accuracy_name="tflite_streaming_roc.txt",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_config",
        type=str,
        default="trained_models/model/training_parameters.yaml",
        help="""\
        Path to the training parameters yaml configuration.action=
        """,
    )
    parser.add_argument(
        "--train",
        type=int,
        default=1,
        help="If 1 run train and test, else run only test",
    )
    parser.add_argument(
        "--test_tf_nonstreaming",
        type=int,
        default=0,
        help="Save the nonstreaming model and test on the test datasets",
    )
    parser.add_argument(
        "--test_tflite_nonstreaming",
        type=int,
        default=0,
        help="Save the TFLite nonstreaming model and test on the test datasets",
    )
    parser.add_argument(
        "--test_tflite_nonstreaming_quantized",
        type=int,
        default=0,
        help="Save the TFLite quantized nonstreaming model and test on the test datasets",
    )
    parser.add_argument(
        "--test_tflite_streaming",
        type=int,
        default=0,
        help="Save the (non-quantized) streaming model and test on the test datasets",
    )
    parser.add_argument(
        "--test_tflite_streaming_quantized",
        type=int,
        default=1,
        help="Save the quantized streaming model and test on the test datasets",
    )
    parser.add_argument(
        "--restore_checkpoint",
        type=int,
        default=0,
        help="If 1 it will restore a checkpoint and resume the training "
        "by initializing model weights and optimizer with checkpoint values. "
        "It will use learning rate and number of training iterations from "
        "--learning_rate and --how_many_training_steps accordinlgy. "
        "This option is useful in cases when training was interrupted. "
        "With it you should adjust learning_rate and how_many_training_steps.",
    )
    parser.add_argument(
        "--use_weights",
        type=str,
        default="best_weights",
        help="Which set of weights to use when creating the model"
        "One of `best_weights`` or `last_weights`.",
    )

    # Function used to parse --verbosity argument
    def verbosity_arg(value):
        """Parses verbosity argument.

        Args:
        value: A member of tf.logging.

        Returns:
        TF logging mode

        Raises:
        ArgumentTypeError: Not an expected value.
        """
        value = value.upper()
        if value == "INFO":
            return logging.INFO
        elif value == "DEBUG":
            return logging.DEBUG
        elif value == "ERROR":
            return logging.ERROR
        elif value == "FATAL":
            return logging.FATAL
        elif value == "WARN":
            return logging.WARN
        else:
            raise argparse.ArgumentTypeError("Not an expected value")

    parser.add_argument(
        "--verbosity",
        type=verbosity_arg,
        default=logging.INFO,
        help='Log verbosity. Can be "INFO", "DEBUG", "ERROR", "FATAL", or "WARN"',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_env_int("MICRO_TRAIN_SEED", -1),
        help="Global random seed. Set -1 to disable explicit seeding.",
    )
    parser.add_argument(
        "--deterministic_ops",
        type=int,
        default=_env_int("MICRO_TRAIN_DETERMINISTIC", 0),
        help="If 1, request deterministic TensorFlow ops where supported.",
    )

    # sub parser for model settings
    subparsers = parser.add_subparsers(dest="model_name", help="NN model name")

    # inception model settings
    parser_inception = subparsers.add_parser("inception")
    inception.model_parameters(parser_inception)

    # mixednet model settings
    parser_mixednet = subparsers.add_parser("mixednet")
    mixednet.model_parameters(parser_mixednet)

    flags, unparsed = parser.parse_known_args()
    if unparsed:
        raise ValueError("Unknown argument: {}".format(unparsed))

    if flags.model_name == "inception":
        model_module = inception
    elif flags.model_name == "mixednet":
        model_module = mixednet
    else:
        raise ValueError("Unknown model type: {}".format(flags.model_name))

    logging.set_verbosity(flags.verbosity)
    _install_minimal_log_routing()
    configure_reproducibility(flags.seed, bool(flags.deterministic_ops))

    config = load_config(flags, model_module)

    data_processor = input_data.FeatureHandler(config)

    if flags.train:
        with _suppress_stdio_if_minimal():
            model = model_module.model(
                flags, config["training_input_shape"], config["batch_size"]
            )
        _log_model_summary(model, "Training model")
        train_model(config, model, data_processor, flags.restore_checkpoint)
    else:
        if not os.path.isdir(config["train_dir"]):
            raise ValueError('model is not trained set "--train 1" and retrain it')

    if (
        flags.test_tf_nonstreaming
        or flags.test_tflite_nonstreaming
        or flags.test_tflite_streaming
        or flags.test_tflite_streaming_quantized
    ):
        with _suppress_stdio_if_minimal():
            model = model_module.model(
                flags, shape=config["training_input_shape"], batch_size=1
            )

        model.load_weights(
            os.path.join(config["train_dir"], flags.use_weights) + ".weights.h5"
        )

        _log_model_summary(model, "Evaluation model")

        evaluate_model(
            config,
            model,
            data_processor,
            flags.test_tf_nonstreaming,
            flags.test_tflite_nonstreaming,
            flags.test_tflite_nonstreaming_quantized,
            flags.test_tflite_streaming,
            flags.test_tflite_streaming_quantized,
        )
