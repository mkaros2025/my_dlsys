import gzip
import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from simple_ml import (  # noqa: E402
    add,
    nn_epoch,
    parse_mnist,
    softmax_loss,
    softmax_regression_epoch,
)


def _write_idx_images(path, images):
    images = np.asarray(images, dtype=np.uint8)
    assert images.ndim == 3
    count, rows, cols = images.shape
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">IIII", 2051, count, rows, cols))
        f.write(images.tobytes())


def _write_idx_labels(path, labels):
    labels = np.asarray(labels, dtype=np.uint8)
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">II", 2049, labels.shape[0]))
        f.write(labels.tobytes())


def _softmax_probs(logits):
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def _reference_softmax_loss(logits, y):
    logits = np.asarray(logits)
    return np.mean(np.log(np.exp(logits).sum(axis=1)) - logits[np.arange(y.size), y])


def _reference_softmax_epoch(X, y, theta, lr, batch):
    for start in range(0, X.shape[0], batch):
        Xb = X[start : start + batch]
        yb = y[start : start + batch]
        probs = _softmax_probs(Xb @ theta)
        probs[np.arange(yb.size), yb] -= 1
        theta -= lr * (Xb.T @ probs) / yb.size


def _reference_nn_epoch(X, y, W1, W2, lr, batch):
    for start in range(0, X.shape[0], batch):
        Xb = X[start : start + batch]
        yb = y[start : start + batch]
        pre_relu = Xb @ W1
        hidden = np.maximum(pre_relu, 0)
        probs = _softmax_probs(hidden @ W2)
        probs[np.arange(yb.size), yb] -= 1
        grad_W2 = hidden.T @ probs / yb.size
        grad_hidden = (probs @ W2.T) * (pre_relu > 0)
        grad_W1 = Xb.T @ grad_hidden / yb.size
        W1 -= lr * grad_W1
        W2 -= lr * grad_W2


def test_add_supports_numpy_broadcasting_without_mutating_inputs():
    x = np.array([[1.0], [2.0], [3.0]])
    y = np.array([10.0, 20.0])
    x_before = x.copy()
    y_before = y.copy()

    out = add(x, y)

    np.testing.assert_allclose(out, np.array([[11.0, 21.0], [12.0, 22.0], [13.0, 23.0]]))
    np.testing.assert_array_equal(x, x_before)
    np.testing.assert_array_equal(y, y_before)


def test_parse_mnist_reads_big_endian_idx_and_normalizes_globally(tmp_path):
    images = np.array(
        [
            [[0, 127, 255], [10, 20, 30]],
            [[1, 2, 3], [4, 5, 6]],
        ],
        dtype=np.uint8,
    )
    labels = np.array([7, 2], dtype=np.uint8)
    image_path = tmp_path / "images.gz"
    label_path = tmp_path / "labels.gz"
    _write_idx_images(image_path, images)
    _write_idx_labels(label_path, labels)

    X, y = parse_mnist(str(image_path), str(label_path))

    assert X.dtype == np.float32
    assert y.dtype == np.uint8
    assert X.shape == (2, 6)
    np.testing.assert_allclose(X, images.reshape(2, 6).astype(np.float32) / 255.0)
    np.testing.assert_array_equal(y, labels)


def test_softmax_loss_matches_definition_for_nontrivial_batch():
    logits = np.array(
        [
            [1.25, -0.75, 0.5, 2.0],
            [-1.0, 0.0, 3.0, 0.25],
            [0.1, -0.2, -0.3, 0.4],
        ],
        dtype=np.float64,
    )
    y = np.array([3, 2, 0], dtype=np.uint8)

    np.testing.assert_allclose(softmax_loss(logits, y), _reference_softmax_loss(logits, y))


def test_softmax_loss_uses_batch_average_not_sum():
    logits_one = np.array([[0.2, -0.4, 1.1]], dtype=np.float64)
    y_one = np.array([2], dtype=np.uint8)
    logits = np.repeat(logits_one, repeats=5, axis=0)
    y = np.repeat(y_one, repeats=5, axis=0)

    np.testing.assert_allclose(softmax_loss(logits, y), softmax_loss(logits_one, y_one))


def test_softmax_regression_epoch_matches_reference_and_updates_in_place():
    X = np.array(
        [
            [0.5, -1.0, 2.0],
            [1.5, 0.0, -0.5],
            [-1.0, 2.0, 0.25],
            [0.0, -0.75, 1.25],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 2, 1, 2], dtype=np.uint8)
    theta = np.array(
        [[0.1, -0.2, 0.3], [0.0, 0.25, -0.15], [-0.4, 0.2, 0.05]],
        dtype=np.float32,
    )
    expected = theta.copy()

    returned = softmax_regression_epoch(X, y, theta, lr=0.3, batch=2)
    _reference_softmax_epoch(X, y, expected, lr=0.3, batch=2)

    assert returned is None
    np.testing.assert_allclose(theta, expected, rtol=1e-6, atol=1e-6)


def test_softmax_regression_epoch_respects_minibatch_order():
    X = np.array(
        [[1.0, 0.0], [0.0, 1.0], [2.0, -1.0], [-1.0, 2.0]],
        dtype=np.float32,
    )
    y = np.array([0, 1, 0, 1], dtype=np.uint8)
    theta_batch_2 = np.zeros((2, 2), dtype=np.float32)
    theta_full_batch = np.zeros((2, 2), dtype=np.float32)
    expected_batch_2 = theta_batch_2.copy()
    expected_full_batch = theta_full_batch.copy()

    softmax_regression_epoch(X, y, theta_batch_2, lr=0.4, batch=2)
    softmax_regression_epoch(X, y, theta_full_batch, lr=0.4, batch=4)
    _reference_softmax_epoch(X, y, expected_batch_2, lr=0.4, batch=2)
    _reference_softmax_epoch(X, y, expected_full_batch, lr=0.4, batch=4)

    np.testing.assert_allclose(theta_batch_2, expected_batch_2, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(theta_full_batch, expected_full_batch, rtol=1e-6, atol=1e-6)
    assert not np.allclose(theta_batch_2, theta_full_batch)


def test_nn_epoch_matches_reference_and_updates_both_layers_in_place():
    X = np.array(
        [
            [1.0, -2.0, 0.5],
            [0.0, 1.5, -1.0],
            [-1.0, 0.25, 2.0],
            [2.0, 0.5, -0.5],
        ],
        dtype=np.float32,
    )
    y = np.array([1, 0, 2, 1], dtype=np.uint8)
    W1 = np.array(
        [[0.2, -0.4, 0.1, 0.3], [-0.5, 0.25, 0.4, -0.2], [0.1, 0.3, -0.6, 0.2]],
        dtype=np.float32,
    )
    W2 = np.array(
        [[0.3, -0.2, 0.1], [-0.4, 0.5, 0.2], [0.1, -0.3, 0.6], [0.2, 0.1, -0.5]],
        dtype=np.float32,
    )
    expected_W1 = W1.copy()
    expected_W2 = W2.copy()

    returned = nn_epoch(X, y, W1, W2, lr=0.2, batch=2)
    _reference_nn_epoch(X, y, expected_W1, expected_W2, lr=0.2, batch=2)

    assert returned is None
    np.testing.assert_allclose(W1, expected_W1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(W2, expected_W2, rtol=1e-6, atol=1e-6)


def test_nn_epoch_relu_gradient_is_zero_for_inactive_hidden_units():
    X = np.array([[1.0, 1.0], [2.0, -1.0]], dtype=np.float32)
    y = np.array([0, 1], dtype=np.uint8)
    W1 = np.array([[-10.0, 0.5], [-10.0, -0.25]], dtype=np.float32)
    W2 = np.array([[1.0, -1.0], [0.5, -0.5]], dtype=np.float32)
    expected_W1 = W1.copy()
    expected_W2 = W2.copy()

    nn_epoch(X, y, W1, W2, lr=0.1, batch=2)
    _reference_nn_epoch(X, y, expected_W1, expected_W2, lr=0.1, batch=2)

    np.testing.assert_allclose(W1, expected_W1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(W2, expected_W2, rtol=1e-6, atol=1e-6)


def test_cpp_softmax_epoch_matches_python_reference_if_extension_is_built():
    if importlib.util.find_spec("simple_ml_ext") is None:
        pytest.skip("Run `make` in hw0 to build src/simple_ml_ext.so")

    from simple_ml_ext import softmax_regression_epoch_cpp

    X = np.array(
        [[0.25, -0.5, 1.0], [1.0, 0.75, -1.5], [-0.25, 2.0, 0.5], [1.5, -1.0, 0.0]],
        dtype=np.float32,
    )
    y = np.array([2, 0, 1, 2], dtype=np.uint8)
    theta = np.array(
        [[0.2, -0.1, 0.0], [-0.3, 0.4, 0.1], [0.05, -0.2, 0.25]],
        dtype=np.float32,
    )
    expected = theta.copy()

    softmax_regression_epoch_cpp(X, y, theta, lr=0.15, batch=2)
    _reference_softmax_epoch(X, y, expected, lr=0.15, batch=2)

    np.testing.assert_allclose(theta, expected, rtol=1e-6, atol=1e-6)
