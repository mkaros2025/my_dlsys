"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        return out_grad * self.scalar * (x ** (self.scalar - 1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b;
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x, y = node.inputs[0], node.inputs[1]
        grad_x = out_grad / y
        grad_y = -out_grad * x / (y ** 2)

        return (grad_x, grad_y)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        ndim = len(a.shape)
        axes_list = list(range(ndim))

        if self.axes is None:
            ax1 = ndim - 2
            ax2 = ndim - 1
        else:
            ax1 = self.axes[0]
            ax2 = self.axes[1]

        axes_list[ax1], axes_list[ax2] = axes_list[ax2], axes_list[ax1]

        return a.permute(tuple(axes_list))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.transpose(self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.compact().reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.reshape(node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.broadcast_to(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        output_shape = self.shape

        padded_shape = (1,) * (len(output_shape) - len(input_shape)) + input_shape

        axes = []
        for i, (in_dim, out_dim) in enumerate(zip(padded_shape, output_shape)):
            if in_dim != out_dim:
                axes.append(i)

        grad = out_grad.sum(tuple(axes)) if axes else out_grad

        return grad.reshape(input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return a.sum()
        if isinstance(self.axes, int):
            return a.sum(self.axes)

        for ax in sorted(list(self.axes), reverse=True):
            a = a.sum(ax)
        return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        if self.axes is None:
            axes = set(range(len(input_shape)))
        elif isinstance(self.axes, int):
            axes = {self.axes % len(input_shape)}
        else:
            axes = {ax % len(input_shape) for ax in self.axes}

        shape_with_ones = tuple(1 if i in axes else input_shape[i] for i in range(len(input_shape)))

        return out_grad.reshape(shape_with_ones).broadcast_to(input_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs[0], node.inputs[1]
        grad_b = a.transpose() @ out_grad
        grad_a = out_grad @ b.transpose()

        return (grad_a, grad_b)
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.log()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        return out_grad / x
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.exp()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * node
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.maximum(0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        data = node.realize_cached_data()
        mask = data > 0
        mask_tensor = Tensor(mask, device=node.device, requires_grad=False)
        return out_grad * mask_tensor
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.tanh()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (1 - node ** 2)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        new_shape = list(args[0].shape)
        new_shape.insert(self.axis, len(args))
        new_shape = tuple(new_shape)

        out = NDArray.make(new_shape, device=args[0].device)

        slices = [slice(None)] * len(new_shape)
        for i, arr in enumerate(args):
            slices[self.axis] = i
            out[tuple(slices)] = arr

        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, axis=self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        # Get the shape without the self.axis
        sub_shape = tuple(s for j, s in enumerate(A.shape) if j != self.axis)

        slices = [slice(None)] * A.ndim
        splits = []
        for i in range(A.shape[self.axis]):
            slices[self.axis] = i
            # compress the axis i
            part = A[tuple(slices)].compact().reshape(sub_shape)
            splits.append(part)

        return tuple(splits)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, axis=self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        for ax in self.axes:
            new_shape[ax] *= (self.dilation + 1)
        new_shape = tuple(new_shape)

        out = NDArray.make(new_shape, device=a.device)
        out.fill(0)

        slices = [slice(None)] * a.ndim
        for ax in self.axes:
            slices[ax] = slice(0, None, self.dilation + 1)

        out[tuple(slices)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        slices = [slice(None)] * a.ndim
        for ax in self.axes:
            slices[ax] = slice(0, None, self.dilation + 1)
        return a[tuple(slices)].compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        # If padding is needed, pad along the Height and Width dimensions
        if self.padding > 0:
            pad_axes = ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
            A_pad = A.pad(pad_axes)
        else:
            A_pad = A

        N, H_in, W_in, C_in = A_pad.shape
        K, _, _, C_out = B.shape

        # Output spatial dimensions determined by the number of valid sliding window positions
        H_out = (H_in - K) // self.stride + 1
        W_out = (W_in - K) // self.stride + 1

        # Construct a 6D view of shape (N, H_out, W_out, K, K, C_in) with zero data copying
        s_N, s_H, s_W, s_C = A_pad.strides
        new_shape = (N, H_out, W_out, K, K, C_in)
        new_strides = (s_N, self.stride * s_H, self.stride * s_W, s_H, s_W, s_C)
        A_strided = A_pad.as_strided(new_shape, new_strides)

        # Compact and flatten into 2D matrices for hardware-accelerated GEMM
        A_mat = A_strided.compact().reshape((N * H_out * W_out, K * K * C_in)) 
        B_mat = B.compact().reshape((K * K * C_in, C_out))

        # Perform 2D matrix multiplication and restore the 4D output tensor shape
        out_mat = A_mat @ B_mat
        return out_mat.reshape((N, H_out, W_out, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.stride > 1:
            out_grad_dilated = dilate(out_grad, axes=(1, 2), dilation=self.stride - 1)
        else:
            out_grad_dilated = out_grad

        A, B = node.inputs[0], node.inputs[1]
        B_flip = flip(B, axes=(0, 1))
        B_trans = B_flip.transpose(axes=(2, 3))

        # H_out = H + 2P - K + 1
        # H = H_out - 2P + K - 1
        pad_A = B.shape[0] - 1 - self.padding
        grad_A = conv(out_grad_dilated, B_trans, padding=pad_A, stride=1)

        A_perm = A.transpose((0, 3))
        grad_perm = out_grad_dilated.transpose((0, 1)).transpose((1, 2))
        conv_B = conv(A_perm, grad_perm, padding=self.padding, stride=1)
        grad_B = conv_B.transpose((0, 1)).transpose((1, 2))

        return (grad_A, grad_B)
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


