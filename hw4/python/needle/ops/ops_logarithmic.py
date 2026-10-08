from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        pass
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            axes = set(range(len(Z.shape)))
        elif isinstance(self.axes, int):
            axes = {self.axes % len(Z.shape)}
        else:
            axes = {ax % len(Z.shape) for ax in self.axes}

        max_z = Z
        for ax in axes:
            max_z = max_z.max(axis = ax, keepdims = True)

        exp_z = (Z - max_z.broadcast_to(Z.shape)).exp()

        sum_exp = exp_z
        for ax in axes:
            sum_exp = sum_exp.sum(axis=ax, keepdims=True)

        res = max_z + sum_exp.log()

        out_shape = tuple(s for i, s in enumerate(Z.shape) if i not in axes)
        return res.compact().reshape(out_shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape

        if self.axes is None:
            axes = set(range(len(input_shape)))
        elif isinstance(self.axes, int):
            axes = {self.axes % len(input_shape)}
        else:
            axes = {ax % len(input_shape) for ax in self.axes}

        shape_with_ones = tuple(1 if i in axes else input_shape[i] for i in range(len(input_shape)))

        out_grad = out_grad.reshape(shape_with_ones).broadcast_to(input_shape)
        node_broadcast = node.reshape(shape_with_ones).broadcast_to(input_shape)

        return out_grad * exp(node.inputs[0] - node_broadcast)
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)