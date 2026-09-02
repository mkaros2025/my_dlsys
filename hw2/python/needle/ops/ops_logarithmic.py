from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=-1, keepdims=True)
        exp_z = array_api.exp(Z - max_z)
        sum_exp = array_api.sum(exp_z, axis=-1, keepdims=True)
        return Z - (max_z + array_api.log(sum_exp))
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        softmax = exp(node)
        # 沿着最后一轴求和再补回一个长度为 1 的类别轴
        grad_sum = out_grad.sum(axes=-1).reshape(
            out_grad.shape[:-1] + (1,)
        )
        grad_sum = grad_sum.broadcast_to(out_grad.shape)

        return out_grad - softmax * grad_sum
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        exp_z = array_api.exp(Z - max_z)
        sum_exp = array_api.sum(exp_z, axis=self.axes, keepdims=True)
        result = max_z + array_api.log(sum_exp)

        if self.axes is not None:
            result = array_api.squeeze(result, axis=self.axes)
        else:
            result = result.reshape(())

        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        input_shape = Z.shape

        # self.axes 是这个算子要压缩的轴，这里就是得到要压缩的轴
        if self.axes is None:
            axes = tuple(range(len(input_shape)))
        elif isinstance(self.axes, int):
            axes = (self.axes, )
        else:
            axes = self.axes

        # 全部变成正数
        axes = tuple(
            a if a >= 0 else a + len(input_shape) 
            for a in axes
        )

        keep_shape = list(input_shape)
        for a in axes:
            keep_shape[a] = 1
        # 得到 squeeze 之前 result 的形状
        keep_shape = tuple(keep_shape)

        # 把 node 从最后的 result 形状变回 squeeze 之前的形状
        output = node.reshape(keep_shape).broadcast_to(input_shape)
        grad = out_grad.reshape(keep_shape).broadcast_to(input_shape)

        return grad * exp(Z - output)
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)