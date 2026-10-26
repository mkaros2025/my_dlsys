在 HW1 中，我们完成了计算图和反向模式自动微分，实现了一个最基础的自动求导引擎。在 HW1 的最后一题里，我们也确实用它训练了一个两层的神经网络，但写起来非常繁琐：
权重的初始化、矩阵维度的匹配、训练数据切 batch、参数的更新（为了避免把减法建进图里还要把数值抠成 numpy），全都是我们在外面用裸代码手写的。

如果每一层网络、每一个训练脚本都要这么写，稍微搭个深一点的网络（比如十几层的 ResNet）代码就会极其混乱，根本没法维护。

所以 HW1 实现的只能叫**自动微分引擎（Autograd Engine）**（相当于 PyTorch 底层的 `torch.autograd`），它只负责对数学算式求导，根本没有“神经网络”的工程抽象。

而 HW2 的任务，就是在 HW1 自动求导引擎的基础上，进一步构建出一个真正的**神经网络库（Neural Network Library）**，也就是我们平时用 PyTorch 时最熟悉的那些上层抽象：

1. **模块与网络层（`needle.nn`）**：比如 `Linear`、`ReLU`、`BatchNorm`、`Dropout`。把参数的创建初始化和前向计算封装在类内部，通过 `Sequential` 或残差结构方便地拼装网络。
2. **优化器（`needle.optim`）**：比如 `SGD`、`Adam`。把参数更新逻辑标准化，调用 `opt.step()` 就能自动遍历参数完成更新，不需要在外面手动去写那些防建图的脏代码。
3. **数据加载（`needle.data`）**：比如 `Dataset` 和 `DataLoader`。把读取数据、随机打乱（shuffle）、分批组装（batch）和数据增强（翻转、裁剪）封装成标准的数据流水线。

有了这些上层基础设施，我们才能像使用现代深度学习框架一样，模块化地定义和训练模型。

#### 0. 代码迁移与环境准备

HW2 直接基于 HW1 构建，需要直接复用 HW1 中写好的两个核心底层文件：
- `python/needle/autograd.py`：计算图拓扑排序与反向自动微分引擎；
- `python/needle/ops/ops_mathematic.py`：基础数学算子的前向与反向实现。

**注意点**：张量默认数据类型为 `float32`。在后续作业中创建任何 Tensor 时都要保持 `float32`，不要随意改动 dtype，以防与测试用例的精度要求不一致导致报错。


#### 1. 权重初始化方法
这一节里要实现多个常见的权重初始化方法，notebook 还附加了各个方法的论文，不过我觉得要搞懂各个初始化方法为什么好还是太费劲了，实在没有必要，我们这里只做了解各个方法的适用场景以及如何实现。

##### 前置概念：什么是激活函数？为什么必须有它？

在理解初始化之前，先要搞清楚什么是激活函数（Activation Function）：
- **如果不加激活函数会怎样？**：
  假设一个两层网络只做矩阵乘法：$y = (x W_1) W_2$。根据乘法结合律，$(x W_1) W_2 = x (W_1 W_2) = x W'$。由于矩阵相乘依然是矩阵，**无论网络叠 10 层还是 100 层，数学上都只等价于 1 层线性变换**！网络只能画平直的超平面，根本画不出弯曲边界来分类复杂的现实数据。
- **激活函数的真正作用**：
  就是在两层矩阵乘法中间插进一个**非线性的“弯曲开关”**（例如 $y = \operatorname{ReLU}(x W_1) W_2$）。这一步非线性操作打破了乘法结合律，使得网络无法再塌陷成一层，真正拥有了弯折拟合任意复杂曲线的能力。
- **为什么叫“激活”？**：
  借用生物神经元“刺激达到阈值才放电响应”的概念。比如最常用的 $\operatorname{ReLU}(x) = \max(0, x)$：正数（有特征信号）被“激活”原样通过，负数（无用噪音）直接被抑制截断为 0。

##### 为什么需要特定的初始化？
- **为什么不能全设为 0（打破对称性）**：
  如果同一层的 100 个神经元权重全设为 0，前向计算时它们算出的值完全一样，反向传播时分到的梯度也完全一样，更新后依然完全一样。这就好比 100 个神经元在“照镜子”，永远步调一致，彻底退化成了 1 个神经元。所以必须用**随机**初始化打破这种对称性，让每个神经元一开始就长得不同，去学习不同的特征。
- **为什么不能随便乱随机（控制方差）**：
  如果直接用常见的标准正态分布 $\mathcal{N}(0, 1)$，网络层数一多，矩阵连续相乘会导致数值方差像滚雪球一样逐层放大（引起数值溢出/梯度爆炸），或者越来越小趋近于 0（梯度消失）。
初始化的核心目标就是：**让数据在向前传播穿过多层网络时，每一层的数值方差尽量维持在相同的数量级，不爆炸也不消失。**

##### 两个流派的适用场景与推导

线性层的前向计算为 $y = x W$。其中输入特征维度记为 **`fan_in`**（矩阵行数），输出特征维度记为 **`fan_out`**（矩阵列数）。
由于每个输出神经元是由 $\text{fan\_in}$ 个输入加权累加得到的（$y_j = \sum_{i=1}^{\text{fan\_in}} x_i W_{ij}$），输出方差会被自然放大 $\text{fan\_in}$ 倍。为了让前后的方差保持平衡，权重的理论方差必须除以神经元数量（$\operatorname{Var}(W) \propto \frac{1}{\text{fan}}$）来抵消累加效应。

针对不同的激活函数，推导分化为两个流派：

1. **Xavier 初始化（适合无激活函数、或 Sigmoid/Tanh 等对称激活）**
   - **理论方差**：前向传播累加了 $\text{fan\_in}$ 项，而反向传播（$\nabla x = \nabla y W^T$）累加了 $\text{fan\_out}$ 项。Xavier 假设激活函数在正负区间平滑对称，为了兼顾前向与反向传播的稳定性，取两者折中，得到方差：
     $$
     \sigma^2 = \frac{2}{\text{fan\_in} + \text{fan\_out}}
     $$
   - **Xavier Normal**：直接采样自高斯正态分布 $\mathcal{N}(0, \text{std}^2)$，其中标准差 $\text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}$。
   - **Xavier Uniform**：连续均匀分布 $\mathcal{U}(-a, a)$ 的方差为 $\sigma^2 = \frac{a^2}{3}$， $a = \sigma \sqrt{3}$。将理论方差乘 3 开根号，分子变为 $\sqrt{6}$，即从 $\mathcal{U}(-a, a)$ 采样，其中 $a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}$。

2. **Kaiming 初始化（专为 ReLU 这类非对称截断激活设计）**
   - **理论方差**：ReLU 极其不对称，强行把所有负数抹成 0，导致信号在经过激活后方差缩水一半（乘以 $\frac{1}{2}$）。何恺明团队通过将增益设为 $\text{gain} = \sqrt{2}$（方差翻倍）来补平这一折损；同时公式专注于前向输入维度 `fan_in`，得到方差：
     $$
     \sigma^2 = \frac{2}{\text{fan\_in}} = \frac{\text{gain}^2}{\text{fan\_in}}
     $$
   - **Kaiming Normal**：直接采样自高斯正态分布 $\mathcal{N}(0, \text{std}^2)$，其中标准差 $\text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_in}}}$。
   - **Kaiming Uniform**：同理根据均匀分布 $a = \sigma \sqrt{3}$，将理论方差乘 3 开根号，分子变为 $\text{gain} \times \sqrt{3}$，即从 $\mathcal{U}(-\text{bound}, \text{bound})$ 采样，其中 $\text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_in}}}$。

##### 代码实现

底层的随机张量生成在 `init_basic.py` 中已经封装好了 `rand`（均匀分布）和 `randn`（正态分布）。我们在 `init_initializers.py` 中要做的事情非常直接，就是按公式算好边界参数，直接调用底层的生成函数：

```python
def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    # 算均匀分布边界 a，在 [-a, a] 之间采样
    a = gain * math.sqrt(6.0 / float(fan_in + fan_out))
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)

def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    # 算正态分布标准差 std，生成零均值高斯分布
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return randn(fan_in, fan_out, mean=0.0, std=std, **kwargs)

def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    # ReLU 砍掉负数导致方差折半，强制 gain = sqrt(2) 补偿翻倍；算边界并在 [-bound, bound] 采样
    gain = math.sqrt(2)
    bound = gain * math.sqrt(3 / fan_in)
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)

def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    # 同样设 gain = sqrt(2) 补偿方差；算标准差生成正态分布
    gain = math.sqrt(2)
    std = gain / math.sqrt(fan_in)
    return randn(fan_in, fan_out, mean=0.0, std=std, **kwargs)
```

#### 2. 神经网络核心层与损失函数封装

##### 1. 为什么需要从算子抽象出 Module？
在 HW1 里，我们训练网络的方式还非常原始：
每次训练都要在外面手动初始化一堆权重变量（如 `W1, W2`），传给训练函数；在训练循环里，还要手动写出具体的计算式（`logits = relu(X @ W1) @ W2`）。
如果网络只有两层还好，一旦网络变成十几层甚至几十层，外部就得维护几十个权重变量，计算式也会写得又长又乱。

所以，我们需要像 PyTorch 一样，把网络封装成面向对象的类——这就是 `Module`：
- **什么是层（Layer）**：其实就是一个带权重的算子。普通的函数（比如加法、乘法）自己是不存数据的，算完就走；但像全连接层这种网络层，它自己带着权重（`weight`）和偏置（`bias`）。我们把“存权重”和“做计算”绑在一起做成一个类，这就是一个层。
- **`Module` 负责干什么**：
  1. **管理参数**：只要把权重存成 `self.weight = Parameter(...)`，调用 `model.parameters()` 时，它就会自动把所有参数找出来打包成列表，直接丢给优化器去更新，不需要在外面手动维护参数列表。
  2. **支持嵌套**：比如 `Sequential` 也是一个 `Module`，但它里面装着好几个子层。当我们调用外层的大模型时，数据会按顺序穿过里面的每一个小层；向大模型要参数时，它也会自动把所有小层的参数都收上来。
  3. **区分训练与测试**：提供 `train()` 和 `eval()` 方法，通过修改 `self.training` 标志位，来控制像 Dropout、BatchNorm 这种在训练和测试时行为不一样的层。
  4. **注意广播问题**：Needle 不支持类似 NumPy 的自动隐式广播。如果在前向计算中矩阵和偏置相加（比如输出是二维矩阵，偏置是一维向量），必须手动写 `self.bias.broadcast_to(out.shape)`。

##### 2. 代表性模块实现与分析
这一节一共要实现 11 个算子和网络层，但它们的写法非常相似。挑出其中最核心、最具代表性的 3 个：

###### 基础带参层：`Linear`
全连接层是最典型的带参数网络层，前向计算公式为 $y = x W + b$。
它的核心有两点：
1. **参数初始化**：权重和偏置都要用 `Parameter` 包起来，这样优化器才能识别到；
2. **显式广播**：由于 Needle 不支持自动广播，偏置相加时必须手动调用 `broadcast_to`。

```python
class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 权重用 Kaiming 均匀分布初始化，并用 Parameter 包裹
        self.weight = Parameter(
            init.kaiming_uniform(fan_in=in_features, fan_out=out_features, device=device, dtype=dtype)
        )
        # 偏置初始化为 (1, out_features) 的行向量，方便后续广播
        if bias:
            self.bias = Parameter(
                init.kaiming_uniform(fan_in=out_features, fan_out=1, device=device, dtype=dtype).reshape((1, out_features))
            )
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        # X 形状 (B, in_features) 乘权重 (in_features, out_features) 得到 (B, out_features)
        out = X @ self.weight
        if self.bias is not None:
            # Needle 无法自动广播，必须显式把偏置广播到与 out 相同的形状
            out = out + self.bias.broadcast_to(out.shape)
        return out
```

###### 数值稳定归约算子：`LogSumExp`

在分类任务中，模型输出的是各个类别的未归一化得分（logits），记为 $Z$，形状通常为 $(B, K)$（$B$ 是 batch 大小，$K$ 是类别数）。`LogSumExp` 负责沿指定轴（如类别轴）将这些分数计算对数指数和：$R = \log \sum_i \exp(Z_i)$，是交叉熵损失函数的底层核心。

**1. 数值稳定原理（为什么要减最大值？）**
如果某个 logits 数值很大（例如 1000），直接算 $e^{1000}$ 会导致计算机浮点数溢出变成 `inf`，整个训练直接崩溃。
数学上利用恒等式：
$$
\log \sum_i \exp(Z_i) = c + \log \sum_i \exp(Z_i - c)
$$
只要令 $c = \max(Z)$，减去最大值后所有指数项 $Z_i - c \le 0$，最大值只有 $e^0 = 1$，**彻底在数学底层杜绝了溢出**。

**2. 局部导数与反向公式**
- **符号约定**：
  - 输入 logits 记为 $Z$；输出对数和记为 $R$（代码中的 `node`）；
  - 最终标量损失记为 $L$；上游传回的梯度记为 $G = \frac{\partial L}{\partial R}$（代码中的 `out_grad`）。
- **局部导数推导**：
  对输入 $Z_i$ 求偏导，链式法则展开：
  $$
  \frac{\partial R}{\partial Z_i} = \frac{1}{\sum_j \exp(Z_j)} \cdot \exp(Z_i) = \frac{\exp(Z_i)}{\sum_j \exp(Z_j)} = P_i
  $$
  可以看到：**`LogSumExp` 的局部导数正好就是 Softmax 概率 $P_i$**！
  又因为由定义可知分母 $\sum_j \exp(Z_j) = \exp(R)$，所以该导数可直接写为：
  $$
  P_i = \frac{\exp(Z_i)}{\exp(R)} = \exp(Z_i - R)
  $$
- **反向传播公式**：
  根据链式法则 $\frac{\partial L}{\partial Z_i} = G \cdot P_i$。因 $R$ 和 $G$ 在前向计算时被压缩（squeeze）掉了维度，反向需广播回输入形状：
  $$
  \frac{\partial L}{\partial Z} = \operatorname{broadcast}(G) \odot \exp(Z - \operatorname{broadcast}(R))
  $$
  在代码中直接体现为 `grad * exp(Z - output)`。

```python
class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        # 1. 沿指定轴找到最大值 M，keepdims=True 保留维度方便广播相减
        max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        # 2. 核心防溢出：exp(Z - M) 最大只会是 exp(0) = 1，绝不会产生 inf
        exp_z = array_api.exp(Z - max_z)
        sum_exp = array_api.sum(exp_z, axis=self.axes, keepdims=True)
        result = max_z + array_api.log(sum_exp)

        # 3. 按照约定，把求和压缩掉的维度 squeeze 掉
        if self.axes is not None:
            return array_api.squeeze(result, axis=self.axes)
        return result.reshape(())

    def gradient(self, out_grad: Tensor, node: Tensor):
        # 导数公式为 Softmax: grad * exp(Z - LogSumExp(Z))
        Z = node.inputs[0]
        input_shape = Z.shape

        # 统一 axes 格式，处理负数索引与 None 的情况
        axes = self.axes
        if axes is None:
            axes = tuple(range(len(input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        axes = tuple(a if a >= 0 else a + len(input_shape) for a in axes)

        # 构造 keep_shape 恢复维度：
        # 前向归约把求和轴 squeeze 掉了（如 (B, K) 变成了 (B,)）。
        # 反向的上游梯度是 (B,)，不能直接广播到 (B, K)（因为广播从末尾对齐，会误将 B 与 K 对齐）。
        # 所以必须把被删掉的轴替换为 1（变成 (B, 1) 的 keep_shape），才能正确广播回 (B, K)。
        keep_shape = list(input_shape)
        for a in axes:
            keep_shape[a] = 1
        keep_shape = tuple(keep_shape)

        # 将前向输出 node(即 R) 和上游梯度 out_grad(即 G) 恢复为 keep_shape 后广播回原形状
        output = node.reshape(keep_shape).broadcast_to(input_shape)
        grad = out_grad.reshape(keep_shape).broadcast_to(input_shape)

        # exp(Z - output) 复用了前向算好的 R，直接得出局部导数 Softmax 概率
        return grad * exp(Z - output)
```

###### 类别对数概率算子：`LogSoftmax`

与 `LogSumExp` 把维度压缩掉不同，`LogSoftmax` 输出的形状与输入保持一致（仍为 $(B, K)$）。它直接计算每个类别概率的对数值，适合与交叉熵配合使用。

**公式推导与代码**
- **前向语义**：
  沿最后一轴（类别轴）计算对数概率：
  $$
  Y_i = \log P_i = Z_i - \operatorname{LogSumExp}(Z) = Z_i - \log \sum_k \exp(Z_k)
  $$
- **局部导数推导（微调输入 $Z_j$ 会影响哪些输出？）**：
  根据上式对输入的第 $j$ 个类别 $Z_j$ 求导：
  - **对自身（$i = j$）**：前面包含自身项（导数为 1），减号后项对数求和（导数为 $P_j$），结果为 $1 - P_j$；
  - **对其他类别（$i \neq j$）**：前面无自身项（导数为 0），后项分母依然包含 $Z_j$（导数为 $P_j$），结果为 $0 - P_j = -P_j$；
  - 统一用指示符号 $\mathbf{1}_{i=j}$（$i=j$ 时为 1，否则为 0）写成：
    $$
    \frac{\partial Y_i}{\partial Z_j} = \mathbf{1}_{i=j} - P_j
    $$
- **反向传播公式**：
  改变一个输入 $Z_j$ 会同时影响所有类别的输出，根据链式法则需将上游传回的各通道梯度 $G_i$ 累加：
  $$
  \frac{\partial L}{\partial Z_j} = \sum_i G_i \cdot \frac{\partial Y_i}{\partial Z_j} = \sum_i G_i (\mathbf{1}_{i=j} - P_j) = G_j - P_j \sum_i G_i
  $$
  即：“某个类别的梯度 = 自身的上游梯度 $G_j$ - 自身的概率 $P_j \times$ 所有类别的上游梯度总和 $\sum G$”。  
  写成张量形式即为：
  $$
  \frac{\partial L}{\partial Z} = G - P \odot \operatorname{broadcast}\left(\sum_{\text{类别轴}} G\right)
  $$
  对应到代码：$G$ 为 `out_grad`，前向输出 `node` 就是 $\log P$（直接用 `softmax = exp(node)` 零成本还原 $P$），$\sum G$ 为 `grad_sum`。

```python
class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        # 前向按防溢出公式计算：Z - (M + log(sum(exp(Z - M))))
        max_z = array_api.max(Z, axis=-1, keepdims=True)
        exp_z = array_api.exp(Z - max_z)
        sum_exp = array_api.sum(exp_z, axis=-1, keepdims=True)
        return Z - (max_z + array_api.log(sum_exp))

    def gradient(self, out_grad: Tensor, node: Tensor):
        # 1. 技巧：前向输出 node 本身就是 log(P)，取指数直接得到 Softmax 概率 P，无需重复计算
        softmax = exp(node)

        # 2. 沿类别轴求和 sum(G)，并将维度恢复为 (..., 1) 后广播为原形状
        grad_sum = out_grad.sum(axes=-1).reshape(
            out_grad.shape[:-1] + (1,)
        )
        grad_sum = grad_sum.broadcast_to(out_grad.shape)

        # 3. 严格对应反向公式：G - P * sum(G)
        return out_grad - softmax * grad_sum
```

###### 状态与模式切换层：`BatchNorm1d`
BatchNorm 沿 Batch 维度把特征拉回均值为 0、方差为 1 的分布，防止深层网络数值爆炸或消失。
写这一层有两个关键细节：
1. **训练与测试区别对待**：
   - 训练时：当前 batch 有足够多的样本，直接算当前 batch 的均值和方差，同时用动量把它们累积到全局均值 `running_mean` 和全局方差 `running_var` 里面；
   - 测试时：可能一次只有一个样本进来，没法算 batch 均值方差，必须直接拿训练期间累积好的全局 `running_mean` 和 `running_var` 来用。
2. **切断计算图（核心避坑）**：
   `running_mean` 和 `running_var` 只是用来记录全局统计量的，**不需要对它们求梯度**。更新它们时，必须写 `mean.data` 取出底层数值，千万不能直接用 `mean`。如果直接用 `mean`，就会把更新公式也建进计算图里，每个 batch 都会越积越大，跑几个 batch 内存就爆了。

```python
class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        # 可学习参数：缩放 weight (gamma) 和平移 bias (beta)，需要优化器更新，包进 Parameter
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        # 全局统计量：普通 Tensor，不需要求梯度更新
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            # 1. 训练时：算当前 batch 的均值和方差（沿 batch 维度 axis=0 求和）
            mean = ops.summation(x, axes=(0,)) / x.shape[0]
            mean_batch = mean.reshape((1, x.shape[1])).broadcast_to(x.shape)
            var = ops.summation((x - mean_batch) ** 2, axes=(0,)) / x.shape[0]
            var_batch = var.reshape((1, x.shape[1])).broadcast_to(x.shape)

            # 2. 避坑点：更新全局均值方差必须加 .data！切断计算图，防止内存泄漏
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data

            x_hat = (x - mean_batch) / ((var_batch + self.eps) ** 0.5)
        else:
            # 3. 测试时：直接用累积的全局均值和方差，不算当前 batch
            mean = self.running_mean.reshape((1, x.shape[1])).broadcast_to(x.shape)
            var = self.running_var.reshape((1, x.shape[1])).broadcast_to(x.shape)
            x_hat = (x - mean) / ((var + self.eps) ** 0.5)

        # 4. 仿射变换：y = weight * x_hat + bias
        w = self.weight.reshape((1, x.shape[1])).broadcast_to(x.shape)
        b = self.bias.reshape((1, x.shape[1])).broadcast_to(x.shape)
        return w * x_hat + b
```

