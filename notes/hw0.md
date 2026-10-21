这是 CMU10-714 的开头，内容是在不使用任何深度学习库的前提下，使用 python 和 C++ 撰写 softmax regression，two-layer neural network 等基础算法，并将这些算法的实现拼接成一个基于有名的 MNIST 数据集的手写数字识别模型训练框架，内容谈不上很难，但是对于我这个完全不了解 machine learning 同时也已经很久没有写过 python 的小白来说，还是废了一些功夫，此时不得不庆幸今天有 ai 的存在，否则恐怕会非常艰难。

#### 1. 配好环境，撰写一个 a + b 来做测试
虽然我不咋写 py 了，不过好在对于 python 的虚拟环境还算熟悉，所以算是没有太多障碍地配好了这个环境。
这里有一个小坑是 notebook 中只写了要装 pybind11 和 numdifftools，这是因为其他的包在 Colab 环境中已经存在了，但是本地新建的虚拟环境中什么也没有，所以还要装 numpy，pytest 和 ipykernel。
然后对我来说比较陌生的就是 jupyter notebook 需要配置 kernel（选择虚拟环境）这个需要注意一下

```bash
pip install numpy pytest pybind11 numdifftools ipykernel
```

#### 2. 加载 MNIST 数据集
MNIST 是一个公开的用于手写数字识别的训练集，包括图片集和标签集，这里需要简单了解其数据的结构或者说格式：

MNIST 图片集的开头 16 个字节是 HEADER，包括  magic, num_images, rows, cols 4 个 uint32 数，其中 magic 是一个标识这是 MNIST 数据集的魔数，num_images 表示图片的数量，rows 和 cols 表示每张图的长宽分别是多少个像素。后续的数据全部都是像素，每个像素使用一个无符号单字节数来表示。

接下来是实现思路：
1. MNIST 数据集是以 gzip 压缩的数据集，所以我们要使用 gzip 库的 gzip.open()
2. MNIST 数据集的 HEADER 是按照大端序存储的。这里复习一下计组，大端序指的是数据的高位在前，即数据的开头放在地址的低位，比如说以一个占据地址段为 0x1000 ~ 0x1003 的 32 位整型数 0x12345678 为例（每个数都占据 4 位，所以每两个数占据一个字节），大端序指的是起点的 0x12 位于 0x1000 这个地址，与之相对的小端序指的就是 0x12 位于 0x1003 这个地址。目前主流的 x86/arm cpu 都是小端序，所以我们需要使用 struct.unpack(">IIII", f.read(16)) 来解包前 16 个字节的二进制流为 4 个四字节整型数，其中 > 表示大端序，IIII 表示 4 个 unsigned int 整型。
3. 后续就是正常读数据为 numpy 数组
4. 除以 uint8 的上限 255.0 得到一个归一化的浮点数


MNIST 标签集开头 8 个字节是 HEADER，包括 magic 和 num_labels，其中的数据是每张 image 对应的正确数字，加载方法和图片集相差不大，这里不多做赘述

#### 3. 实现 Softmax Loss 函数
到这里就算是正式开始学习了，Softmax Loss 其实是两个部分，Softmax 和 Loss(也称交叉熵损失 Cross-Entropy Loss)，这个函数的功能就是衡量我们当前模型在评测集上预测的结果 z(logits)与 y(label) 差了多少然后对预测失败的做惩罚，函数的参数以及返回值都是预先定义好的，虽然 python 本身的语义信息很弱，但是这个实验提供了 docstring（就是函数定义下的注释）， docstring 的特殊之处在于它被视为 python 对象的内置属性，能够被 ide/vscode 扩展索引作为 lint 信息。

我们首先来看看 Softmax Loss 的公式：
$$
\ell_{\mathrm{softmax}}(z, y) = \log\sum_{i=1}^k \exp z_i - z_y.
$$
这个公式乍一看肯定是不明所以的，因为它是防止数值溢出的一个变换公式，我们先看 Softmax 的公式：
$$
\operatorname{softmax}(z)_i
= \frac{\exp(z_i)}{\sum_j \exp(z_j)}
$$
这个公式就一眼可以看出来作用是将 logits 转换成概率分布，即模型推测出来的对于 0 ~ 9 这 10 个数字的概率分布。

接下来再来看 Loss，前面说过 Loss 的功能是惩罚预测错误的，预测完全正确的就不惩罚，现在我们又有概率分布了，通过一个 -log() 就可以实现对于正确项的预测概率极低的样本做严重惩罚。于是得到公式：
$$
\ell
= -\log\left(
\frac{\exp(z_y)}{\sum_i \exp(z_i)}
\right)
$$
根据对数的除法规则：
$$
\ell = \log\left(\sum_i \exp(z_i)\right) - \log(\exp(z_y))
$$
又因为：
$$
\log(\exp(z_y)) = z_y
$$
得到最终 Softmax Loss 的公式：
$$
\ell_{\mathrm{softmax}}(z, y) = \log\sum_{i=1}^k \exp z_i - z_y.
$$

接下来我们来看代码实现，顺着公式的推导写即可，但有几个细节需要注意：
1. 首先，`np.exp()` 这种数学函数都是逐元素（Element-wise）计算的，直接作用于整个矩阵。
2. 其次，$Z$ 是一个形状为 `(batch_size, num_classes)` 的二维数组，表示模型对一个 batch 内所有图片的原始打分（Logits）。因此计算 $\log\left(\sum_i \exp(z_i)\right)$ 时，我们需要沿第二维（`axis=1`）求和，因为每一行才代表一张图片对各个类别的打分，求和后会得到一个形状为 `(batch_size,)` 的一维数组。
3. 然后是提取每个样本对应的正确项得分 $Z_y$：也就是在第 $i$ 行挑出第 $y[i]$ 列的那个打分。在 NumPy 中可以用向量化写法 `Z[np.arange(batch_size), y]` 一次性取出（相当于免去了写 `for` 循环逐行提取 `Z[i, y[i]]`）。
4. 两者相减后，对整个一维数组取 `np.mean()` 求平均损失即可。

#### 4. 实现基于 Softmax Loss 的随机梯度下降
这一节 notebook 写的相当吓人，我对此强烈谴责，因为实际上没那么复杂。

先说什么是**随机梯度下降（SGD）**以及 **Epoch**：
假设我们要训练模型，MNIST 训练集一共有 60000 张手写数字图片。如果一次性把全部 60000 张图塞给模型算一遍总误差再更新参数，这叫全量梯度下降（BGD），计算极其缓慢甚至爆内存。
所以实际做法是每次只抓一个 Batch（比如 100 张图）给模型做，算完这 100 张的误差就立刻更新一次参数 $\Theta$。
接着抓下一个 100 张……直到把这 60000 张图全部遍历完一遍，这就叫做完成了一个 **Epoch（一轮训练）**。

> 在阅读下面对于公式的分析之前，首先要注意这里的一个坑：Notebook 前后对于符号的命名变化：在上一节（题目 3）中，作者把模型输出的打分（Logits）记作了 $z$。
> 但到了这一节（题目 4），作者改用 $h = \Theta^T x$ 来代表打分（Logits），而把经过 Softmax 归一化后的**概率**记为了 $z$（即 $z = \operatorname{softmax}(h)$）。

接下来看 notebook 中的一堆数学公式：

1. $\Theta$ 是形状为 $(784, 10)$ 的模型权重参数，负责把每张图的 784 个像素映射到 10 个类别的打分。
2. $\operatorname*{minimize}_{\Theta} \; \frac{1}{m} \sum_{i=1}^m \ell_{\mathrm{softmax}}(\Theta^T x^{(i)}, y^{(i)})$：这是训练的目标，调整 $\Theta$，让这 $m$ 张图片的平均 Softmax Loss 降到最小。这里的 $\Theta^T x$ 就是打分向量 $h$。
3. $z = \frac{\exp(\Theta^T x)}{\mathbf{1}^T \exp(\Theta^T x)} \equiv \operatorname*{normalize}(\exp(\Theta^T x))$：分母的 $\mathbf{1}^T$ 其实只是全 1 向量做内积（也就是求和 $\sum$），$\operatorname*{normalize}$ 就是归一化的意思，整个公式其实就是最基础的 Softmax。

然后来看全新的此前没有接触过的单张图片梯度公式：
$$
\nabla_\Theta \ell_{\mathrm{softmax}}(\Theta^T x, y) = x (z - e_y)^T
$$
这里的 $\nabla_\Theta \ell_{\mathrm{softmax}}$ 指的是使用 Softmax Loss 对模型参数中的每个权重分别求偏导，得到一个与模型参数一样大的矩阵（形状为 $784 \times 10$），这个矩阵就是梯度，用于指导如何修改模型参数：
* $x$ 是单张图片的输入像素，列向量形状为 $(784, 1)$；
* $z$ 是模型预测的 10 个类别概率，列向量形状为 $(10, 1)$；
* $e_y$ 是标准答案的 One-Hot 向量（$10 \times 1$，只有真实类别那一项是 1，其余全为 0）；
* $(z - e_y)$ 是预测误差（预测概率减去满分 1），转置后 $(z - e_y)^T$ 是行向量 $(1, 10)$；
* 列向量乘行向量（外积），形状正好是 $(784, 1) \times (1, 10) = (784, 10)$，与模型权重 $\Theta$ 完全一致。

**利用微积分链式法则推导这个公式**

根据微积分的**链式求导法则**：
$$
\frac{\partial \ell}{\partial \Theta_{ij}} = \frac{\partial \ell}{\partial h_j} \cdot \frac{\partial h_j}{\partial \Theta_{ij}}
$$
（因为权重 $\Theta_{ij}$ 只负责第 $j$ 个类别的打分 $h_j$，对其他类别的打分没有任何影响，所以中间变量只经过 $h_j$）：

**1. 打分对权重求导：$\frac{\partial h_j}{\partial \Theta_{ij}}$**
第 $j$ 个类别的打分 $h_j$ 是所有像素与对应权重的乘积累加：
$$
h_j = \Theta_{1j} x_1 + \dots + \Theta_{ij} x_i + \dots + \Theta_{nj} x_n
$$
我们要对其中的某一个具体权重 $\Theta_{ij}$ 求偏导：
* 整个多项式里除了 $\Theta_{ij} x_i$ 这一项以外，其余所有项（比如 $\Theta_{1j} x_1$、$\Theta_{2j} x_2$ 等）都不包含 $\Theta_{ij}$，求导时全部被当作常数变成 0；
* 而 $\Theta_{ij} x_i$ 对 $\Theta_{ij}$ 求导，系数就是 $x_i$。
因此：
$$
\frac{\partial h_j}{\partial \Theta_{ij}} = x_i
$$
也就是说：**打分对权重的导数，就是输入的第 $i$ 个像素值**。

**2. 损失对打分求导：$\frac{\partial \ell}{\partial h_j}$**
回顾 Softmax Loss 的公式（这里打分用 $h$ 表示）：
$$
\ell = \log\left(\sum_{k=1}^K \exp(h_k)\right) - h_y
$$
我们对变量 $h_j$ 求偏导，分别看减号的前后两项：
* **前半项 $\log\left(\sum_{k=1}^K \exp(h_k)\right)$ 求偏导**：
  设里面的求和为 $S = \exp(h_1) + \dots + \exp(h_j) + \dots + \exp(h_K)$。
  根据复合函数对数求导法则 $(\log S)' = \frac{1}{S} \cdot S'$：
  * $S$ 里面只有 $\exp(h_j)$ 含有变量 $h_j$，它对 $h_j$ 的导数就是它自己 $\exp(h_j)$（其余项求导全为 0）；
  * 所以导数是 $\frac{1}{S} \cdot \exp(h_j) = \frac{\exp(h_j)}{\sum_{k=1}^K \exp(h_k)}$；
  * **看！这正好就是 Softmax 给第 $j$ 个类别的预测概率 $z_j$！**
* **后半项 $-h_y$ 求偏导**（真实类别是 $y$）：
  * 如果当前考察的类别 $j$ 刚好是真实类别（$j = y$），这一项就是 $-h_j$，导数是 $-1$；
  * 如果 $j$ 不是真实类别（$j \neq y$），这一项是 $-h_y$，根本不包含 $h_j$，当作常数求导为 $0$。
  * 我们可以用 One-Hot 标签 $(e_y)_j$ 统一写成 $-(e_y)_j$（只有在 $j = y$ 时为 $-1$，其余为 0）。

把前后两项合并：
$$
\frac{\partial \ell}{\partial h_j} = z_j - (e_y)_j
$$
这就是最关键的结论：**损失对打分的导数，直接等于“预测概率 减去 真实标签”的误差！**

**3. 两者相乘并写成矩阵形式：**
把上述两项代回链式法则：
$$
\frac{\partial \ell}{\partial \Theta_{ij}} = \frac{\partial \ell}{\partial h_j} \cdot \frac{\partial h_j}{\partial \Theta_{ij}} = x_i \cdot (z_j - (e_y)_j)
$$
整个梯度矩阵在第 $i$ 行、第 $j$ 列的值，恰好就是向量 $x$ 的第 $i$ 项与误差向量 $(z - e_y)$ 的第 $j$ 项相乘。
根据线性代数定义，列向量乘行向量生成的矩阵其 $(i, j)$ 元素正是 $x_i (z - e_y)_j$：
$$
\nabla_\Theta \ell = x (z - e_y)^T
$$
物理意义也很直观：某个像素越亮（$x_i$ 越大），且模型在这个类别上猜得越离谱（误差越大），该权重受罚调整的幅度就越大；纯黑像素（$x_i = 0$）则不承担任何调整责任。


**推广到 Batch**
$$
\nabla_\Theta \ell_{\mathrm{softmax}}(X \Theta, y) = \frac{1}{m} X^T (Z - I_y)
$$
在代码中我们要对一个 Batch 的 $m$ 张图片同时计算（例如 $m = 100$）。整个 Batch 的平均梯度是所有 $m$ 张图的梯度求和除以 $m$：
$$
\nabla_{\Theta} = \frac{1}{m} \sum_{s=1}^m x^{(s)} (z^{(s)} - e_{y^{(s)}})^T
$$
为了不用极慢的循环，我们需要构造矩阵乘法：

* **输入矩阵 $X$**：形状为 $(m, n) = (100, 784)$，每一行是一张图片 $(x^{(s)})^T$（形状 $1 \times 784$）；
* **转置矩阵 $X^T$**：形状为 $(n, m) = (784, 100)$，**每一列**是一张图片的列向量 $x^{(s)}$（形状 $784 \times 1$）：
  $$
  X^T = \begin{bmatrix} x^{(1)} & x^{(2)} & \cdots & x^{(m)} \end{bmatrix}_{(784 \times 100)}
  $$
* **预测概率矩阵 $Z$ 与 One-Hot 标签矩阵 $I_y$**：形状均为 $(m, k) = (100, 10)$；
* **误差矩阵 $(Z - I_y)$**：形状为 $(m, k) = (100, 10)$，**每一行**是一张图片的误差行向量 $(z^{(s)} - e_{y^{(s)}})^T$（形状 $1 \times 10$）：
  $$
  (Z - I_y) = \begin{bmatrix} (z^{(1)} - e_{y^{(1)}})^T \\ (z^{(2)} - e_{y^{(2)}})^T \\ \vdots \\ (z^{(m)} - e_{y^{(m)}})^T \end{bmatrix}_{(100 \times 10)}
  $$
* **做矩阵乘法 $X^T (Z - I_y)$**：
  $$(784 \times 100) \times (100 \times 10) \longrightarrow (784 \times 10)$$
  按照矩阵乘法“左矩阵各列 乘 右矩阵各行”的规则展开：
  $$
  X^T (Z - I_y) = \sum_{s=1}^m \underbrace{x^{(s)}_{(784 \times 1)} \cdot (z^{(s)} - e_{y^{(s)}})^T_{(1 \times 10)}}_{\text{单张图片的梯度矩阵 }(784 \times 10)}
  $$
  **做一次矩阵乘法，底层就自动完成了所有 $m$ 张图片梯度的外积计算与求和累加！**
* 前面再乘上 $\frac{1}{m}$，就得到了平均梯度矩阵 $\nabla_\Theta$，形状正好是 $(784, 10)$，与模型权重 $\Theta$ 严丝合缝匹配！这也解释了为什么 $X$ 必须转置（行存变列存），而误差矩阵 $(Z - I_y)$ 本身就在行上所以不需要转置。

---

懂了上面的推导，代码还是比较好写的，首先我们遍历 X 的每一行（也就是每一个 batch）对于每个 batch，我们使用模型权重计算这一批样本的 logits，然后使用 Softmax 将其转换为概率分布 Z。后面唯一值得注意的是并没有真正构造一个 one-hot 矩阵，而是采用 `probs[np.arange(yb.shape[0]), yb] -= 1` 这样变相的实现。

#### 5. 两层神经网络的随机梯度下降
这里将 4 中实现的随机梯度下降称呼为线性分类器的 SGD，是因为在 4 的实现中，所有的像素都是一次方的，属于线性组合。一般来说都认为线性函数的表达能力的有限的，至于为什么那就有待进一步学习了，我对此兴趣不大。总之，这里要实现的就是包含非线性激活函数 ReLU() 的两层神经网络的 SGD。

notebook 中以一个包含两个权重矩阵的神经网络为例，中间使用 ReLU() 处理输入经过一个权重矩阵的结果，后续的公式只需要关心反向传播更新公式（也就是梯度）的推导：

首先回顾题目 4 留下的通用法则：对于形如 $\text{输出} = \text{输入} \times \text{权重}$ 的矩阵乘法，权重的梯度永远遵守：
$$
\text{梯度} = \frac{1}{m} \times (\text{本层输入})^T \times (\text{本层输出处的误差})
$$

两层神经网络的前向传播分为两步：
1. 第一层：$Z_1 = \operatorname{ReLU}(X W_1)$
2. 第二层：$\text{Logits} = Z_1 W_2$

反向传播也就是把上面的通用法则套用两遍，中间穿插一次误差回传：

1. **算第二层的梯度 $\nabla_{W_2}$（完全就是题目 4 的翻版）**：
   * 第二层输出端的误差依然是预测概率减去真实标签：$G_2 = \operatorname{normalize}(\exp(Z_1 W_2)) - I_y$
   * 第二层的输入是 $Z_1$，套用模板直接写出：
     $$
     \nabla_{W_2} = \frac{1}{m} Z_1^T G_2
     $$
2. **将误差回传给第一层（穿透 $W_2$ 与 $\operatorname{ReLU}$）**：
   * $G_2$ 是最终打分处的误差（$\frac{\partial \ell}{\partial \text{Logits}}$），而 $G_1$ 就是**回传到第一层线性输出处的误差矩阵**（$\frac{\partial \ell}{\partial (X W_1)}$），代表每个隐藏神经元为最终错误背了多少锅。
   * 误差逆向穿透 $W_2$ 分摊到隐层：$G_2 W_2^T$；
   * 穿透 $\operatorname{ReLU}$ 门控：前向激活值 $Z_1 > 0$ 的神经元导数为 1（误差通过），$Z_1 \le 0$ 的导数为 0（误差截断清零）。
   * 由此得到第一层承受的误差 $G_1$：
     $$
     G_1 = \mathbf{1}\{Z_1 > 0\} \circ (G_2 W_2^T)
     $$
3. **算第一层的梯度 $\nabla_{W_1}$（再次套用模板）**：
   * 第一层的输入是原始图像 $X$，承受的误差是 $G_1$，套用模板：
     $$
     \nabla_{W_1} = \frac{1}{m} X^T G_1
     $$

代码实现上与题目 4 几乎一致，核心就多了两行误差回传与第一层梯度：

```python
# 误差回传：先反向穿过 W2，再经 ReLU 门控截断
G1 = (Z1 > 0) * (G2 @ W2.T)
# 套用通用法则计算 W1 梯度
grad_W1 = Xb.T @ G1 / Xb.shape[0]
```

#### 6. 使用 C++ 实现 Softmax Regression 以及 Pybind11

前面我们的实现都是直接使用了 python 的 numpy 库，这里用 C++ 再写一遍，但是并不要求写的很高效，只是单纯做个实现，所以并不难写，主要就是对照 python 去手写矩阵乘法。

但是经过测试就可以发现这个 C++ 版本比 Python 版本慢了 5 倍。
这是因为 numpy 这个库是经过高度优化的，充分调用了 CPU 的指令集，但是我们自己写的 C++  很朴素而且每个 batch 都在重复申请内存。这里不多做赘述了。

这里的重点是 Pybind11，重点得学习一下这个东西。

pybind11 的作用是让 Python 直接调用 C/C++。在深度学习框架中，底层算子通常用 C++/CUDA 实现，而上层模型和训练用 Python 调度，连接两者的桥梁就是 pybind11。

接下来来看 `src/simple_ml_ext.cpp` 末尾（88~108行）的这段代码：

```cpp
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp", 
        [](py::array_t<float, py::array::c_style> X, 
           py::array_t<unsigned char, py::array::c_style> y, 
           py::array_t<float, py::array::c_style> theta,
           float lr, int batch) {
        softmax_regression_epoch_cpp(
            static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0], X.request().shape[1], theta.request().shape[1],
            lr, batch
        );
    }, py::arg("X"), py::arg("y"), py::arg("theta"), py::arg("lr"), py::arg("batch"));
}
```

它的本质就是一个**参数适配器**。因为 Python 端调用只传了 5 个 NumPy 数组，而手写的 C++ 函数需要 8 个参数（3 个裸指针 + 矩阵长宽 $m, n, k$ + lr + batch）。

代码逻辑主要分为两层：

1. **模块入口 `PYBIND11_MODULE(simple_ml_ext, m)`**：
   定义模块名为 `simple_ml_ext`（编译成 `.so` 后，Python `import simple_ml_ext` 找的就是它）。`m` 是代表当前模块的句柄。

2. **函数注册 `m.def(...)`（由三大部分组成）**：
   * **第一部分（函数名）**：`"softmax_regression_epoch_cpp"`，即暴露给 Python 调用的函数名。
   * **第二部分（函数实现）**：`[] (...) { ... }` 现场写了一个 C++ Lambda 匿名函数（相当于 Python 的 `lambda`）。
     * **为什么需要它？** 因为 Python 端调用只传 5 个 NumPy 数组，而底层算子需要 8 个参数，所以用它当中介接盘。
     * **形参声明**：`py::array_t<float, py::array::c_style> X` 等，接收 Python 传来的 NumPy 数组，`c_style` 保证行优先连续存储。
     * **函数体拆包**：用 `X.request().ptr` 直接获取堆上的**真实裸指针地址**（零拷贝 Zero-Copy；原地修改 `theta` 也是直接写这块内存）；用 `X.request().shape` 提取长宽补齐 $m, n, k$，然后去调用底层真正的算子。
   * **第三部分（参数命名）**：末尾的一串 `py::arg("X"), py::arg("y")...`。
     * 按顺序给前面的形参“贴名字标签”。因为 C++ 编译成二进制（`.so`）后形参名字会被编译器抹除，加上它后 Python 才能支持关键字传参（如 `lr=0.1, batch=100`），并在 IDE 补全时显示清晰的参数名。

最后用 `Makefile` 里的命令编译：
```bash
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) src/simple_ml_ext.cpp -o src/simple_ml_ext.so
```
因为 pybind11 是**纯头文件库（header-only）**，所以不需要额外链接任何库。编译出的 `.so` 放在当前目录，Python 就能直接 `import simple_ml_ext` 调用了。






