从 HW1 开始，我们就要正式从零构建这门课的核心产物——一个名为 needle（Necessary Elements of Deep Learning）的迷你深度学习框架了。

在 HW0 中，我们虽然也跑通了 MNIST 手写数字识别模型的训练，但当时所有的反向传播公式全靠自己把求导公式硬编码写进代码里。这种方式只要网络层数稍微一多、或者换个损失函数，就得改代码，根本没法扩展。

因此，HW1 的核心是实现深度学习的灵魂：**计算图（Computational Graph）与反向模式自动微分（Reverse-mode Auto-Differentiation）**。我们要让用户只需像写普通数学表达式一样定义前向计算，而反向传播的梯度全部由框架底层自动推导生成。

以下内容是对于 hw1 要接触到的一些类的简要解释，来自翻译版的课程 notebook：

- `Value`：计算图中计算出的值，既可以是对其他 `Value` 对象应用某种操作后的输出，也可以是常数（叶子节点）`Value` 对象。我们在这里使用一个通用基类（之后会专门派生出如 Tensor 等类），以便在后续版本的 needle 中支持其他数据结构；但目前你主要通过它的子类 `Tensor`（见下文）与其进行交互。
- `Op`：计算图中的操作符（算子）。算子需要通过 `compute()` 方法定义其“前向传播”（即如何在底层数据对象上计算该操作），并通过 `gradient()` 方法定义其“反向传播”（定义如何与传入的上游输出梯度相乘）。编写此类算子的细节将在下文详细说明。
- `Tensor`：这是 `Value` 的子类，对应实际的张量输出，即计算图中的多维数组。你在本次作业（以及后续大部分作业）中的所有代码都将使用这个 `Value` 的子类，而不是上述通用基类。我们提供了一些便捷功能（例如运算符重载），使你可以使用常规的 Python 语法操作张量，但在你实现相应的算子之前，这些功能尚无法正常工作。
- `TensorOp`：这是 `Op` 的子类，专门用于返回 Tensor 的算子。你在本次作业中实现的所有算子都属于该类型。


#### 1. 实现前向计算

实现前向计算的各个算子。在当前阶段，前向计算的输入都是 NDArray 对象（即 numpy 的数组），所以我们只需要使用 numpy 封装好的函数去对这些数组完成计算就可以了。这些 TensorOp 子类的 `compute()` 方法传入的是 NDArray 对象，返回的也是 NDArray 对象，后续在 hw3 中我们才会实现自己的类去替换掉 NDArray。

这里基本就是调库，唯独有一个小坑：NumPy 的 `np.transpose` 如果不传参数默认会反转全部维度（$N$ 维倒序），但 needle 的规范明确要求“默认只交换最后两个轴”（即 `(-1, -2)`），所以用 `swapaxes(a, *(self.axes or (-1, -2)))` 最稳妥。这里 a 就是传入的 NDArray，(self.axes or (-1, -2)) 是没传参时默认取最后两个轴，前面的 `*` 则是将元组解包成两个独立参数传给 `swapaxes`。

#### 2. 实现反向传播

这里就是实现各个 TensorOp 子类的 `gradient(out_grad, node)` 方法。

首先，我们得先搞明白计算图是什么：
* **计算图的节点全部都是 `Tensor` 对象**（比如输入 $x, y$ 和输出 $z$）
* **算子（`Op`）连接节点的边**。每次运算产生一个新 `Tensor` 时，它就会在内部记录两件事：`node.inputs`（我是从哪些前置 Tensor 算来的）以及 `node.op`（是用哪个算子算出来的）。

搞懂了这个结构，`gradient(out_grad, node)` 的参数就完全清晰了：
* **`node`**：代表**当前算子在前向计算时生成的那个输出 `Tensor` 节点本身**（例如 $z = x \cdot y$ 中的 $z$）。算子本身是个没有记忆的单例工具类，把它传进来，是为了能通过 `lhs, rhs = node.inputs` 顺藤摸瓜找到当初参与计算的输入；
* **`out_grad`**：代表**下游所有分支反向流回当前节点 `node` 的梯度总和**（同样是 `Tensor`）。它把下游所有层的影响打包成一个数值，算子只需用它乘以局部导数，就能把误差分发给各个输入；
* **返回值**：同样必须是 `Tensor` 对象。这也意味着反向求梯度的过程，实际上是在计算图旁边长出了一批新的求导节点，从而自然支持高阶导数。

虽然作业里要求实现 12 个算子，但其实可以归纳为以下 4 种类型，为了简化篇幅，这里只对每一种类型取一个算子做分析：

##### 1. 逐元素数学运算（Element-wise Ops）
* **特点**：输入与输出维度完全一致，元素之间互不影响。直接套用初等微积分求导公式，再乘以 `out_grad`。
* **涵盖算子**：`PowerScalar`, `EWiseDiv`, `DivScalar`, `Negate`, `Exp`, `Log`, `EWisePow`。
* **代表算子（EWiseDiv）**：$z = x / y$
  * 对 $x$ 求偏导：$\frac{\partial z}{\partial x} = \frac{1}{y} \implies \text{grad}_x = \text{out\_grad} / y$
  * 对 $y$ 求偏导：$\frac{\partial z}{\partial y} = -\frac{x}{y^2} \implies \text{grad}_y = -\text{out\_grad} \cdot x / y^2$
  * 代码直接返回：`out_grad / y, -out_grad * x / (y ** 2)`。

##### 2. 纯形状变换（Shape Transformation Ops）
* **特点**：元素数值本身完全没变，只是组织排布方式变了。输出与输入元素是一对一映射（偏导为 1），反向传播只需要**把梯度的形状“原路变换回去”**，这个很好理解，如果说当前算子的输出对于模型权重参数的贡献或者说影响是当前的形状，那么在形状变换之前的这个算子的输入对 loss 的贡献或者说影响当然就是它原本的形状了。
* **涵盖算子**：`Reshape`, `Transpose`。
* **代表算子（Reshape）**：$z = \text{reshape}(x, \text{shape})$
  * 反向时直接将 `out_grad` 变回输入的原始形状：
    `return reshape(out_grad, node.inputs[0].shape)`。
  * `Transpose` 同理，轴怎么交换过去的，反向就再用相同的 `axes` 交换回来。

##### 3. 维度广播与聚合（Broadcast 与 Sum 对偶算子）
* **特点**：这两者在数学和计算图上互为逆操作（对偶关系）：
  * **Broadcast 的反向是 Sum**：前向把某个轴复制了多份（广播），反向时根据链式法则多元求和，这些复制出来的分支梯度必须沿着广播的轴**全部累加回去**，这也很好理解，一个轴被复制了多份，它的贡献显然就是多倍的。
  * **Sum 的反向是 Broadcast**：前向把多个元素加成了一个标量，反向时这个标量的梯度必须原封不动地**广播发回给每个参与相加的输入元素**。
* **代表算子（BroadcastTo）**：$z = \text{broadcast\_to}(x, \text{shape})$
  * 比如 $x$ 形状为 `(3, 1)`，广播成 `(3, 4)`。反向时 `out_grad` 是 `(3, 4)`，需要沿着被拉长的第 1 维求和，将梯度还原回 `(3, 1)`：
    `return summation(out_grad, axes=(1,))`。
  * 实现时只需找出哪些维度从 1 变成了更大值（或者前面多出来的维度），沿这些维度调用 `summation` 即可。

##### 4. 矩阵乘法（Matrix Multiplication）
* **特点**：涉及行列内积与维度的收缩，是最重要的一个算子。
* **代表算子（MatMul）**：前向传播为 $Z = X Y$
  * 假设 $X$ 形状为 $(m, n)$，$Y$ 形状为 $(n, k)$，则输出 $Z$ 与上游传回的 `out_grad`（即 $\frac{\partial \ell}{\partial Z}$）形状均为 $(m, k)$。
  * **元素级推导**：
    矩阵乘法展开为元素形式：
    $$
    Z_{ij} = \sum_{p=1}^n X_{ip} \cdot Y_{pj}
    $$
    根据多元链式法则求 $\frac{\partial \ell}{\partial X_{ab}}$：
    $$
    \frac{\partial \ell}{\partial X_{ab}} = \sum_{i, j} \frac{\partial \ell}{\partial Z_{ij}} \cdot \frac{\partial Z_{ij}}{\partial X_{ab}}
    $$
    在 $Z_{ij} = \sum_p X_{ip} Y_{pj}$ 中，只有当行号 $i = a$ 且求和下标 $p = b$ 时，$X_{ab}$ 才会出现（导数为 $Y_{bj}$），其余情况导数均为 0。代入得：
    $$
    \frac{\partial \ell}{\partial X_{ab}} = \sum_j \frac{\partial \ell}{\partial Z_{aj}} \cdot Y_{bj} = \sum_j \left(\frac{\partial \ell}{\partial Z}\right)_{aj} (Y^T)_{jb}
    $$
    这正好是矩阵 $\frac{\partial \ell}{\partial Z}$ 的第 $a$ 行与矩阵 $Y^T$ 的第 $b$ 列做内积！写成矩阵形式就是：
    $$
    \frac{\partial \ell}{\partial X} = \frac{\partial \ell}{\partial Z} \cdot Y^T = \text{out\_grad} \cdot Y^T
    $$
    同理，对 $Y$ 求偏导：
    $$
    \frac{\partial \ell}{\partial Y_{ab}} = \sum_i X_{ia} \cdot \frac{\partial \ell}{\partial Z_{ib}} = \sum_i (X^T)_{ai} \left(\frac{\partial \ell}{\partial Z}\right)_{ib} \implies \frac{\partial \ell}{\partial Y} = X^T \cdot \text{out\_grad}
    $$
  * **维度验证**：
    * $\text{grad}_X = \text{out\_grad}_{(m, k)} \cdot Y^T_{(k, n)} \longrightarrow (m, n)$，与 $X$ 形状严格对齐；
    * $\text{grad}_Y = X^T_{(n, m)} \cdot \text{out\_grad}_{(m, k)} \longrightarrow (n, k)$，与 $Y$ 形状严格对齐。
  * **代码实现**：
    ```python
    return matmul(out_grad, transpose(Y)), matmul(transpose(X), out_grad)
    ```

#### 3. 拓扑排序

拓扑排序的实现就是一个后序 DFS，非常简单，这边的重点是要搞清楚为什么需要它：

反向传播初始时我们手里只有一个 loss。在前项传播中，如果节点 $A$ 依赖节点 $B$，拓扑排序保证 $B$ 永远排在 $A$ 前面。这样在反向传播时按照逆拓扑序直接倒序（`reversed`）遍历，就能保证下游产物节点的梯度一定在上游节点之前全部算完，从而确保多分支汇总到某个节点时的梯度全部到齐。

#### 4. 实现反向模式自动微分

核心目标：**计算出计算图中所有输入节点（特别是模型权重参数）对最终 loss 的导数（梯度）。**

##### 怎么被调用？在哪里被调用？
用户训练时直接调用 `loss.backward()`（位于 `autograd.py` 的 `Tensor.backward`）。
因为 $\frac{\partial \ell}{\partial \ell} = 1$，默认生成全 1 的张量作为初始梯度，然后调用我们在这里实现的函数：
```python
compute_gradient_of_variables(self, out_grad)
```

##### 函数实现逻辑
逻辑非常简单，就是顺着逆拓扑序倒着走一遍：

1. **维护梯度字典**：
   维护字典 `node_to_output_grads_list = {output_tensor: [out_grad]}`，键是节点，值是该节点接收到的梯度列表（暂存用）。
2. **按逆拓扑序遍历**：
   ```python
   for node in reverse_topo_order:
       # 1. 把当前节点暂存的所有梯度累加起来，赋给 node.grad
       node.grad = sum_node_list(node_to_output_grads_list[node])

       # 2. 如果是叶子节点（即最初的输入，权重 W 或输入 X，这些节点没有输入了），跳过
       if node.is_leaf():
           continue

       # 3. 算出当前算子对各个输入节点的梯度
       input_grads = node.op.gradient_as_tuple(node.grad, node)

       # 4. 把算好的梯度 append 到各输入节点的字典列表中暂存
       for input_node, input_grad in zip(node.inputs, input_grads):
           if input_node not in node_to_output_grads_list:
               node_to_output_grads_list[input_node] = []
           node_to_output_grads_list[input_node].append(input_grad)
   ```
跑完后，所有叶子节点（权重）都存好了 `W.grad`，优化器就可以直接拿去更新参数了。

#### 5. Softmax Loss

这里可以回读 hw0 笔记中的 3. 实现 Softmax Loss 函数，这个函数的功能就是衡量我们当前模型在评测集上预测的结果 z(logits)与 y(label) 差了多少然后对预测失败的做惩罚。但是这里的实现和 hw0 中有两个核心不同点：

##### 1. 角色不同：纯数值计算 vs 前向计算终点（建图）
- **hw0**：直接用 `np.log`、`np.exp` 做数值计算，算完就结束了。
- **hw1**：Softmax Loss 是模型**前向计算的终点**，也是后续**反向传播求导的起点**。为了能直接调用 `loss.backward()` 自动求梯度，输入与输出必须是 `Tensor`，中间也必须全程用 Needle 算子（调 numpy 会掐断计算图）。

##### 2. 提取真实类别预测值 $z_y$ 的方式不同
- **hw0**：传入的 `y` 是类别下标数组，NumPy 支持高级切片语法 `Z[np.arange(m), y]` 直接把得分抠出来。
- **hw1**：Needle 目前还没实现高级切片算子，所以传入的是 One-Hot 编码矩阵 `y_one_hot`，改为用**逐元素相乘 + 沿类别轴求和**来提取目标类别得分：
  ```python
  Z_y = ndl.summation(Z * y_one_hot, axes=(1,))
  ```

##### 计算公式
单样本的 Softmax Loss 公式为：
$$
\ell_{\mathrm{softmax}}(z, y) = \log\sum_{i=1}^k \exp z_i - z_y
$$
整个批次的平均损失则是把 batch 内每个样本的损失累加起来，除以样本总数 $m$（即 `Z.shape[0]`）。

##### 代码实现
```python
def softmax_loss(Z, y_one_hot):
    # 1. 计算 log-sum-exp 项
    exp_Z = ndl.exp(Z)
    sum_exp = ndl.summation(exp_Z, axes=(1,))
    log_sum_exp = ndl.log(sum_exp)

    # 2. 提取真实类别对应的得分 z_y
    Z_y = ndl.summation(Z * y_one_hot, axes=(1,))

    # 3. 计算每个样本损失并对 batch 求均值
    return ndl.summation(log_sum_exp - Z_y) / Z.shape[0]
```

#### 6. 为 needle 实现两层神经网络的随机梯度下降

这里可以对照 hw0 笔记中的 5. 两层神经网络的随机梯度下降。
在 hw0 里，我们为了训练两层网络，在代码里手写了一大堆矩阵转置和乘法去传导误差。而在 hw1 里，前面的 1 ~ 5 题已经把自动微分引擎和各种基础算子全部搭建好了，现在只要前向搭好模型，调一句 `loss.backward()`，引擎就会自动把所有参数的梯度全算好，我们只管拿去更新就行。

##### 1. 补齐 ReLU 算子
在写训练循环之前，先在 `ops_mathematic.py` 里补上非线性激活函数 ReLU：
- **前向计算**：`array_api.maximum(a, 0)`，负数直接归零。
- **反向传播**：
  ```python
  return out_grad * Tensor(node.realize_cached_data() > 0, requires_grad=False)
  ```
  ReLU 的导数很简单：输入大于 0 导数就是 1，小于等于 0 导数就是 0。
  所以反向传回来的梯度，只要看刚才前向算出来的数值是不是正数：是正数就保留原样，负数直接抹成 0。
  代码里调 `.realize_cached_data() > 0`，就是直接把前向存下的实际数值拿出来比一下，大于 0 的位置变成 1，小于等于 0 的变成 0，再乘给上游梯度。

##### 2. 训练循环 `nn_epoch` 实现
每个 batch 的训练流程非常标准化：
```python
def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    num_examples = X.shape[0]
    num_classes = W2.shape[1]

    for start in range(0, num_examples, batch):
        end = min(start + batch, X.shape[0])
        # 1. 把输入数据打包成 Tensor（输入数据不需要算梯度）
        Xb = ndl.Tensor(X[start:end], requires_grad=False)
        yb = y[start:end]
        yb_one_hot = np.zeros((yb.shape[0], num_classes))
        yb_one_hot[np.arange(yb.size), yb] = 1
        yb_tensor = ndl.Tensor(yb_one_hot, requires_grad=False)

        # 2. 前向传播：两层网络 ReLU(X @ W1) @ W2
        logits = ndl.relu(Xb @ W1) @ W2

        # 3. 算损失
        loss = softmax_loss(logits, yb_tensor)

        # 4. 一键反向传播：自动算出 W1.grad 和 W2.grad
        loss.backward()

        # 5. 更新权重
        W1 = ndl.Tensor(W1.numpy() - lr * W1.grad.numpy())
        W2 = ndl.Tensor(W2.numpy() - lr * W2.grad.numpy())

    return W1, W2
```

##### 3. 踩坑点：为什么更新权重非得用 `.numpy()` 重新包一下？
在 PyTorch 里我们更新参数一般用优化器 `optimizer.step()`，或者包在 `with torch.no_grad():` 里。
但我们自己写的 needle 目前还没有写优化器，也不支持原地修改（in-place 操作）。
如果直接写 `W1 = W1 - lr * W1.grad`，needle 会以为**这个减法也是网络模型的一部分**，把这步减法也悄悄建成了计算图节点。这样每训一个 batch，计算图就越来越大，跑几个 batch 内存就直接被撑爆了。
所以必须先用 `.numpy()` 把纯数值抠出来算减法，减完之后再用 `Tensor(...)` 重新包起来。这样它就是一个干干净净的初始参数，不会拖家带口带着上一个 batch 的计算图。



