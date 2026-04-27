# MatMul
>[MatMul - ONNX documentation](https://onnx.ai/onnx/operators/onnx__MatMul.html)

标准的矩阵乘法算子，支持二维矩阵乘法和批量矩阵乘法（广播机制）。
- input1：[..., i, k]
- input2：[..., k, j]
- output：[..., i, j]

... 表示任意数量的批量维度。

# Einsum 算子
>[torch.einsum - PyTorch 文档](https://docs.pytorch.ac.cn/docs/stable/generated/torch.einsum.html)

遵循爱因斯坦求和约定（Einstein summation convention）的张量运算算子，通过简洁的字符串表达式描述复杂的张量操作。

- equation 属性：描述张量操作的字符串
- 使用字母标记张量维度
- 重复字母表示在该维度上相乘
- 输出中省略的字母表示在该维度上求和

Einsum 算子支持多种计算规则。可以经过适当变换，最终分解为二维矩阵乘法，从而用 MatMul 算子替换。

# 替换方案

## ...ij, ...jc -> ...ic
矩阵乘法：可以直接用 MatMul 算子替换。

## ...ic, ...jc -> ...ij
矩阵乘法：转置第二个输入为 `...cj`，然后用 MatMul 算子替换。  
```python
Transpose(input2, perm=[..., -1, -2])
```

## ...nic, ...ncj -> ...nij
批量矩阵乘法：可以直接用 MatMul 算子替换。

## ...hic, ...hjc -> ...hij
批量矩阵乘法：转置第二个输入为 `...hjc`，然后用 MatMul 算子替换。  
```python
Transpose(input2, perm=[..., -1, -2])
```

## ...nij, ijc -> ...nic（复杂）
### 替换思路
... 计算前后一致，为了方便理解，将其当作 batch 维度。  
对 j 维度求和，保留 batch, n, i, c 维度。由于输入输出达到了 4 维，无法直接用 MatMul 算子计算。需要先将输入维度对齐到 3 维，经过 MatMul 计算后，再恢复 4 维。  
查看输入输出，i 维度在两个输入中都存在，且保留到输出。可以将第一个输入的 i 维度合并至 batch 维度，成为 `batch*i` 维度。同时将第二个输入的 i 维度也扩展成 `batch*i` 维度，输入输出就对齐到 3 维，可以用 MatMul 算子计算。由于 batch 计算前后不变，可以提出拆分出 batch 从而恢复 4 维。
- input1:  [batch, n, i, j] -> [batch*i, n, j]
- input2:  [i, j, c] -> [batch*i, j, c]
- output:  [batch*i, n, c] -> [batch, n, i, c]

### 具体替换步骤
1. 转置第一个输入，将 i 维度移到 n 前面，为后续合并做准备。
  ```python
  # input1: [batch, n, i, j] -> [batch, i, n, j]
  Transpose(input1, perm=[0, 2, 1, 3])
  ```

1. 合并第一个输入的前两个维度，将 batch 和 i 合并为新的批次维度。
  ```python
  # input1: [batch, i, n, j] -> [batch*i, n, j]
  Reshape(transposed, [batch*i, n, j])
  ```

1. 扩展第二个输入，将 i 维度扩展为 `batch*i` 维度。
  ```python
  # 添加批次维度
  # [i, j, c] -> [1, i, j, c]
  Unsqueeze(input2, axes=[0])
  # 每个 i 对应一个 [j, c] 矩阵，需要复制 batch 次以匹配新的批次维度
  # [1, i, j, c] -> [batch, i, j, c]
  Tile(unsqueezed, [batch, 1, 1, 1])
  # 合并维度
  # [batch, i, n, j] -> [batch*i, n, j]
  Reshape(tiled, [batch*i, j, c])
  ```

1. 批量矩阵乘法
  ```python
  # [batch*i, n, j] @ [batch*i, j, c] = [batch*i, n, c]
  MatMul(reshaped1, reshaped2)
  ```

1. 恢复维度
  ```python
  # output: [batch*i, n, c] -> [batch, i, n, c]
  Reshape(matmul, [batch, i, n, c])
  ```

1. 转置恢复顺序
  ```python
  # output: [batch, i, n, c] -> [batch, n, i, c]
  Transpose(reshaped, [0, 2, 1, 3])
  ```

完整流程图：
```
输入1 [B,N,I,J]          输入2 [I,J,C]
    ↓                          ↓
Transpose [B,I,N,J]      Unsqueeze [1,I,J,C]
    ↓                          ↓
Reshape [B*I,N,J]        Tile [B,I,J,C]
    |                          ↓
    |                    Reshape [B*I,J,C]
    |                          |
    └-─→ MatMul [B*I,N,C] ←────┘
                ↓
	     Reshape [B,I,N,C]
                ↓
         Transpose [B,N,I,C]
```

## aijk, aijh -> ajkh
批量矩阵乘法：转置两个输入，然后合并 aj 作为 batch 维度，就得到了熟悉的公式 (aj)ki, (aj)ih -> (aj)kh
```python
# input1: [a, i, j, k] -> [a, j, k, i]
Transpose(input1, perm=[0, 2, 3, 1])

# input2: [a, i, j, h] -> [a, j, i, h]
Transpose(input2, perm=[0, 2, 1, 3])

B = a * j
# input1: [a, j, k, i] -> [B, k, i]
Reshape(transposed_input1, [B, k, i])
# input2: [a, j, i, h] -> [B, i, h]
Reshape(transposed_input2, [B, i, h])

# [B, k, i] × [B, i, h] -> [B, k, h]
MatMul(reshaped_input1, reshaped_input2)

# [B, k, h] -> [a, j, k, h]
Reshape(matmul_output, [a, j, k, h])
```
