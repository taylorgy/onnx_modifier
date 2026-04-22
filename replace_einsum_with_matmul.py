"""
替换 Einsum 节点为 MatMul 节点

这个模块提供了将 Einsum 节点替换为 MatMul 节点的功能，支持多种模式。

1. ...ic, ...jc -> ...ij (矩阵乘法，需要转置第二个输入)
2. ...ij, ...jc -> ...ic (矩阵乘法)
3. ...nij, ijc -> ...nic (批量矩阵乘法)
4. ...hic,...hjc -> ...hij (批量矩阵乘法)
"""

import onnx
import numpy as np
from onnx import helper, numpy_helper, TensorProto

def get_tensor_shape(model, tensor_name):
    """
    获取张量的形状

    参数:
        model: ONNX 模型
        tensor_name: 张量名称

    返回:
        list: 张量形状，如果找不到则返回 None
    """
    # 从 value_info 中获取形状
    for vi in model.graph.value_info:
        if vi.name == tensor_name:
            return [d.dim_value for d in vi.type.tensor_type.shape.dim]

    # 从 graph.input 中获取形状
    for inp in model.graph.input:
        if inp.name == tensor_name:
            return [d.dim_value for d in inp.type.tensor_type.shape.dim]

    # 从 graph.output 中获取形状
    for out in model.graph.output:
        if out.name == tensor_name:
            return [d.dim_value for d in out.type.tensor_type.shape.dim]

    # 从 initializer 中获取形状
    for init in model.graph.initializer:
        if init.name == tensor_name:
            return list(init.dims)

    return None

def replace_einsum_with_matmul(model_path, output_path):
    """
    将 Einsum 节点替换为 MatMul 节点

    参数:
        model_path: 输入模型路径
        output_path: 输出模型路径

    返回:
        bool: 替换是否成功
    """
    try:
        # 加载 ONNX 模型
        model = onnx.load(model_path)

        # 找到所有 Einsum 节点
        einsum_nodes = [node for node in model.graph.node if node.op_type == 'Einsum']
        print(f'找到 {len(einsum_nodes)} 个 Einsum 节点')

        # 创建新节点列表
        new_nodes = []
        new_initializers = []
        replaced_count = 0

        for node in model.graph.node:
            if node.op_type == 'Einsum':
                # 获取 equation 属性
                equation = None
                for attr in node.attribute:
                    if attr.name == 'equation':
                        equation = attr.s.decode('utf-8')
                        break

                print(f'\n处理 Einsum 节点: {node.name}')
                print(f'  方程: {equation}')

                input1_name = node.input[0]
                input2_name = node.input[1]
                output_name = node.output[0]

                # 获取输入和输出的形状
                input1_shape = get_tensor_shape(model, input1_name)
                input2_shape = get_tensor_shape(model, input2_name)
                output_shape = get_tensor_shape(model, output_name)

                # 处理不同的 einsum 模式
                if '...ic, ...jc -> ...ij' in equation:
                    # 模式1: ...ic, ...jc -> ...ij
                    # 对应于矩阵乘法，需要转置第二个输入

                    # 创建转置后的输入2名称
                    input2_transposed_name = f'{input2_name}_transposed_{node.name}'

                    # 创建转置节点（交换最后两个维度）
                    if input2_shape and len(input2_shape) >= 2:
                        perm = list(range(len(input2_shape)))
                        perm[-2], perm[-1] = perm[-1], perm[-2]
                    else:
                        perm = [0, 1, 3, 2]  # 默认值

                    transpose_node = helper.make_node(
                        'Transpose',
                        inputs=[input2_name],
                        outputs=[input2_transposed_name],
                        name=f'{node.name}_transpose',
                        perm=perm
                    )
                    new_nodes.append(transpose_node)

                    # 创建 MatMul 节点
                    matmul_node = helper.make_node(
                        'MatMul',
                        inputs=[input1_name, input2_transposed_name],
                        outputs=[output_name],
                        name=f'{node.name}_matmul'
                    )
                    new_nodes.append(matmul_node)

                    replaced_count += 1
                    print(f'  已替换为 MatMul + Transpose')

                elif '...ij, ...jc -> ...ic' in equation:
                    # 模式2: ...ij, ...jc -> ...ic
                    # 对应于矩阵乘法

                    # 创建 MatMul 节点
                    matmul_node = helper.make_node(
                        'MatMul',
                        inputs=[input1_name, input2_name],
                        outputs=[output_name],
                        name=f'{node.name}_matmul'
                    )
                    new_nodes.append(matmul_node)

                    replaced_count += 1
                    print(f'  已替换为 MatMul')

                elif '...nij, ijc -> ...nic' in equation:
                    # 模式3: ...nij, ijc -> ...nic
                    # 批量矩阵乘法
                    # 输入1: [batch, n, i, j]
                    # 输入2: [i, j, c]
                    # 输出: [batch, n, i, c]
                    #
                    # 关键点：对 j 维度求和，输出保留 n、i 和 c 维度
                    # 正确的计算：
                    # 1. 输入1: [batch, n, i, j] -> [batch, i, n, j] (转置 n 和 i 维度)
                    # 2. 输入1: [batch, i, n, j] -> [batch*i, n, j] (reshape，合并 batch 和 i 维度)
                    # 3. 输入2: [i, j, c] -> [batch, i, j, c] (扩展 batch 维度，使用 Tile)
                    # 4. 输入2: [batch, i, j, c] -> [batch*i, j, c] (reshape，合并 batch 和 i 维度)
                    # 5. MatMul: [batch*i, n, j] x [batch*i, j, c] -> [batch*i, n, c]
                    # 6. 输出: [batch*i, n, c] -> [batch, i, n, c] (reshape)
                    # 7. 输出: [batch, i, n, c] -> [batch, n, i, c] (转置 i 和 n 维度)

                    if input1_shape and output_shape:
                        # 获取维度
                        batch = input1_shape[0]  # 2
                        n = input1_shape[1]  # 8
                        i = input1_shape[2]  # 230
                        j = input1_shape[3]  # 230
                        c = input2_shape[2]  # 32

                        # 步骤1: 输入1: [batch, n, i, j] -> [batch, i, n, j] (转置 n 和 i 维度)
                        input1_transposed_name = f'{input1_name}_transposed_{node.name}'
                        transpose1_node = helper.make_node(
                            'Transpose',
                            inputs=[input1_name],
                            outputs=[input1_transposed_name],
                            name=f'{node.name}_transpose1',
                            perm=[0, 2, 1, 3]  # [batch, n, i, j] -> [batch, i, n, j]
                        )
                        new_nodes.append(transpose1_node)

                        # 步骤2: 输入1: [batch, i, n, j] -> [batch*i, n, j] (reshape，合并 batch 和 i 维度)
                        input1_reshaped_shape = [batch * i, n, j]
                        input1_reshaped_name = f'{input1_name}_reshaped_{node.name}'

                        # 创建形状常量
                        shape_tensor1 = numpy_helper.from_array(
                            np.array(input1_reshaped_shape, dtype=np.int64),
                            name=f'{input1_reshaped_name}_shape'
                        )
                        new_initializers.append(shape_tensor1)

                        reshape1_node = helper.make_node(
                            'Reshape',
                            inputs=[input1_transposed_name, shape_tensor1.name],
                            outputs=[input1_reshaped_name],
                            name=f'{node.name}_reshape1'
                        )
                        new_nodes.append(reshape1_node)

                        # 步骤3: 输入2: [i, j, c] -> [batch, i, j, c] (扩展 batch 维度，使用 Tile)
                        # 首先Unsqueeze 添加 batch 维度
                        input2_unsqueeze_name = f'{input2_name}_unsqueeze_{node.name}'
                        # 创建 axes 常量
                        axes_tensor = numpy_helper.from_array(
                            np.array([0], dtype=np.int64),
                            name=f'{input2_unsqueeze_name}_axes'
                        )
                        new_initializers.append(axes_tensor)
                        unsqueeze_node = helper.make_node(
                            'Unsqueeze',
                            inputs=[input2_name, axes_tensor.name],
                            outputs=[input2_unsqueeze_name],
                            name=f'{node.name}_unsqueeze'
                        )
                        new_nodes.append(unsqueeze_node)

                        # 然后Tile 扩展 batch 维度
                        input2_tiled_name = f'{input2_name}_tiled_{node.name}'
                        # 创建 repeats 常量
                        repeats_tensor = numpy_helper.from_array(
                            np.array([batch, 1, 1, 1], dtype=np.int64),
                            name=f'{input2_tiled_name}_repeats'
                        )
                        new_initializers.append(repeats_tensor)

                        tile_node = helper.make_node(
                            'Tile',
                            inputs=[input2_unsqueeze_name, repeats_tensor.name],
                            outputs=[input2_tiled_name],
                            name=f'{node.name}_tile'
                        )
                        new_nodes.append(tile_node)

                        # 步骤4: 输入2: [batch, i, j, c] -> [batch*i, j, c] (reshape，合并 batch 和 i 维度)
                        input2_reshaped_shape = [batch * i, j, c]
                        input2_reshaped_name = f'{input2_name}_reshaped_{node.name}'

                        # 创建形状常量
                        shape_tensor2 = numpy_helper.from_array(
                            np.array(input2_reshaped_shape, dtype=np.int64),
                            name=f'{input2_reshaped_name}_shape'
                        )
                        new_initializers.append(shape_tensor2)

                        reshape2_node = helper.make_node(
                            'Reshape',
                            inputs=[input2_tiled_name, shape_tensor2.name],
                            outputs=[input2_reshaped_name],
                            name=f'{node.name}_reshape2'
                        )
                        new_nodes.append(reshape2_node)

                        # 步骤5: MatMul: [batch*i, n, j] x [batch*i, j, c] -> [batch*i, n, c]
                        matmul_name = f'{output_name}_matmul'
                        matmul_node = helper.make_node(
                            'MatMul',
                            inputs=[input1_reshaped_name, input2_reshaped_name],
                            outputs=[matmul_name],
                            name=f'{node.name}_matmul'
                        )
                        new_nodes.append(matmul_node)

                        # 步骤6: 输出: [batch*i, n, c] -> [batch, i, n, c] (reshape)
                        output_intermediate_shape = [batch, i, n, c]
                        output_intermediate_name = f'{output_name}_intermediate'

                        # 创建形状常量
                        shape_tensor3 = numpy_helper.from_array(
                            np.array(output_intermediate_shape, dtype=np.int64),
                            name=f'{output_intermediate_name}_shape'
                        )
                        new_initializers.append(shape_tensor3)

                        reshape3_node = helper.make_node(
                            'Reshape',
                            inputs=[matmul_name, shape_tensor3.name],
                            outputs=[output_intermediate_name],
                            name=f'{node.name}_reshape3'
                        )
                        new_nodes.append(reshape3_node)

                        # 步骤7: 输出: [batch, i, n, c] -> [batch, n, i, c] (转置 i 和 n 维度)
                        transpose2_node = helper.make_node(
                            'Transpose',
                            inputs=[output_intermediate_name],
                            outputs=[output_name],
                            name=f'{node.name}_transpose2',
                            perm=[0, 2, 1, 3]  # [batch, i, n, c] -> [batch, n, i, c]
                        )
                        new_nodes.append(transpose2_node)

                        replaced_count += 1
                        print(f'  已替换为 Transpose + Reshape + Unsqueeze + Tile + Reshape + MatMul + Reshape + Transpose')
                        print(f'    输入1: {input1_shape} -> [batch, i, n, j] -> {input1_reshaped_shape}')
                        print(f'    输入2: {input2_shape} -> [batch, i, j, c] -> {input2_reshaped_shape}')
                        print(f'    MatMul: {input1_reshaped_shape} x {input2_reshaped_shape} -> [batch*i, n, c]')
                        print(f'    输出: [batch*i, n, c] -> {output_intermediate_shape} -> {output_shape}')
                    else:
                        print(f'  警告: 无法获取形状信息，保持原样')
                        new_nodes.append(node)

                elif '...hic,...hjc -> ...hij' in equation:
                    # 模式4: ...hic,...hjc -> ...hij
                    # 批量矩阵乘法

                    # 转置输入2
                    input2_transposed_name = f'{input2_name}_transposed_{node.name}'

                    if input2_shape and len(input2_shape) >= 2:
                        perm = list(range(len(input2_shape)))
                        perm[-2], perm[-1] = perm[-1], perm[-2]
                    else:
                        perm = [0, 1, 2, 4, 3]  # 默认值

                    transpose_node = helper.make_node(
                        'Transpose',
                        inputs=[input2_name],
                        outputs=[input2_transposed_name],
                        name=f'{node.name}_transpose',
                        perm=perm
                    )
                    new_nodes.append(transpose_node)

                    # MatMul
                    matmul_node = helper.make_node(
                        'MatMul',
                        inputs=[input1_name, input2_transposed_name],
                        outputs=[output_name],
                        name=f'{node.name}_matmul'
                    )
                    new_nodes.append(matmul_node)

                    replaced_count += 1
                    print(f'  已替换为 Transpose + MatMul')

                else:
                    # 不支持的 einsum 模式，保持原样
                    print(f'  警告: 不支持的 einsum 模式，保持原样')
                    new_nodes.append(node)
            else:
                # 非 Einsum 节点，保持原样
                new_nodes.append(node)

        # 替换图中的节点
        model.graph.ClearField('node')
        model.graph.node.extend(new_nodes)

        # 添加新的 initializers
        model.graph.initializer.extend(new_initializers)

        # 保存模型
        onnx.save(model, output_path)
        print(f'\n替换了 {replaced_count} 个 Einsum 节点')
        print(f'模型已保存到: {output_path}')

        # 验证模型
        try:
            onnx.checker.check_model(model)
            print('模型验证通过')
            return True
        except Exception as e:
            print(f'模型验证失败: {e}')
            import traceback
            traceback.print_exc()
            return False
    except Exception as e:
        print(f'Einsum 节点替换失败: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("用法: python replace_einsum.py <输入模型路径> <输出模型路径>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    replace_einsum_with_matmul(input_path, output_path)
