"""
修复 LayerNormalization 节点

这个模块提供了修复 LayerNormalization 节点的功能，
为只有 2 个输入的 LayerNormalization 节点添加全零的 bias 输入，
转换为标准的 3 输入形式。
"""

import onnx
import numpy as np
from onnx import numpy_helper

def fix_layernorm_nodes(model_path, output_path):
    """
    修复 LayerNormalization 节点，确保每个节点有 3 个输入

    参数:
        model_path: 输入模型路径
        output_path: 输出模型路径

    返回:
        bool: 修复是否成功
    """
    try:
        # 加载 ONNX 模型
        model = onnx.load(model_path)

        # 找到所有 LayerNormalization 节点
        layer_norm_nodes = [node for node in model.graph.node if node.op_type == 'LayerNormalization']
        print(f'找到 {len(layer_norm_nodes)} 个 LayerNormalization 节点')

        fixed_count = 0
        for node in layer_norm_nodes:
            # 检查输入数量
            if len(node.input) < 3:
                # 获取输入名称
                input_name = node.input[0]
                scale_name = node.input[1]

                # 获取 scale 的形状
                scale_init = None
                for init in model.graph.initializer:
                    if init.name == scale_name:
                        scale_init = init
                        break

                if scale_init is not None:
                    # 创建全零的 bias
                    scale_array = numpy_helper.to_array(scale_init)
                    bias_array = np.zeros_like(scale_array)
                    bias_name = f"{node.name}_bias"
                    bias_tensor = numpy_helper.from_array(bias_array, name=bias_name)

                    # 添加到 initializers
                    model.graph.initializer.append(bias_tensor)

                    # 修改节点，添加 bias 输入
                    # 清空原有输入
                    while len(node.input) > 0:
                        node.input.pop()

                    # 添加新的输入
                    node.input.extend([input_name, scale_name, bias_name])
                    fixed_count += 1

        print(f'修复了 {fixed_count} 个 LayerNormalization 节点')

        # 保存模型
        onnx.save(model, output_path)
        print(f'修复后的模型已保存到: {output_path}')
        return True
    except Exception as e:
        print(f'LayerNormalization 节点修复失败: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("用法: python fix_layernorm.py <输入模型路径> <输出模型路径>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    fix_layernorm_nodes(input_path, output_path)
