# onnx\_modifier
a toolset to modify onnx models

## intro
### add\_0bias\_to\_2\_input\_layernorm
convert 2-input layernorm nodes (x, scale) to standard 3 inputs (x, scale, bias) by adding a all-zero bias input.

### replace\_einsum\_with\_matmul
> [说明文档](./doc/MatMul替换Einsum算子.md)

replace einsum nodes with matmul, supported equations:
- ...ij, ...jc -> ...ic / ...nic,...ncj -> ...nij
- ...ic, ...jc -> ...ij / ...hic,...hjc -> ...hij
- ...nij, ijc -> ...nic (complicated)

## usage
```
python {script.py} <model_input_path> <model_output_path>
```
