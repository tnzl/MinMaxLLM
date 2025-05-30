import onnx
from onnx import utils

# Define paths to your original model and the output model
input_path = "c:\\Users\\iamta\\Downloads\\model.onnx"
output_path = "qwen3_block.onnx"

# Specify the input and output tensor names for the subgraph
input_names = ["/model/layers.2/attn/GroupQueryAttention/output_0"]  # Replace with actual names
output_names= ["/model/layers.3/attn/GroupQueryAttention/output_0"]  # Replace with actual names

# Extract the sub-model
utils.extract_model(
    input_path=input_path,
    output_path=output_path,
    input_names=input_names,
    output_names=output_names,
    # check_model=True,
    # infer_shapes=True
)

print(f"Subgraph extracted and saved to {output_path}")