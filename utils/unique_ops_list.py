import onnx

model_path = "c:\\Users\\iamta\\Downloads\\model.onnx"  # Replace with your ONNX model path

# Load the ONNX model
print("Loading model", end="...")
model = onnx.load(model_path)
print("[DONE]")

# Extract the graph from the model
graph = model.graph
# Collect all op types in the graph
ops = {node.op_type for node in graph.node}

print("Unique Operators in the Model:")

for op in ops :
    print(op)