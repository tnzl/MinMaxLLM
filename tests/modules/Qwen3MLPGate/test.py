import subprocess
import sys
import numpy as np
from transformers import AutoModelForCausalLM
import torch
import os
from tests.test_utils import parse_shape, print_error_analysis, create_temp_folder, generate_random_txt
import argparse
import shutil

np.random.seed(42)

def run_gate_proj(gate_model, in_file: str, out_file: str, shape: tuple):
    # --- Load input ---
    if not os.path.exists(in_file):
        raise FileNotFoundError(f"Input file not found: {in_file}")
    flat_data = np.loadtxt(in_file, dtype=np.float32)

    # --- Reshape input ---
    expected_size = np.prod(shape)
    if flat_data.size != expected_size:
        raise ValueError(
            f"Input data size ({flat_data.size}) does not match expected shape {shape} "
            f"(total elements {expected_size})"
        )
    input_data = flat_data.reshape(shape)

    # --- Run gate projection ---
    input_tensor = torch.from_numpy(input_data)
    gate_model = gate_model.float()  # ensure weights are in float32 for matmul
    with torch.no_grad():
        output = gate_model(input_tensor).cpu().numpy()

    # --- Save output ---
    out_dir = os.path.dirname(out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savetxt(out_file, output.flatten(), fmt="%.6f")  # flatten output to 1D
    print(f"✅ Output saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and compare C++/Python Qwen3 MLP gate matmul implementations.")
    parser.add_argument('--build_dir', type=str, required=True, help='Path to build directory containing run_Qwen3RMSNorm.exe')
    parser.add_argument('--model_path', type=str, required=True, help='Path to Qwen model for Python RMSNorm')
    parser.add_argument('--save_results', action='store_true', help='Save temp folder and results')
    args = parser.parse_args()

    temp_folder = create_temp_folder()
    print(f"Using temporary folder: {temp_folder}")

    # load model in PyTorch (will automatically read safetensors)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",   # keeps dtype as stored (fp16, bf16, fp32, etc.)
        device_map="cpu"      # or "auto" if you want GPU mapping
    )

    gate_proj = model.model.layers[0].mlp.gate_proj

    # Prepare input
    shape = (1, gate_proj.in_features)

    # Generate random input file
    input_path = os.path.join(temp_folder, "input.txt")
    generate_random_txt(input_path, shape)
    print(f"Random input generated at {input_path}")

    N, K, M = shape[0], gate_proj.in_features, gate_proj.out_features

    # Run Python implementation
    out_py_path = os.path.join(temp_folder, "out_py.txt")
    run_gate_proj(gate_proj, input_path, out_py_path, (N, K))
    print("Outpit (first 10 elements):", np.loadtxt(out_py_path, dtype=np.float32).flatten()[:10])
    print("Outpit (first 10 elements):", np.loadtxt(out_py_path, dtype=np.float32).flatten()[-10:])
    print(f"✅ Python output saved to {out_py_path}")

    #  full path to weight file
    weight_path = os.path.join(temp_folder, "gate_proj_weight.bin")
    weights_flat = gate_proj.weight.to(torch.float32).flatten().detach().cpu().numpy()
    weights_flat.tofile(weight_path)

    print(f"Gate projection weights saved to {weight_path}")

    # print first 10 elements of input and weight for verification
    input_data = np.loadtxt(input_path, dtype=np.float32)
    print("Input (first 10 elements):", input_data.flatten()[:10])
    print("Input (last 10 elements):", input_data.flatten()[-10:])

    print("Weight (first 10 elements):", weights_flat[:10])
    print("Weight (last 10 elements):", weights_flat[-10:])

    # Run C++ implementation
    # Usage: <build_dir>\Release\run_Qwen3MLPGate.exe <input.txt> <weight.txt> <output.txt> <N> <K> <M>
    out_cpp_path = os.path.join(temp_folder, "out_cpp.txt")
    cpp_exe = os.path.join(args.build_dir, "Release\\run_Qwen3MLPGate.exe")
    print(f"Using C++ executable: {cpp_exe}")
    if not os.path.exists(cpp_exe):
        raise FileNotFoundError(f"C++ executable not found: {cpp_exe}")

    cmd = [cpp_exe, input_path, weight_path, out_cpp_path, str(N), str(K), str(M)]
    print("Running C++ Qwen3 MLP gate projection:", ' '.join(cmd))
    subprocess.run(cmd, check=True)
    print(f"✅ C++ output saved to {out_cpp_path}")

    # Compare outputs
    print_error_analysis(out_cpp_path, out_py_path)
    print("✅ Comparison done.")

    if not args.save_results:
        shutil.rmtree(temp_folder)
        print(f"Temporary folder {temp_folder} deleted.")
    else : 
        print(f"Temporary folder {temp_folder} retained fpr inspection.")
    


    
    



    