import subprocess
import sys
import numpy as np
from transformers import AutoModelForCausalLM
import torch
import os
from tests.test_utils import parse_shape, print_error_analysis, create_temp_folder, generate_random_txt
import argparse
import shutil
import time
from tqdm import tqdm

np.random.seed(42)

def run_mlp(mlp, in_file: str, out_file: str, shape: tuple):
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

    # --- Run mlp ---
    input_tensor = torch.from_numpy(input_data)
    mlp = mlp.float()  # ensure weights are in float32 for matmul
    with torch.no_grad():
        iters = 10
        lat = 0        
        # run for multiple iterations to get stable timing
        for _ in tqdm(range(iters), desc="Running Python MLP"):
            st_t = time.time_ns()
            output = mlp(input_tensor).cpu().numpy()
            end_t = time.time_ns()
            lat += (end_t - st_t)
        print(f"Python MLP execution time: {((lat)/1e3)/iters :.2f} µs over 10 runs")

    # --- Save output ---
    out_dir = os.path.dirname(out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savetxt(out_file, output.flatten(), fmt="%.6f")  # flatten output to 1D
    print(f"✅ Python Output saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and compare C++/Python Qwen3 MLP gate matmul implementations.")
    parser.add_argument('--exe_path', type=str, required=True, help='Path to run_Qwen3MLP.exe')
    parser.add_argument('--model_path', type=str, required=True, help='Path to Qwen model')
    parser.add_argument('--safetensors_path', type=str, required=True, help='Path to Qwen model safetensor for cpp run')
    parser.add_argument('--layer_index', default=0, type=int, help='Layer index to run MLP on')
    parser.add_argument('--save_results', action='store_true', help='Save temp folder and results')
    parser.add_argument('--mmap', action='store_true', help='Use memory-mapped safetensors in C++')
    parser.add_argument('--advise', action='store_true', help='Use memory-mapped safetensors in C++')
    args = parser.parse_args()

    temp_folder = create_temp_folder()
    print(f"Using temporary folder: {temp_folder}")

    # load model in PyTorch (will automatically read safetensors)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",   # keeps dtype as stored (fp16, bf16, fp32, etc.)
        device_map="cpu"      # or "auto" if you want GPU mapping
    )

    mlp = model.model.layers[0].mlp

    # Prepare input
    input_dim = mlp.gate_proj.in_features
    up_dim = mlp.up_proj.out_features
    output_dim = mlp.down_proj.out_features

    shape = (1, input_dim)  # example shape: (1, input_dim)

    # Generate random input file
    input_path = os.path.join(temp_folder, "input.txt")
    generate_random_txt(input_path, shape)
    print(f"Random input generated at {input_path}")

    # Run Python implementation
    out_py_path = os.path.join(temp_folder, "out_py.txt")
    run_mlp(mlp, input_path, out_py_path, (1, input_dim))

    # delete the model to save memory before running cpp
    del model

    # Run C++ implementation
    out_cpp_path = os.path.join(temp_folder, "out_cpp.txt")
    cpp_exe = args.exe_path
    cpp_cmd = [
        cpp_exe,
        args.safetensors_path,
        input_path,
        out_cpp_path,
        str(shape[0]),
        str(input_dim),
        str(up_dim),
        str(output_dim),
        "0",
        "1" if args.mmap else "0",
        "1" if args.advise else "0"
    ]

    # print the command to be run
    cpp_cmd_str = " ".join(cpp_cmd)
    print(f"Running C++ command: {cpp_cmd_str}")

    # run command
    if not os.path.exists(cpp_exe):
        raise FileNotFoundError(f"C++ executable not found: {cpp_exe}")

    result = subprocess.run(cpp_cmd, text=True)
    if result.returncode != 0:
        print("C++ execution failed:")
        print(result.stderr)
        sys.exit(1)

    # Compare outputs
    print_error_analysis(out_cpp_path, out_py_path)
    print("✅ Comparison done.")

    if not args.save_results:
        shutil.rmtree(temp_folder)
        print(f"Temporary folder {temp_folder} deleted.")
    else : 
        print(f"Temporary folder {temp_folder} retained for inspection.")