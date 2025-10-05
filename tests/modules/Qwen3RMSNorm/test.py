
import subprocess
import sys
import numpy as np
from transformers import AutoModelForCausalLM
import torch
import os
from tests.test_utils import parse_shape, print_error_analysis, create_temp_folder, generate_random_txt
import argparse
import shutil

# Keep only the required functions here
def run_rms_norm(norm_model, in_file: str, out_file: str, shape: tuple):
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

	# --- Run normalization ---
	input_tensor = torch.from_numpy(input_data)
	with torch.no_grad():
		output = norm_model(input_tensor).cpu().numpy()

	# --- Save output ---
	out_dir = os.path.dirname(out_file)
	if out_dir:
		os.makedirs(out_dir, exist_ok=True)
	np.savetxt(out_file, output.flatten(), fmt="%.6f")  # flatten output to 1D
	print(f"✅ Output saved to {out_file}")

def run_py(model_path, input_path, output_path, shape):
	py_path = "run_Qwen3RMSNorm.py"
	shape_str = ','.join(str(s) for s in shape)
	cmd = [sys.executable, py_path, "--model", model_path, "--input", input_path, "--output", output_path, "--shape", shape_str]
	print("Running Python RMSNorm:", ' '.join(cmd))
	subprocess.run(cmd, check=True)

def save_norm_weight(norm, file_path):
	# save the weights to a file
	with open(file_path, "w") as f:
		for weight in norm.weight:
			f.write(f"{weight.item()}\n")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run and compare C++/Python RMSNorm implementations.")
	parser.add_argument('--build_dir', type=str, required=True, help='Path to build directory containing run_Qwen3RMSNorm.exe')
	parser.add_argument('--model_path', type=str, required=True, help='Path to Qwen model for Python RMSNorm')
	parser.add_argument('--shape', type=str, default='1,2560', help='Shape as comma-separated string, e.g. "1,2560"')
	parser.add_argument('--eps', type=float, default=1e-6, help='Epsilon value (default 1e-6)')
	parser.add_argument('--out_cpp', type=str, default='out_cpp.txt', help='C++ output file')
	parser.add_argument('--out_py', type=str, default='out_py.txt', help='Python output file')
	parser.add_argument('--save_results', action='store_true', help='Save temp folder and results')
	args = parser.parse_args()

	shape = tuple(int(x) for x in args.shape.split(','))
	temp_folder = create_temp_folder()
	input_path = os.path.join(temp_folder, 'input.txt')
	weight_path = os.path.join(temp_folder, 'weight.txt')
	out_cpp_path = os.path.join(temp_folder, args.out_cpp)
	out_py_path = os.path.join(temp_folder, args.out_py)

	# Generate random input
	arr = generate_random_txt(input_path, shape)

	# Load model and save norm weights
	model = AutoModelForCausalLM.from_pretrained(
		args.model_path,
		torch_dtype="auto",
		device_map="cpu"
	)
	norm = model.model.norm
	save_norm_weight(norm, weight_path)

	exe_path = f"{args.build_dir}/Release/run_Qwen3RMSNorm.exe"
	def run_py(model_path, input_path, output_path, shape):
		print("Running Python RMSNorm")
		run_rms_norm(model_path, input_path, output_path, shape)

	def run_cpp_arg(input_path, weight_path, output_path, shape, eps):
		shape_str = ','.join(str(s) for s in shape)
		cmd = [exe_path, input_path, weight_path, output_path, shape_str, str(eps)]
		print("Running C++ RMSNorm:", ' '.join(cmd))
		subprocess.run(cmd, check=True)

	run_py(norm, input_path, out_py_path, shape)
	del model

	run_cpp_arg(input_path, weight_path, out_cpp_path, shape, args.eps)
	
	print_error_analysis(out_cpp_path, out_py_path)

	if not args.save_results:
		shutil.rmtree(temp_folder)
		print(f"Temporary folder '{temp_folder}' deleted.")
	else:
		print(f"Temporary folder '{temp_folder}' retained for inspection.")