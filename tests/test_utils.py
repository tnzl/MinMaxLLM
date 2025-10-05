import numpy as np
import os
import argparse
import shutil

def parse_shape(shape_str: str):
	"""Convert comma-separated string to tuple of ints, e.g., '2,256,8,9' -> (2,256,8,9)"""
	try:
		return tuple(int(dim) for dim in shape_str.split(","))
	except Exception as e:
		raise argparse.ArgumentTypeError(f"Invalid shape '{shape_str}': {e}")

def print_error_analysis(ref_path, actual_path):
	ref = np.loadtxt(ref_path, dtype=np.float32)
	actual = np.loadtxt(actual_path, dtype=np.float32)
	l2_error = np.sqrt(np.mean((ref - actual) ** 2))
	max_error = np.max(np.abs(ref - actual))
	norm_ref = np.linalg.norm(ref)
	norm_actual = np.linalg.norm(actual)
	relative_error = np.abs(norm_ref - norm_actual) / (norm_ref + 1e-12)
	significant_error_threshold = 1e-4
	significant_errors = np.sum(np.abs(ref - actual) > significant_error_threshold)
	print("\nError Analysis:")
	print(f"L2 Error: {l2_error}")
	print(f"Max Error: {max_error}")
	print(f"Relative Error (L2 norm): {relative_error * 100}%")
	print(f"Elements with error > {significant_error_threshold}: {significant_errors} ({100.0 * significant_errors / ref.size}%)")

def create_temp_folder(folder_name="temp_test_folder"):
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)
	return folder_name

def generate_random_txt(file_path, shape):
	arr = np.random.randn(*shape).astype(np.float32)
	np.savetxt(file_path, arr.flatten())
	return arr
