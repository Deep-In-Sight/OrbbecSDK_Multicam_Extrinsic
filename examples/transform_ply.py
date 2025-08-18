import open3d as o3d
import numpy as np
import argparse
import os

def load_transformation_matrix(filepath):
    """
    Loads a 4x4 transformation matrix from a text file.
    Skips lines starting with '#'.
    """
    try:
        # We need to specify the full path relative to the script's execution directory
        matrix = np.loadtxt(filepath, comments='#')
        if matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must be 4x4.")
        return matrix
    except Exception as e:
        print(f"Error loading transformation matrix from {filepath}: {e}")
        return None

def main():
    # Get the directory where the script is located to build full paths
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Apply a transformation matrix to a PLY file.")
    parser.add_argument("ply_file", help="Path to the input PLY file (relative to the script).")
    parser.add_argument("transform_file", help="Path to the text file with the 4x4 matrix (relative to the script).")
    
    args = parser.parse_args()

    # Construct full paths
    ply_file_path = os.path.join(script_dir, args.ply_file)
    transform_file_path = os.path.join(script_dir, args.transform_file)

    # --- 1. Load PLY file ---
    print(f"Loading PLY file: {ply_file_path}")
    if not os.path.exists(ply_file_path):
        print(f"Error: Input PLY file not found at {ply_file_path}")
        return
        
    pcd = o3d.io.read_point_cloud(ply_file_path)
    if not pcd.has_points():
        print("Error: The loaded PLY file has no points.")
        return
    print(f"Successfully loaded point cloud with {len(pcd.points)} points.")

    # --- 2. Load transformation matrix ---
    print(f"Loading transformation matrix: {transform_file_path}")
    if not os.path.exists(transform_file_path):
        print(f"Error: Transformation file not found at {transform_file_path}")
        return

    transform_matrix = load_transformation_matrix(transform_file_path)
    if transform_matrix is None:
        return
    print("Transformation matrix loaded successfully:")
    print(transform_matrix)

    # --- 3. Apply transformation ---
    print("\nApplying transformation...")
    pcd.transform(transform_matrix)
    print("Transformation applied.")

    # --- 4. Save the transformed PLY file ---
    output_filename = os.path.splitext(ply_file_path)[0] + "_transformed.ply"
    print(f"Saving transformed point cloud to: {output_filename}")
    o3d.io.write_point_cloud(output_filename, pcd, write_ascii=True)
    print("Done.")

if __name__ == "__main__":
    main()
