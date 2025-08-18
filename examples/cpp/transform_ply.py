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
        matrix = np.loadtxt(filepath, comments='#')
        if matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must be 4x4.")
        return matrix
    except Exception as e:
        print(f"Error loading transformation matrix from {filepath}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Apply a transformation matrix to a PLY file.")
    parser.add_argument("ply_file", help="Path to the input PLY file.")
    parser.add_argument("transform_file", help="Path to the text file containing the 4x4 transformation matrix.")
    
    args = parser.parse_args()

    # --- 1. Load PLY file ---
    print(f"Loading PLY file: {args.ply_file}")
    if not os.path.exists(args.ply_file):
        print(f"Error: Input PLY file not found at {args.ply_file}")
        return
        
    pcd = o3d.io.read_point_cloud(args.ply_file)
    if not pcd.has_points():
        print("Error: The loaded PLY file has no points.")
        return
    print(f"Successfully loaded point cloud with {len(pcd.points)} points.")

    # --- 2. Load transformation matrix ---
    print(f"Loading transformation matrix: {args.transform_file}")
    if not os.path.exists(args.transform_file):
        print(f"Error: Transformation file not found at {args.transform_file}")
        return

    transform_matrix = load_transformation_matrix(args.transform_file)
    if transform_matrix is None:
        return
    print("Transformation matrix loaded successfully:")
    print(transform_matrix)

    # --- 3. Apply transformation ---
    print("\nApplying transformation...")
    pcd.transform(transform_matrix)
    print("Transformation applied.")

    # --- 4. Save the transformed PLY file ---
    output_filename = os.path.splitext(args.ply_file)[0] + "_transformed.ply"
    print(f"Saving transformed point cloud to: {output_filename}")
    o3d.io.write_point_cloud(output_filename, pcd, write_ascii=True)
    print("Done.")

if __name__ == "__main__":
    main()

