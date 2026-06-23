import open3d as o3d
import numpy as np
import copy
from scipy.io import loadmat
from scipy.spatial import cKDTree

labels = ['low', 'medium', 'high']
num_configs = 30

def propagate_deformation(low_res_src, low_res_deformed, high_res_points, k = 3):
    tree = cKDTree(low_res_src)
    distances, indices = tree.query(high_res_points, k=k)
    # weights based on inverse distance
    weights = 1.0 / (distances + 1e-8)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    # weighted deformation
    displacements = low_res_deformed[indices] - low_res_src[indices]
    weighted_displacements = np.sum(displacements * weights[:, :, np.newaxis], axis=1)
    return high_res_points + weighted_displacements


# Read source point cloud (high resolution)
filename_SRC = r"C:\Users\olocc\OneDrive\Desktop\Thesis_new\Dataset\Base_meshes\Model27\Merged.ply"
SRC = o3d.io.read_point_cloud(filename_SRC)
SRC.estimate_normals()

# Read source point cloud (medium resolution)
filename_src = r"C:\Users\olocc\OneDrive\Desktop\Thesis_new\Dataset\Clouds_for_lepard\Model27\base.ply"
src = o3d.io.read_point_cloud(filename_src)
src.estimate_normals()

# Define folder to save high resolution deformed clouds
folder_out = "C:\\Users\\olocc\\OneDrive\\Desktop\\Thesis_new\\Dataset\\For_tests\\Lepard\\"

# Generate high resolution deformed clouds
for label in labels:
    for i in range(num_configs):
      config = label + str(i)
    filename_src_def = ("C:\\Users\\olocc\\OneDrive\\Desktop\\Thesis_new\\Dataset\\Lepard_nr_icp_results\\" +
                        config + ".mat")
    # Read deformed medium resolution point clouds
    data = loadmat(filename_src_def)
    src_def_pts = data['src_def_pts']
    src_def = o3d.geometry.PointCloud()
    src_def.points = o3d.utility.Vector3dVector(src_def_pts)
    src_def.estimate_normals()

    # Extend deformation to high resolution clouds
    SRC_def = copy.deepcopy(SRC)
    SRC_def.points = o3d.utility.Vector3dVector(
        propagate_deformation(
            np.asarray(src.points),
            np.asarray(src_def.points),
            np.asarray(SRC_def.points),
            k=6
        )
    )
    SRC_def.estimate_normals()
    filename_out = folder_out + config + ".ply"
    o3d.io.write_point_cloud(filename_out, SRC_def)