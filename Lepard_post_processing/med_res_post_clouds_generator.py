import open3d as o3d
import numpy as np
import copy
from scipy.spatial import cKDTree

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

folder_base = "C:\\Users\\olocc\\OneDrive\\Desktop\\Thesis_new\\LepardTests\\2711\\test_results\\"
med_res_path = "C:\\Users\\olocc\\OneDrive\\Desktop\\Thesis_new\\Dataset\\Clouds_for_lepard\\Model27\\"
folder_out = "C:\\Users\\olocc\\OneDrive\\Desktop\\Thesis_new\\LepardTests\\2711_med_res\\"

# Load medium resolution source point cloud
SRC_filename = med_res_path + "base.ply"
SRC = o3d.io.read_point_cloud(SRC_filename)
SRC.estimate_normals()
SRC_center = SRC.get_center()

# Load Lepard final clouds
n_pred = 90
for i in range(n_pred):
    src_filename = folder_base + "prediction_" + str(i) + ".npz"
    # Final source clouds
    data = np.load(src_filename)
    src_pts = data['s_pcd'].squeeze()
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(src_pts)
    src.estimate_normals()
    src_center = src.get_center()
    offset = SRC_center - src_center
    src.translate(offset)
    # Apply transformation
    flow = data['s2t_flow'].squeeze()
    R = data['R'].squeeze()
    t = data['t'].squeeze()
    def_pts = src_pts + flow
    src_def_pts = np.zeros((def_pts.shape[0], 3))
    for j in range(def_pts.shape[0]):
        src_def_pts[j, :] = R @ def_pts[j, :] + t
    src_def = o3d.geometry.PointCloud()
    src_def.points = o3d.utility.Vector3dVector(src_def_pts)
    src_def.estimate_normals()
    src_def.translate(offset)
    # Extend deformation to high resolution clouds
    SRC_def = copy.deepcopy(SRC)
    SRC_def.points = o3d.utility.Vector3dVector(
        propagate_deformation(
            np.asarray(src.points),
            np.asarray(src_def.points),
            np.asarray(SRC_def.points),
            k = 6
        )
    )
    SRC_def.estimate_normals()
    # Save deformed medium resolution clouds
    filename_out = folder_out + "prediction_" + str(i) + ".ply"
    o3d.io.write_point_cloud(filename_out, SRC_def)