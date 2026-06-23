import copy
import numpy as np
import open3d as o3d
from open3d.pipelines import registration as reg
from probreg import cpd
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
import time
import os

# scale conversion factor
scale = 680
# Labels to identify deformation degrees
labels = ['low', 'medium', 'high']
# Number of configurations per deformation degree
num_configs = 30

# Preprocess point cloud (downsampling and FPFH feature extraction)
def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

# Plot point clouds
def show_point_clouds(src, tgt, title):
    vis = o3d.visualization.Visualizer()
    vis.create_window(title, 1400, 1050)
    src.paint_uniform_color([0.459, 0.98, 0.38])
    vis.add_geometry(src)
    tgt.paint_uniform_color([0.459, 0.063, 0.49])
    vis.add_geometry(tgt)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=75.0, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.078, 0.212, 0.212])
    vis.run()
    vis.destroy_window()

# point cloud normalization to make faster the process
def normalize_point_cloud(pcd):
    pts = np.asarray(pcd.points)
    centroid = np.mean(pts, axis=0)
    pts -= centroid
    scale = np.max(np.linalg.norm(pts, axis=1))
    pts /= scale
    pcd_norm = o3d.geometry.PointCloud()
    pcd_norm.points = o3d.utility.Vector3dVector(pts)
    return pcd_norm, centroid, scale

def denormalize_points(points, centroid, scale):
    return points * scale + centroid

def compute_global_metrics(src, tgt):
    src_points = np.asarray(src.points)
    kdtree = o3d.geometry.KDTreeFlann(tgt)
    dists = []
    # Find minimum distances per each point of the source point cloud
    for pt in src_points:
        [_, _, dist] = kdtree.search_knn_vector_3d(pt, 1)
        dists.append(np.sqrt(dist[0]))
    index = np.mean(dists)
    return index

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def optimized_registration(error_in, src, tgt, den_params):
    # Compute beta
    src_down = o3d.geometry.PointCloud()
    src_down.points = o3d.utility.Vector3dVector(denormalize_points(np.asarray(src.points), den_params[0], den_params[1]))
    tree = cKDTree(np.asarray(src_down.points))
    distances, _ = tree.query(np.asarray(src_down.points), k=6)
    mean_dist = np.mean(distances[:, 1:])
    print("Mean distance:", mean_dist)
    beta = (-0.5749 * (error_in / 100) ** 3
            - 1.2169 * (error_in / 100) ** 2
            + 4.5134 * error_in / 100
            + 0.00727) * mean_dist
    # Golden Section Search setup
    invphi = (np.sqrt(5) - 1) / 2
    tol = 0.001
    lambda_low = 0.1
    lambda_up = 5.0
    def compute_error(l):
        # Compute CPD error for a specific value of lambda
        tf_param, _, _ = cpd.registration_cpd(
            src, tgt, tf_type_name="nonrigid", w=0.05, beta=beta,
            lmd=l, maxiter=100, tol=1e-5
        )
        reg_pts = tf_param.transform(np.asarray(src.points))
        src_def = o3d.geometry.PointCloud()
        src_def.points = o3d.utility.Vector3dVector(
            denormalize_points(reg_pts, den_params[0], den_params[1])
        )
        tgt_den = o3d.geometry.PointCloud()
        tgt_den.points = o3d.utility.Vector3dVector(
            denormalize_points(np.asarray(tgt.points), den_params[0], den_params[1])
        )
        err = 1000 * compute_global_metrics(src_def, tgt_den) / scale
        return err, src_def
    # Initialization
    c = lambda_up - invphi * (lambda_up - lambda_low)
    d = lambda_low + invphi * (lambda_up - lambda_low)
    f_c, src_c = compute_error(c)
    f_d, src_d = compute_error(d)
    prev_best = min(f_c, f_d)
    # Optimization loop
    while True:
        if f_c < f_d:
            # min([lambda_low, d])
            lambda_up = d
            d, f_d, src_d = c, f_c, src_c
            c = lambda_up - invphi * (lambda_up - lambda_low)
            f_c, src_c = compute_error(c)
        else:
            # min([c, lambda_up])
            lambda_low = c
            c, f_c, src_c = d, f_d, src_d
            d = lambda_low + invphi * (lambda_up - lambda_low)
            f_d, src_d = compute_error(d)
        current_best = min(f_c, f_d)
        if abs(prev_best - current_best) < tol:
            break
        prev_best = current_best
    # Choose best result
    if f_c < f_d:
        l_opt, src_def = c, src_c
    else:
        l_opt, src_def = d, src_d
    print("CPD parameters:")
    print("Beta =", beta)
    print("Lambda =", l_opt)
    return src_def

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

def distance_preservation_error(idxs, original, deformed):
    errors = np.zeros(idxs.shape[0])
    for i in range(idxs.shape[0]):
        start = int(idxs[i, 0])
        end = int(idxs[i, 1])
        orig_pts = np.asarray(original.points)[start:end, :]
        def_pts = np.asarray(deformed.points)[start:end, :]
        orig_dist = pdist(orig_pts, metric='euclidean')
        def_dist = pdist(def_pts, metric='euclidean')
        valid = orig_dist > 1e-8
        rel_error = ((def_dist[valid] - orig_dist[valid]) / orig_dist[valid]) ** 2
        errors[i] = np.mean(rel_error)
    return np.mean(errors)

def compute_distance_metrics(source, target):
    source_points = np.asarray(source.points, dtype=np.float32)
    target_points = np.asarray(target.points)
    # Prepare KDTree for efficient query
    tree = cKDTree(target_points)
    dists, _ = tree.query(source_points, k=1)
    # Point-to-point
    std_p2p = np.std(dists)
    dist_min = np.min(dists)
    dist_max = np.max(dists)
    dist_mean = np.mean(dists)
    range_p2p = dist_max - dist_min
    rmse_p2p = np.sqrt(np.mean(dists ** 2))
    return (dist_min, dist_max, dist_mean, std_p2p, range_p2p, rmse_p2p)

def compute_couple_metrics(src, tgt):
    src_points = np.asarray(src.points)
    tgt_points = np.asarray(tgt.points)
    # Find distances between the source point and the related target point
    dists = np.linalg.norm(src_points - tgt_points, axis=1)
    index = np.mean(dists)
    return index


# Load source and target Point Clouds
filename_src = r"C:\Users\olocc\OneDrive\Desktop\Thesis_new\Dataset\For_tests\Model27\base.ply"
sep_filename = r"C:\Users\olocc\OneDrive\Desktop\Thesis_new\Dataset\Base_meshes\Model27\Infos.txt"

# Write header into the output filename
filename_out = r"C:\Users\olocc\OneDrive\Desktop\Thesis_new\TC_model_registration\Results\CPD.txt"
with open(filename_out, "w") as f:
    f.write("Time" + "\t" + "Error out" + "\t" + "Deformation" + "\t" + "Dist min" + "\t" + "Dist max" +
            "\t" + "Dist mean" + "\t" + "std" + "\t" + "Range" + "\t" + "RMSE")

# Compute body separators
separators = np.zeros((17 , 2))
i = 0
with open(sep_filename, 'r') as f:
    for line in f:
        for line in f:
            if "Scale factor" not in line:
                _, start, end = line.split()
                separators[i, 0] = int(start)
                separators[i, 1] = int(end) + 1
                i += 1


# Load source and target point clouds
for label in labels:
    for num_config in range(num_configs):
        filename_tgt = ("C:\\Users\\olocc\\OneDrive\\Desktop\\Thesis_new\\Dataset\\For_tests\\Model27\\" +
                        label + str(num_config) + ".ply")

        src = o3d.io.read_point_cloud(filename_src)
        tgt = o3d.io.read_point_cloud(filename_tgt)
        # show_point_clouds(src, tgt, "Beforer Registration")
        error_in = 1000 * compute_global_metrics(src, tgt) /scale
        print("Evaluation metric before registration: " + str(scale * error_in / 1000) + " mm")

        # Define voxel size for clouds downsampling
        voxel_size = scale * 0.01
        
        starting_time = time.time()
        src_down, src_fpfh = preprocess_point_cloud(src, voxel_size)
        tgt_down, tgt_fpfh = preprocess_point_cloud(tgt, voxel_size)
        if error_in > 60.0:
            result_ransac = execute_global_registration(src_down, tgt_down,
                                                        src_fpfh, tgt_fpfh,
                                                        voxel_size)
            src_ransac = copy.deepcopy(src_down)
            src_ransac.transform(result_ransac.transformation)
            # Deformable point clouds normalization
            src_norm, src_center, src_factor = normalize_point_cloud(copy.deepcopy(src_ransac))
        else:
            src_norm, src_center, src_factor = normalize_point_cloud(copy.deepcopy(src_down))
        tgt_norm, tgt_center, tgt_factor = normalize_point_cloud(copy.deepcopy(tgt_down))
        src_def_down = optimized_registration(error_in, src_norm, tgt_norm, [tgt_center, tgt_factor])
        src_def = copy.deepcopy(src)
        def_start = time.time()
        src_def.points =  o3d.utility.Vector3dVector(propagate_deformation(
            np.asarray(src_down.points),
            np.asarray(src_def_down.points),
            np.asarray(src_def.points),
            k=6
        ))
        src_def.estimate_normals()
        reg_time = time.time() - starting_time
        print("Registration total time: " + str(int(reg_time / 60)) + " min " + str(int(reg_time % 60)) + " s")
        error_out = compute_global_metrics(src_def, tgt)
        print("Global evaluation metric after registration: " + str(error_out) + " mm")
        # show_point_clouds(src_def, tgt, "After Registration")
        # src_def_t = copy.deepcopy(src_def)
        # src_def_t.translate(np.array([0, 0, 0.2 * scale]))
        # show_point_clouds(src_def_t, tgt, "After CPD registration - Separated clouds")

        # Computing and saving registration results
        dist_min, dist_max, dist_mean, std_p2p, range_p2p, rmse_p2p = compute_distance_metrics(src_def, tgt)
        error_def = distance_preservation_error(separators, src, src_def)
        err_couple_out = compute_couple_metrics(src_def, tgt)

        # Save results into a text file
        with open(filename_out, "a") as f:
            f.write("\n" + str(reg_time) + "\t")
            f.write(str(err_couple_out) + "\t")
            f.write(str(error_def) + "\t")
            f.write(str(dist_min) + "\t")
            f.write(str(dist_max) + "\t")
            f.write(str(dist_mean) + "\t")
            f.write(str(std_p2p) + "\t")
            f.write(str(range_p2p ) + "\t")
            f.write(str(rmse_p2p))