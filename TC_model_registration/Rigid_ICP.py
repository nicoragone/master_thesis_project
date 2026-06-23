import numpy as np
import open3d as o3d
import copy
import time
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree

# scale factor (adapt OpenSim model to the segmented one)
scale = 680
# Labels to distinguish different deformations
labels = ['low', 'medium', 'high']
# Number of configurations per deformation degree
n_configs = 30

# Plot point clouds during the registration phases
def show_point_clouds(src_buffer, tgt_buffer, title):
    vis = o3d.visualization.Visualizer()
    vis.create_window(title, 1400, 1050)
    for pc in src_buffer:
        pc.paint_uniform_color([1, 0.706, 0])
        vis.add_geometry(pc)
    for pc in tgt_buffer:
        pc.paint_uniform_color([0, 0.651, 0.929])
        vis.add_geometry(pc)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=75.0, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.141, 0.157, 0.239])
    vis.run()
    vis.destroy_window()

# Preprocess (downsampling and extracting features) point clouds
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

# Prepare dataset (imposing an initial transformation)
def prepare_dataset(source, target, trans_init, voxel_size):
    source.transform(trans_init)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source_down, target_down, source_fpfh, target_fpfh

# Execute global registration (using a RANSAC approach)
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

# Local refinement (generate points into the down-sampled clouds)
def refine_registration(source, target, result_ransac, voxel_size):
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

# Evaluation metrics
def compute_global_metrics(src_buffer, tgt_buffer):
    # Generate merged point clouds
    src_pc = o3d.geometry.PointCloud()
    for pc in src_buffer:
        src_pc += pc
    tgt_pc = o3d.geometry.PointCloud()
    for pc in tgt_buffer:
        tgt_pc += pc
    src_points = np.asarray(src_pc.points)
    kdtree = o3d.geometry.KDTreeFlann(tgt_pc)
    dists = []
    # Find minimum distances per each point of the source point cloud
    for pt in src_points:
        [_, _, dist] = kdtree.search_knn_vector_3d(pt, 1)
        dists.append(np.sqrt(dist[0]))
    index = np.mean(dists)
    return index

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

# Inverse transformation matrix (for the inverse test):
def inverse_transform(T):
    T_inv = np.eye(4)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return T_inv


src_filename = r"C:\Users\olocc\OneDrive\Desktop\Thesis_new\Dataset\For_tests\Model27\base.ply"
sep_filename = r"C:\Users\olocc\OneDrive\Desktop\Thesis_new\Dataset\Base_meshes\Model27\Infos.txt"

# Write header into the output filename
filename_pre = r"C:\Users\olocc\OneDrive\Desktop\Thesis_new\TC_model_registration\Results\Pre.txt"
filename_out = r"C:\Users\olocc\OneDrive\Desktop\Thesis_new\TC_model_registration\Results\RANSAC+ICP.txt"
with open(filename_out, "a") as f:
    f.write("Time" + "\t" + "Error out" + "\t" + "Deformation" + "\t" + "Dist min" + "\t" + "Dist max" +
            "\t" + "Dist mean" + "\t" + "std" + "\t" + "Range" + "\t" + "RMSE")
with open(filename_pre, "a") as f:
    f.write("Error out" + "\t" + "Deformation" + "\t" + "Dist min" + "\t" + "Dist max" + "\t" + "Dist mean" +
            "\t" + "std" + "\t" + "Range" + "\t" + "RMSE")
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

# Evaluation loop
for label in labels:
    for num_config in range(n_configs):
        tgt_filename = ("C:\\Users\\olocc\\OneDrive\\Desktop\\Thesis_new\\Dataset\\For_tests\\Model27\\" +
                        label + str(num_config) + ".ply")

        # Read source and target point clouds
        PC_src = o3d.io.read_point_cloud(src_filename)
        PC_tgt = o3d.io.read_point_cloud(tgt_filename)

        # Compute metrics before registration
        error_def_in = distance_preservation_error(separators, PC_src, PC_src)
        dist_min_in, dist_max_in, dist_mean_in, std_p2p_in, range_p2p_in, rmse_p2p_in = compute_distance_metrics(
            PC_src, PC_tgt
        )
        error_in = compute_couple_metrics(PC_src, PC_tgt)

        # Save initial results into a text file
        with open(filename_pre, "a") as f:
            f.write("\n" + str(error_in) + "\t")
            f.write(str(error_def_in) + "\t")
            f.write(str(dist_min_in) + "\t")
            f.write(str(dist_max_in) + "\t")
            f.write(str(dist_mean_in) + "\t")
            f.write(str(std_p2p_in) + "\t")
            f.write(str(range_p2p_in) + "\t")
            f.write(str(rmse_p2p_in))

        # Homogeneous transform matrix to align the model with the axes of the global reference system
        T_align = np.array([[0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 1]])

        # Initial transform matrix (for point clouds dataset preparation - valid for all point clouds)
        trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0],
                                 [1.0, 0.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0]])

        # Loading base configuration
        src_pc_buffer_start = []
        tgt_pc_buffer = []
        torso_src = o3d.geometry.PointCloud()
        torso_tgt = o3d.geometry.PointCloud()
        for i in range(separators.shape[0]):
            start = int(separators[i, 0])
            end = int(separators[i, 1])
            pc_src = o3d.geometry.PointCloud()
            pc_tgt = o3d.geometry.PointCloud()
            pc_src.points = o3d.utility.Vector3dVector(np.asarray(PC_src.points)[start : end])
            pc_tgt.points = o3d.utility.Vector3dVector(np.asarray(PC_tgt.points)[start: end])
            pc_src.estimate_normals()
            pc_tgt.estimate_normals()
            if i < 5:
                src_pc_buffer_start.append(pc_src)
                tgt_pc_buffer.append(pc_tgt)
            else:
                torso_src += pc_src
                torso_tgt += pc_tgt
        src_pc_buffer_start.append(torso_src)
        tgt_pc_buffer.append(torso_tgt)
        # show_point_clouds(src_pc_buffer_start, tgt_pc_buffer, "Before Registration")

        # Point cloud registrations buffer
        src_pc_buffer_ransac = []
        src_pc_buffer_icp = []

        # Final Transformation Matrix buffer
        T_buffer = []

        # Registration duration
        t_start = time.time()

        for pc_src, pc_tgt in zip(src_pc_buffer_start, tgt_pc_buffer):
            # RANSAC
            voxel_size = scale * 0.005  # means 5 mm for this dataset (min 1.37 mm, max 2.19 cm)
            source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(pc_src, pc_tgt, trans_init, voxel_size)
            result_ransac = execute_global_registration(source_down, target_down,
                                                        source_fpfh, target_fpfh,
                                                        voxel_size)
            pc_src_ransac = copy.deepcopy(pc_src)
            pc_src_ransac.transform(result_ransac.transformation)
            src_pc_buffer_ransac.append(pc_src_ransac)
            # ICP
            result_icp = refine_registration(pc_src, pc_tgt, result_ransac, voxel_size)
            pc_src_icp = copy.deepcopy(pc_src)
            pc_src_icp.transform(result_icp.transformation)
            src_pc_buffer_icp.append(pc_src_icp)
            T_buffer.append(result_icp.transformation)
            # Take the source point cloud to its original pose
            pc_src.transform(inverse_transform(trans_init))

        reg_time = time.time() - t_start
        print("Registration total time: " + str(round(reg_time, 3)) + " s")

        # show_point_clouds(src_pc_buffer_start, tgt_pc_buffer, "Configuration " + str(i +1) + " - Before registration")
        error_in = 1000 * compute_global_metrics(src_pc_buffer_start, tgt_pc_buffer) / scale
        print("Evaluation metric before registration: " + str(error_in) + " mm")

        # show_point_clouds(src_pc_buffer_ransac, tgt_pc_buffer, "After RANSAC")
        # show_point_clouds(src_pc_buffer_icp, tgt_pc_buffer, "After Registration")

        error_out = 1000 * compute_global_metrics(src_pc_buffer_icp, tgt_pc_buffer) / scale
        print("Evaluation metric after registration: " + str(error_out) + " mm")

        # # Inverse test:
        # pc_back_buffer = []
        # for T_icp, tgt_pc in zip(T_buffer, tgt_pc_buffer):
        #     T = T_icp @ trans_init
        #     T_inv = inverse_transform(T)
        #     pc_back = copy.deepcopy(tgt_pc)
        #     pc_back.transform(T_inv)
        #     pc_back_buffer.append(pc_back)
        # show_point_clouds(pc_back_buffer, src_pc_buffer_start, "Registration inverse check")

        # Compute Evaluation metrics
        PC_reg = o3d.geometry.PointCloud()
        for pc in src_pc_buffer_icp:
            PC_reg += pc
        PC_reg.estimate_normals()
        error_def = distance_preservation_error(separators, PC_src, PC_reg)
        dist_min, dist_max, dist_mean, std_p2p, range_p2p, rmse_p2p = compute_distance_metrics(PC_reg, PC_tgt)
        err_couple_out = compute_couple_metrics(PC_reg, PC_tgt)

        # Save results into a text file
        with open(filename_out, "a") as f:
            f.write("\n" + str(reg_time) + "\t")
            f.write(str(err_couple_out) + "\t")
            f.write(str(error_def) + "\t")
            f.write(str(dist_min) + "\t")
            f.write(str(dist_max) + "\t")
            f.write(str(dist_mean) + "\t")
            f.write(str(std_p2p) + "\t")
            f.write(str(range_p2p) + "\t")
            f.write(str(rmse_p2p))