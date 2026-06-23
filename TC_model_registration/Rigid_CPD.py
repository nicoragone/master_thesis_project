import copy
import numpy as np
import open3d as o3d
from probreg import cpd
import matplotlib.pyplot as plt
import time

# scale conversion factor
scale = 1020.0

# flex_extension weights
rotation_weights_fe = [
    ("L5", 0.125),
    ("L4", 0.185),
    ("L3", 0.204),
    ("L2", 0.231),
    ("L1", 0.255),
    ("Torso" , 0.0)
]
weights_fe = dict(rotation_weights_fe)

#lateral_bending weights
rotation_weights_lb = [
    ("L5", 0.1356),
    ("L4", 0.1812),
    ("L3", 0.2452),
    ("L2", 0.25),
    ("L1", 0.188),
    ("Torso" , 0.0)
]
weights_lb = dict(rotation_weights_lb)

# axial_rotation weights
rotation_weights_ar = [
    ("L5", 0.355556),
    ("L4", 0.037778),
    ("L3", 0.037778),
    ("L2", 0.0311111),
    ("L1", 0.0288889),
    ("Torso" , 0.0)
]
weights_ar = dict(rotation_weights_ar)

# associate a joint to each body
joints = [
    ("L5" , "L5_S1_IVDjnt") ,
    ("L4" , "L4_L5_IVDjnt") ,
    ("L3" , "L3_L4_IVDjnt") ,
    ("L2" , "L2_L3_IVDjnt") ,
    ("L1" , "L1_L2_IVDjnt") ,
    ("Torso" , "torsojnt")
]
Joints = dict(joints)

# relative position of the joint with respect to the body
location = [
    ("L5_S1_IVDjnt" , scale * np.array([0.0 , 0.0 , 0.0])) ,
    ("L4_L5_IVDjnt" , scale * np.array([0.001 , -0.0025 , 0.0])) ,
    ("L3_L4_IVDjnt" , scale * np.array([-0.0006 , -0.0012 , 0.0])) ,
    ("L2_L3_IVDjnt" , scale * np.array([-0.0013 , -0.0039 , 0.0])) ,
    ("L1_L2_IVDjnt" , scale * np.array([0.0014 , -0.0038 , 0.0])) ,
    ("torsojnt" , scale * np.array([0.0 , 0.0 , 0.0]))
]
Location = dict(location)

# relative orientation of the joint with respect to the body
orientation = [
    ("L5_S1_IVDjnt" , [0.0 , 0.0 , 0.0]) ,
    ("L4_L5_IVDjnt" , [0.0 , 0.0 , 0.0]) ,
    ("L3_L4_IVDjnt" , [0.0 , 0.0 , 0.0]) ,
    ("L2_L3_IVDjnt" , [0.0 , 0.0 , 0.0]) ,
    ("L1_L2_IVDjnt" , [0.0 , 0.0 , 0.0]) ,
    ("torsojnt" , [0.0 , 0.0 , 0.0])
]
Orientation = dict(orientation)

def rotation_matrix(angleX , angleY, angleZ):
    # lateral bending
    c, s = np.cos(angleX), np.sin(angleX)
    Rx = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])
    # axial rotation
    c, s = np.cos(angleY), np.sin(angleY)
    Ry = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])
    # flex extension
    c, s = np.cos(angleZ), np.sin(angleZ)
    Rz = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    R = Rx @ Ry @ Rz
    return R

def homogeneous_transformation_matrix(angleX , angleY , angleZ, position):
    T = np.eye(4)
    T[:3, :3] = rotation_matrix(angleX , angleY , angleZ)
    T[:3, 3] = position
    return T

# Pose generator
def generate_pose(transformed_pcs):
    global src_pc_buffer_start, bodies, scale
    angle_total_deg_fe = np.random.randint(-70, 20)
    angle_total_deg_lb = np.random.randint(-25, 25)
    angle_total_deg_ar = np.random.randint(-45, 45)
    posX = np.random.uniform(-1.0, 1.0) * scale
    posY = np.random.uniform(-1.0, 1.0) * scale
    posZ = np.random.uniform(-1.0, 1.0) * scale
    alpha = np.random.randint(-180, 180)
    beta = np.random.randint(-180, 180)
    gamma = np.random.randint(-180, 180)
    T_acc_global = np.eye(4)
    first = True
    for pc, name in zip(src_pc_buffer_start, bodies):
        new_pc = copy.deepcopy(pc)
        # Rotation weights
        weight_fe = weights_fe[name]
        weight_lb = weights_lb[name]
        weight_ar = weights_ar[name]
        # Joint and mesh transformation
        center = pc.get_center()
        T_mesh = homogeneous_transformation_matrix(0, 0, 0, center)
        joint = Joints[name]
        T_joint_body = homogeneous_transformation_matrix(*Orientation[joint], Location[joint])
        # Joint absolute pose
        T_joint = T_mesh @ T_joint_body
        # Local rotations
        angle_rad_fe = np.radians(weight_fe * angle_total_deg_fe)
        angle_rad_lb = np.radians(weight_lb * angle_total_deg_lb)
        angle_rad_ar = np.radians(weight_ar * angle_total_deg_ar)
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        gamma_rad = np.radians(gamma)
        # Global rotations and translations
        if first:
            T_rotation = homogeneous_transformation_matrix(
                angle_rad_lb + alpha_rad,
                angle_rad_ar + beta_rad,
                angle_rad_fe + gamma_rad,
                [posX, posY, posZ]
            )
            first = False
        else:
            T_rotation = homogeneous_transformation_matrix(angle_rad_lb, angle_rad_ar, angle_rad_fe, [0, 0, 0])
        T_to_origin = np.eye(4)
        T_to_origin[:3, :3] = T_joint[:3, :3].T
        T_to_origin[:3, 3] = -T_joint[:3, :3].T @ T_joint[:3, 3]
        T_local = T_joint @ T_rotation @ T_to_origin
        T_acc_global = T_acc_global @ T_local
        # Apply transformation
        new_pc.transform(T_acc_global)
        new_pc.estimate_normals()
        transformed_pcs.append(new_pc)
    return transformed_pcs

# Load point clous from a .ply mesh file
def load_point_cloud(filename):
    return o3d.io.read_point_cloud(filename)

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

# RANSAC preliminary registration
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

# Plot point clouds
def show_point_clouds(src_buffer, tgt_buffer, title):
    vis = o3d.visualization.Visualizer()
    vis.create_window(title, 1400, 1050)
    for pc in src_buffer:
        pc.paint_uniform_color([1, 0.706, 0])
        vis.add_geometry(pc)
    for pc in tgt_buffer:
        pc.paint_uniform_color([0, 0.651, 0.929])
        vis.add_geometry(pc)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.141, 0.157, 0.239])
    vis.run()
    vis.destroy_window()

def plot_metrics(index, labels, title):
    plt.bar(labels, index, 0.5, color = "skyblue", edgecolor = "k")
    plt.grid(True)
    plt.xlabel("Body")
    plt.ylabel("Value [m]")
    plt.title(title)
    plt.show()

# Get homogeneous transform matrix from RigidTransformation object
def get_homogeneous_transformation_matrix(rigid_transform):
    T = np.eye(4)
    T[:3, :3] = rigid_transform.rot
    T[:3, 3] = rigid_transform.t
    return T

# Compute inverse transformation matrix
def inverse_transform(T):
    T_inv = np.eye(4)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return T_inv

# Evaluation metrics
def compute_fre(fiducials_src_pc, fiducials_tgt_pc, T):
    fiducials_src = np.asarray(fiducials_src_pc.points)
    fiducials_tgt = np.asarray(fiducials_tgt_pc.points)
    fid_src_transformed = (T[:3, :3] @ fiducials_src.T + T[:3, 3:4]).T
    errors = np.sqrt((fid_src_transformed[:, 0] - fiducials_tgt[:, 0])**2 +
                     (fid_src_transformed[:, 1] - fiducials_tgt[:, 1])**2 +
                     (fid_src_transformed[:, 2] - fiducials_tgt[:, 2]) ** 2)
    return np.mean(errors)

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

folder_src = r"C:\Users\olocc\OneDrive\Desktop\Thesis_new\Base_pose_img27"

# Initial transformation matrix (align point clouds to global reference system)
T_align = np.array([[0, -1, 0, 0],
                    [0, 0, 1, 0],
                    [-1, 0, 0, 0],
                    [0, 0, 0, 1]])

bodies = [name for name, _ in rotation_weights_fe]
src_pc_buffer_start = []
# Loading base configuration
for name in bodies:
    filename_src = folder_src + "/" + name + ".ply"
    # Starting point clouds
    pc_src = load_point_cloud(filename_src)
    pc_src.transform(T_align)
    src_pc_buffer_start.append(pc_src)

n = int(input("Insert the number of configurations: "))

for i in range(n):
    # Point cloud buffer
    src_pc_buffer_end = []
    src_pc_ransac = []
    tgt_pc_buffer = []
    # Final Transformation Matrix buffer
    T_buffer = []
    # Error metrics buffer
    fre_buffer = []

    voxel_size = scale * 0.005

    # Registration duration
    t_start = time.time()

    tgt_pc_buffer = generate_pose(tgt_pc_buffer)

    for pc_src, pc_tgt in zip(src_pc_buffer_start, tgt_pc_buffer):
        src_down, src_fpfh = preprocess_point_cloud(copy.deepcopy(pc_src), voxel_size)
        tgt_down, tgt_fpfh = preprocess_point_cloud(copy.deepcopy(pc_tgt), voxel_size)
        result_ransac = execute_global_registration(src_down, tgt_down,
                                                    src_fpfh, tgt_fpfh,
                                                    voxel_size)
        src_ransac = copy.deepcopy(src_down)
        src_ransac.transform(result_ransac.transformation)
        src_pc_ransac.append(src_ransac)
        tf_param, _, _ = cpd.registration_cpd(src_ransac, tgt_down, tf_type_name="rigid")
        result = copy.deepcopy(src_ransac)
        result.transform(get_homogeneous_transformation_matrix(tf_param))
        src_pc_buffer_end.append(copy.deepcopy(result))

        # Fiducial Registration Error
        T = get_homogeneous_transformation_matrix(tf_param) @ result_ransac.transformation
        fre_buffer.append(compute_fre(pc_src, pc_tgt, T) / scale)

        # Save total transformation matrix
        T_buffer.append(T)

    print("Registration " + str(i + 1) + " completed in " + str(round(time.time() - t_start, 3)) + " s")

    # Initial point clouds
    show_point_clouds(src_pc_buffer_start, tgt_pc_buffer, "Registration " + str(i + 1) + " - Before registration")
    glb_dist_metric = compute_global_metrics(src_pc_buffer_start, tgt_pc_buffer) / scale
    print("Registration " + str(i + 1) + " - Evaluation metric before registration: " + str(glb_dist_metric) + " m")

    # RANSAC pre-alignement result
    show_point_clouds(src_pc_ransac, tgt_pc_buffer, "Registration " + str(i + 1) + " - After RANSAC")

    # Registration result
    show_point_clouds(src_pc_buffer_end, tgt_pc_buffer, "Registration " + str(i + 1) + " - After CPD registration")

    # Registration evaluation
    plot_metrics(fre_buffer, bodies, "Configuration " + str(i + 1) + " - FRE bar graph")

    glb_dist_metric = compute_global_metrics(src_pc_buffer_end, tgt_pc_buffer) / scale
    print("Registration " + str(i + 1) + " - Global evaluation metric: " + str(glb_dist_metric) + " m")

    # Inverse test
    pc_back_buffer = []
    for T, pc_end in zip(T_buffer, tgt_pc_buffer):
        T_inv = inverse_transform(T)
        pc_back = copy.deepcopy(pc_end)
        pc_back.transform(T_inv)
        pc_back_buffer.append(pc_back)
    show_point_clouds(pc_back_buffer, src_pc_buffer_start,"Registration " + str(i + 1) + " - Registration inverse check")