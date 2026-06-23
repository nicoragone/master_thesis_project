import copy
import numpy as np
import open3d as o3d
import os

# series number containing vertebral meshes:
# series = ['14', '20', '21', '25', '27']
series = ['14', '20', '21', '25']
# scale conversion factor
# scales = [769, 763, 750, 642, 680]
scales = [769, 763, 750, 642]
# deformation classes
labels = ['low', 'medium', 'high']

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
    ("L5_S1_IVDjnt" , np.array([0.0 , 0.0 , 0.0])) ,
    ("L4_L5_IVDjnt" , np.array([0.001 , -0.0025 , 0.0])) ,
    ("L3_L4_IVDjnt" , np.array([-0.0006 , -0.0012 , 0.0])) ,
    ("L2_L3_IVDjnt" , np.array([-0.0013 , -0.0039 , 0.0])) ,
    ("L1_L2_IVDjnt" , np.array([0.0014 , -0.0038 , 0.0])) ,
    ("torsojnt" , np.array([0.0 , 0.0 , 0.0]))
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
def generate_pose(src_buffer_start, bodies, label, num_points, scale):
    transformed_pc = o3d.geometry.PointCloud()
    vector_field = np.zeros((num_points, 3))
    j = 0
    if label == "low":
        angle_total_deg_fe = np.random.randint(-23, 7)
        angle_total_deg_lb =np.random.randint(-8, 8)
        angle_total_deg_ar = np.random.randint(-15, 15)
    elif label == "medium":
        angle_total_deg_fe = np.random.choice(np.concatenate((np.arange(-46, -23), np.arange(7,14))))
        angle_total_deg_lb = np.random.choice([-1, 1]) * np.random.randint(8, 16)
        angle_total_deg_ar = np.random.choice([-1, 1]) * np.random.randint(15, 30)
    else:
        angle_total_deg_fe = np.random.choice(np.concatenate((np.arange(-70, -46), np.arange(14, 20))))
        angle_total_deg_lb = np.random.choice([-1, 1]) * np.random.randint(16, 25)
        angle_total_deg_ar = np.random.choice([-1, 1]) * np.random.randint(30, 45)
    T_acc_global = np.eye(4)
    for pc, name in zip(src_buffer_start, bodies):
        new_pc = copy.deepcopy(pc)
        # Rotation weights
        weight_fe = weights_fe[name]
        weight_lb = weights_lb[name]
        weight_ar = weights_ar[name]
        # Joint and mesh transformation
        center = pc.get_center()
        T_mesh = homogeneous_transformation_matrix(0, 0, 0, center)
        joint = Joints[name]
        T_joint_body = homogeneous_transformation_matrix(*Orientation[joint], scale * Location[joint])
        # Joint absolute pose
        T_joint = T_mesh @ T_joint_body
        # Local rotations
        angle_rad_fe = np.radians(weight_fe * angle_total_deg_fe)
        angle_rad_lb = np.radians(weight_lb * angle_total_deg_lb)
        angle_rad_ar = np.radians(weight_ar * angle_total_deg_ar)
        # Global rotations and translations
        T_rotation = homogeneous_transformation_matrix(angle_rad_lb, angle_rad_ar, angle_rad_fe,
                                                       [0, 0, 0])
        T_to_origin = np.eye(4)
        T_to_origin[:3, :3] = T_joint[:3, :3].T
        T_to_origin[:3, 3] = -T_joint[:3, :3].T @ T_joint[:3, 3]
        T_local = T_joint @ T_rotation @ T_to_origin
        T_acc_global = T_acc_global @ T_local
        # Apply transformation
        new_pc.transform(T_acc_global)
        new_pc.estimate_normals()
        flow = np.zeros((np.asarray(new_pc.points).shape[0], 3))
        for i in range(flow.shape[0]):
            flow[i, :] = np.asarray(new_pc.points)[i, :] - np.asarray(pc.points)[i, :]
        vector_field[j : j + (np.asarray(new_pc.points)).shape[0], :] = flow
        j += (np.asarray(new_pc.points)).shape[0]
        transformed_pc += new_pc
    return transformed_pc, angle_total_deg_fe, angle_total_deg_lb, angle_total_deg_ar, vector_field

def downsample_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return pcd_down


# Main cicle
for serie, scale in zip(series, scales):
    folder_src = "C:\\Users\\olocc\\OneDrive\\Desktop\\Thesis_new\\Dataset\\Base_meshes\\Model" + serie
    bodies = [name for name, _ in rotation_weights_fe]
    src_buffer_start = []
    T_align = np.array([[0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1]])

    # Loading base configuration
    num_points = 0
    SRC_pc = o3d.geometry.PointCloud()
    for name in bodies:
        filename_src = folder_src + "/" + name + ".ply"
        # Starting point clouds (unique point cloud) and meshes
        pc = o3d.io.read_point_cloud(filename_src)
        pc.transform(T_align)
        pc.estimate_normals()
        # Downsample source point cloud, in a way to have a max point number 10000
        voxel_size = scale * 0.005
        pc_down = downsample_point_cloud(pc, voxel_size)
        num_points += np.asarray(pc_down.points).shape[0]
        src_buffer_start.append(pc_down)
        SRC_pc += pc_down
    output_folder = "C:\\Users\\olocc\\OneDrive\\Desktop\\Thesis_new\\Dataset\\Clouds_for_lepard\\Model" + serie
    os.makedirs(output_folder, exist_ok=True)
    # Save source point cloud
    filename_src_out = output_folder + "//base.ply"
    o3d.io.write_point_cloud(filename_src_out, SRC_pc)

    # Generate deformed configuration
    for label in labels:
        num_configs = 600
        for i in range(num_configs):
            tgt_pc, theta_fe, theta_lb, theta_ar, vector_field = generate_pose(src_buffer_start, bodies, label,
                                                                               num_points, scale)
            filename_tgt_out = output_folder + "//" + label + str(i) + ".ply"
            o3d.io.write_point_cloud(filename_tgt_out, tgt_pc)
            filename_info = filename_tgt_out
            idx = filename_info.find(".ply")
            filename_info = filename_info[:idx] + "_Info.txt"
            with open(filename_info, "a") as f:
                f.write("Theta_fe[°] = " + str(theta_fe) + "\n")
                f.write("Theta_lb[°] = " + str(theta_lb) + "\n")
                f.write("Theta_ar[°] = " + str(theta_ar) + "\n\n")
                f.write("Vector filed [mm] :" + "\n")
                for j in range(vector_field.shape[0]):
                    f.write(str(vector_field[j, 0]) + "\t")
                    f.write(str(vector_field[j, 1]) + "\t")
                    f.write(str(vector_field[j, 2]) + "\n")