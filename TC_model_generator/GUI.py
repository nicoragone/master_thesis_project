import open3d as o3d
import numpy as np
import os

input_folder = r"C:\Users\olocc\OneDrive\Desktop\Thesis_new\Dataset\Base_meshes\Model27"
output_base = r"C:\Users\olocc\OneDrive\Desktop\Thesis_new\\"

# angles initialization for the first view
angle_total_deg_fe = 0
angle_total_deg_lb = 0
angle_total_deg_ar = 0

# translations initialization
posX = 0.0
posY = 0.0
posZ = 0.0

# global angles initialization
alpha = 0
beta = 0
gamma = 0

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
    ("L5_S1_IVDjnt" , [0.0 , 0.0 , 0.0]) ,
    ("L4_L5_IVDjnt" , [0.001 , -0.0025 , 0.0]) ,
    ("L3_L4_IVDjnt" , 1020.0 * np.array([-0.0006 , -0.0012 , 0.0])) ,
    ("L2_L3_IVDjnt" , 1020.0 * np.array([-0.0013 , -0.0039 , 0.0])) ,
    ("L1_L2_IVDjnt" , 1020.0 * np.array([0.0014 , -0.0038 , 0.0])) ,
    ("torsojnt" , [0.0 , 0.0 , 0.0])
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

# updating mesh on button pressure
def update_meshes(vis):
    global angle_total_deg_fe, meshes, T_acc_global, T_align, alpha, beta, gamma
    first = True
    # cumulative transform reset
    T_acc_global = np.eye(4)
    for mesh, name in zip(meshes, bodies):
        # geometry reset
        mesh.clear()

        # reload mesh fresh to avoid cumulative transformations
        mesh_path = input_folder + "/" + name + ".ply"
        new_mesh = o3d.io.read_triangle_mesh(mesh_path)
        new_mesh.compute_vertex_normals()
        new_mesh.compute_triangle_normals()
        new_mesh.transform(T_align)

        # rotation weights
        weight_fe = weights_fe[name]
        weight_lb = weights_lb[name]
        weight_ar = weights_ar[name]

        # Body reference system global pose
        center = new_mesh.get_center()
        T_mesh = homogeneous_transformation_matrix(0, 0, 0, center)

        # Joint global pose (take the joint back to its position)
        joint = Joints[name]
        angles_joint_body = Orientation[joint]
        position_joint_body = Location[joint]
        T_joint_body = homogeneous_transformation_matrix(angles_joint_body[0], angles_joint_body[1], angles_joint_body[2], position_joint_body)
        T_joint = T_mesh @ T_joint_body

        # Rotation around the axis of the reference system of the joint
        angle_rad_fe = np.radians(weight_fe * angle_total_deg_fe)
        angle_rad_lb = np.radians(weight_lb * angle_total_deg_lb)
        angle_rad_ar = np.radians(weight_ar * angle_total_deg_ar)

        # Global rotation angles
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        gamma_rad = np.radians(gamma)

        # Adding global roto-translation of the whole body (only for the first body)
        if first:
            T_rotation = homogeneous_transformation_matrix(angle_rad_lb + alpha_rad, angle_rad_ar + beta_rad, angle_rad_fe + gamma_rad, [posX, posY, posZ])
            first = False
        else:
            T_rotation = homogeneous_transformation_matrix(angle_rad_lb, angle_rad_ar, angle_rad_fe, np.zeros(3))

        # take the joint to the origin of the global reference system (inverse transformation)
        T_to_origin = np.eye(4)
        T_to_origin[:3, :3] = T_joint[:3, :3].T
        T_to_origin[:3, 3] = -T_joint[:3, :3].T @ T_joint[:3, 3]

        T_back = T_joint

        # Local and cumulative transformation
        T_local = T_back @ T_rotation @ T_to_origin
        T_acc_global = T_acc_global @ T_local
        new_mesh.transform(T_acc_global)

        # mesh updating into the scene
        mesh.vertices = new_mesh.vertices
        mesh.triangles = new_mesh.triangles
        mesh.vertex_normals = new_mesh.vertex_normals
        mesh.triangle_normals = new_mesh.triangle_normals
        mesh.vertex_colors = new_mesh.vertex_colors

    for mesh in meshes:
        vis.update_geometry(mesh)

    return False

def increase_fe(vis):
    global angle_total_deg_fe
    angle_total_deg_fe = min(angle_total_deg_fe + 1, 20)  # upper limit
    print("Flex-extension angle: " + str(angle_total_deg_fe) + "°")
    update_meshes(vis)
    return False

def decrease_fe(vis):
    global angle_total_deg_fe
    angle_total_deg_fe = max(angle_total_deg_fe - 1, -70)  # lower limit
    print("Flex-extension angle: " + str(angle_total_deg_fe) + "°")
    update_meshes(vis)
    return False

def increase_lb(vis):
    global angle_total_deg_lb
    angle_total_deg_lb = min(angle_total_deg_lb + 1, 25)  # upper limit
    print("Lateral bending angle: " + str(angle_total_deg_lb) + "°")
    update_meshes(vis)
    return False

def decrease_lb(vis):
    global angle_total_deg_lb
    angle_total_deg_lb = max(angle_total_deg_lb - 1, -25)  # lower limit
    print("Lateral Bending angle: " + str(angle_total_deg_lb) + "°")
    update_meshes(vis)
    return False

def increase_ar(vis):
    global angle_total_deg_ar
    angle_total_deg_ar = min(angle_total_deg_ar + 1, 45)  # upper limit
    print("Axial Rotation angle: " + str(angle_total_deg_ar) + "°")
    update_meshes(vis)
    return False

def decrease_ar(vis):
    global angle_total_deg_ar
    angle_total_deg_ar = max(angle_total_deg_ar - 1, -45)  # lower limit
    print("Axial Rotation angle: " + str(angle_total_deg_ar) + "°")
    update_meshes(vis)
    return False

def increase_X_position(vis):
    global posX
    posX = min(posX + 0.01, 1) # upper limit
    print("Position along X: " + str(round(posX, 2)) + " m")
    update_meshes(vis)
    return False

def decrease_X_position(vis):
    global posX
    posX = max(posX - 0.01, -1) # lower limit
    print("Position along X: " + str(round(posX, 2)) + " m")
    update_meshes(vis)
    return False

def increase_Y_position(vis):
    global posY
    posY = min(posY + 0.01, 1) # upper limit
    print("Position along Y: " + str(round(posY, 2)) + " m")
    update_meshes(vis)
    return False

def decrease_Y_position(vis):
    global posY
    posY = max(posY - 0.01, -1) # lower limit
    print("Position along Y: " + str(round(posY, 2)) + " m")
    update_meshes(vis)
    return False

def increase_Z_position(vis):
    global posZ
    posZ = min(posZ + 0.01, 3) # upper limit
    print("Position along Z: " + str(round(posZ, 2)) + " m")
    update_meshes(vis)
    return False

def decrease_Z_position(vis):
    global posZ
    posZ = max(posZ - 0.01, 0) # lower limit
    print("Position along Z: " + str(round(posZ, 2)) + " m")
    update_meshes(vis)
    return False

def increase_angleX(vis):
    global alpha
    alpha = min(alpha + 1, 180) # upper limit
    print("Angle X: " + str(alpha) + " °")
    update_meshes(vis)
    return False

def decrease_angleX(vis):
    global alpha
    alpha = max(alpha - 1, -180) # lower limit
    print("Angle X: " + str(alpha) + " °")
    update_meshes(vis)
    return False

def increase_angleY(vis):
    global beta
    beta = min(beta + 1, 180) # upper limit
    print("Angle Y: " + str(beta) + " °")
    update_meshes(vis)
    return False

def decrease_angleY(vis):
    global beta
    beta = max(beta - 1, -180) # lower limit
    print("Angle Y: " + str(beta) + " °")
    update_meshes(vis)
    return False

def increase_angleZ(vis):
    global gamma
    gamma = min(gamma + 1, 180) # upper limit
    print("Angle Z: " + str(gamma) + " °")
    update_meshes(vis)
    return False

def decrease_angleZ(vis):
    global gamma
    gamma = max(gamma - 1, -180) # lower limit
    print("Angle Z: " + str(gamma) + " °")
    update_meshes(vis)
    return False

def save_mesh(vis):
    # Insert the name of the destination folder
    output_folder = str(input("Insert the name of the output folder: "))
    os.makedirs(output_base + output_folder, exist_ok=True)
    # Combing all the mesh into a unique mesh
    combined_mesh = o3d.geometry.TriangleMesh()
    for name, mesh in zip(bodies, meshes):
        # Save single meshes firstly
        filename_mesh_out = output_base + output_folder + "/" + name + ".ply"
        o3d.io.write_triangle_mesh(filename_mesh_out, mesh)
        combined_mesh += mesh
    combined_mesh.compute_vertex_normals()
    combined_mesh.compute_triangle_normals()
    # Saving mesh in a .ply file
    output_file = output_base + output_folder + "/" + output_folder + "_merged.ply"
    o3d.io.write_triangle_mesh(output_file, combined_mesh)
    print("Mesh saved to: " + output_file)

# first mesh loading
bodies = [name for name, _ in rotation_weights_fe]
meshes = []
# Align meshes to global reference system
T_align = np.array([[0, 1, 0, 0],
              [0, 0, 1, 0],
              [1, 0, 0, 0],
              [0, 0, 0, 1]])
for name in bodies:
    mesh_path = input_folder + "/" + name + ".ply"
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.transform(T_align)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    meshes.append(mesh)

# resetting cumulative transform for each update
T_acc_global = np.eye(4)

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()
# Setting dark background
render_option = vis.get_render_option()
render_option.background_color = np.array([0.1, 0.1, 0.2])

for mesh in meshes:
    vis.add_geometry(mesh)

# Showing global reference system (X red , Y green, Z blue)
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=75, origin=[0,0,0])
vis.add_geometry(axis)

# execute transformation at the button pressure
vis.register_key_callback(265, increase_fe)  # up arrow
vis.register_key_callback(264, decrease_fe)  # down arrow
vis.register_key_callback(263, increase_lb) # right arrow
vis.register_key_callback(262, decrease_lb) # left arrow
vis.register_key_callback(334, increase_ar) # + of the numerical keyboard
vis.register_key_callback(333, decrease_ar) # - of the numerical keyboard
vis.register_key_callback(70, increase_X_position) # F
vis.register_key_callback(66, decrease_X_position) # B
vis.register_key_callback(85, increase_Y_position) # U
vis.register_key_callback(68, decrease_Y_position) # D
vis.register_key_callback(82, increase_Z_position) # R
vis.register_key_callback(76, decrease_Z_position) # L
vis.register_key_callback(88, increase_angleX) # X
vis.register_key_callback(65, decrease_angleX) # A
vis.register_key_callback(89, increase_angleY) # Y
vis.register_key_callback(67, decrease_angleY) # C
vis.register_key_callback(90, increase_angleZ) # Z
vis.register_key_callback(71, decrease_angleZ) # G

vis.register_key_callback(257, save_mesh) # enter

# Draw the initial scene
update_meshes(vis)

vis.run()
vis.destroy_window()