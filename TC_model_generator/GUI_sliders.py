import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import os

input_folder = r"C:\Users\olocc\OneDrive\Desktop\Thesis_new\Dataset\Base_meshes\Model27"
output_base = r"C:\Users\olocc\OneDrive\Desktop\Thesis_new\\"

# scale comparison
scale = 680

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
    ("Lr5", 0.355556),
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
    ("gndpelvis" , [0.0 , 0.0 , 0.0]) ,
    ("gndsacrum" , [0.0 , 0.0 , 0.0]) ,
    ("L5_S1_IVDjnt" , [0.0 , 0.0 , 0.0]) ,
    ("L4_L5_IVDjnt" , [0.001 , -0.0025 , 0.0]) ,
    ("L3_L4_IVDjnt" , [-0.0006 , -0.0012 , 0.0]) ,
    ("L2_L3_IVDjnt" , [-0.0013 , -0.0039 , 0.0]) ,
    ("L1_L2_IVDjnt" , [0.0014 , -0.0038 , 0.0]) ,
    ("torsojnt" , [0.0 , 0.0 , 0.0])
]
Location = dict(location)

# relative orientation of the joint with respect to the body
orientation = [
    ("gndpelvis" , [0.0 , 0.0 , 0.0]) ,
    ("gndsacrum" , [0.0 , 0.0 , 0.0]) ,
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

# Mesh updating into scenes (values setted by sliders)
def update_meshes():
    global angle_total_deg_fe, angle_total_deg_lb, angle_total_deg_ar
    global posX, posY, posZ, alpha, beta, gamma
    global meshes, bodies, original_vertices
    global scale
    T_acc_global = np.eye(4)
    first = True
    for i, (mesh, name) in enumerate(zip(meshes, bodies)):
        vertices_orig = original_vertices[i]
        # Rotation weights
        weight_fe = weights_fe[name]
        weight_lb = weights_lb[name]
        weight_ar = weights_ar[name]
        # Joint and mesh transformation
        center = np.mean(vertices_orig, axis=0)
        T_mesh = homogeneous_transformation_matrix(0,0,0,center)
        joint = Joints[name]
        T_joint_body = homogeneous_transformation_matrix(*Orientation[joint], scale * Location[joint])
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
                scale * [posX, posY, posZ]
            )
            first = False
        else:
            T_rotation = homogeneous_transformation_matrix(angle_rad_lb, angle_rad_ar, angle_rad_fe, [0,0,0])
        T_to_origin = np.eye(4)
        T_to_origin[:3,:3] = T_joint[:3,:3].T
        T_to_origin[:3,3] = -T_joint[:3,:3].T @ T_joint[:3,3]
        T_local = T_joint @ T_rotation @ T_to_origin
        T_acc_global = T_acc_global @ T_local
        # Apply transformation
        vertices_h = np.hstack([vertices_orig, np.ones((vertices_orig.shape[0],1))])
        vertices_transformed = (T_acc_global @ vertices_h.T).T[:, :3]
        mesh.vertices = o3d.utility.Vector3dVector(vertices_transformed)
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        # Update geometry into the scene
        scene.scene.remove_geometry(name)
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        scene.scene.add_geometry(name, mesh, mat)

# Setup scene and camera
def setup_scene(factor):
    scene.scene.clear_geometry()
    # Insert bodies into the scene
    for name, mesh in zip(bodies, meshes):
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        scene.scene.add_geometry(name, mesh, mat)
    # Create global reference system
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2 * factor, origin=[0, 0, 0])
    mat_axis = rendering.MaterialRecord()
    mat_axis.shader = "defaultUnlit"  # Usa unlit per gli assi (nessuna illuminazione)
    scene.scene.add_geometry("axis", axis, mat_axis)
    scene.scene.set_background([0.1, 0.1, 0.2, 1.0])
    # Compute scen bounds
    all_geometries = []
    for mesh in meshes:
        all_geometries.append(mesh)
    all_geometries.append(axis)
    all_vertices = np.vstack([np.asarray(geom.vertices) for geom in all_geometries])
    bounds = o3d.geometry.AxisAlignedBoundingBox(
        all_vertices.min(axis=0), all_vertices.max(axis=0)
    )
    center = bounds.get_center()
    # Adapt camera to scene bounds
    scene.setup_camera(60.0, bounds, center)
    scene.scene.camera.look_at(center, center + [factor, 0, 0.6 * factor], [0, 0, 1])

# Callback sliders:

# Flex-extension
def on_fe_changed(value):
    global angle_total_deg_fe
    angle_total_deg_fe = int(value)
    print("Flex-extension angle: " + str(int(value)) + "°")
    update_meshes()

# Lateral bending
def on_lb_changed(value):
    global angle_total_deg_lb
    angle_total_deg_lb = int(value)
    print("Lateral bending angle: " + str(int(value)) + "°")
    update_meshes()

# Axial rotation
def on_ar_changed(value):
    global angle_total_deg_ar
    angle_total_deg_ar = int(value)
    print("Axial rotation angle: " + str(int(value)) + "°")
    update_meshes()

# Translation along X
def on_X_changed(value):
    global posX
    posX = value
    print("Translation along X: " + str(round(value, 3)) + " m")
    update_meshes()

# Translation along Y
def on_Y_changed(value):
    global posY
    posY = value
    print("Translation along Y: " + str(round(value, 3)) + " m")
    update_meshes()

# Translation along Z
def on_Z_changed(value):
    global posZ
    posZ = value
    print("Translation along Z: " + str(round(value, 3)) + " m")
    update_meshes()

# Global rotation around X
def on_alpha_changed(value):
    global alpha
    alpha = int(value)
    print("Global rotation around X: " + str(int(value)) + "°")
    update_meshes()

# Global rotation around Y
def on_beta_changed(value):
    global beta
    beta = int(value)
    print("Global rotation around Y: " + str(int(value)) + "°")
    update_meshes()

# Global rotation around Z
def on_gamma_changed(value):
    global gamma
    gamma = int(value)
    print("Global rotation around Z: " + str(int(value)) + "°")
    update_meshes()

# Reset view and basic Pose
def on_reset_clicked():
    global fe_slider, lb_slider, ar_slider, rX_slider, rY_slider, rZ_slider, tX_slider, tY_slider, tZ_slider
    global angle_total_deg_fe, angle_total_deg_lb, angle_total_deg_ar, alpha, beta, gamma, posX, posY, posZ
    global meshes
    for i, mesh in enumerate(meshes):
        mesh.vertices = o3d.utility.Vector3dVector(original_vertices[i].copy())
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
    fe_slider.int_value = 0
    lb_slider.int_value = 0
    ar_slider.int_value = 0
    rX_slider.int_value = 0
    rY_slider.int_value = 0
    rZ_slider.int_value = 0
    tX_slider.int_value = 0
    tY_slider.int_value = 0
    tZ_slider.int_value = 0
    angle_total_deg_fe = 0
    angle_total_deg_lb = 0
    angle_total_deg_ar = 0
    alpha = 0
    beta = 0
    gamma = 0
    posX = 0.0
    posY = 0.0
    posZ = 0.0
    setup_scene(scale)

# Save meshes
def on_save():
    global bodies, meshes
    global output_base
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

# Manual Layout: full screen scene, right floating panel
def on_layout(layout_context):
    r = window.content_rect
    scene.frame = r
    # panel on the top right
    pref = panel.calc_preferred_size(layout_context, gui.Widget.Constraints())
    margin = 10
    panel.frame = gui.Rect(r.get_right() - pref.width - margin, r.y + margin, pref.width, pref.height)

# Manage mouse clicks
def on_mouse_event(event):
    return gui.SceneWidget.EventCallbackResult.IGNORED

# First mesh loading
bodies = [name for name, _ in rotation_weights_fe]
meshes = []
for name in bodies:
    mesh_path = input_folder + "/" + name + ".ply"
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    meshes.append(mesh)

# Get original vertices (for mesh updating)
original_vertices = [np.asarray(mesh.vertices).copy() for mesh in meshes]

# Create an empty GUI
app = gui.Application.instance
app.initialize()

# Create a window
window = app.create_window("Spine Viewer", 1200, 900)

# SceneWidget
scene = gui.SceneWidget()
scene.scene = rendering.Open3DScene(window.renderer)
window.add_child(scene)

# Floating panel
panel = gui.Vert()
# Sliders
panel.add_child(gui.Label("Flex-Extension [°]"))
fe_slider = gui.Slider(gui.Slider.INT)
fe_slider.set_limits(-70, 20)
fe_slider.int_value = 0
panel.add_child(fe_slider)
panel.add_child(gui.Label("")) # Empty space between two consecutive sliders
panel.add_child(gui.Label("Lateral Bending [°]"))
lb_slider = gui.Slider(gui.Slider.INT)
lb_slider.set_limits(-25, 25)
lb_slider.int_value = 0
panel.add_child(lb_slider)
panel.add_child(gui.Label(""))
panel.add_child(gui.Label("Axial Rotation [°]"))
ar_slider = gui.Slider(gui.Slider.INT)
ar_slider.set_limits(-45, 45)
ar_slider.int_value = 0
panel.add_child(ar_slider)
panel.add_child(gui.Label(""))
panel.add_child(gui.Label("Translation X [m]"))
tX_slider = gui.Slider(gui.Slider.DOUBLE)
tX_slider.set_limits(-1, 1)
tX_slider.int_value = 0
panel.add_child(tX_slider)
panel.add_child(gui.Label(""))
panel.add_child(gui.Label("Translation Y [m]"))
tY_slider = gui.Slider(gui.Slider.DOUBLE)
tY_slider.set_limits(-1, 1)
tY_slider.int_value = 0
panel.add_child(tY_slider)
panel.add_child(gui.Label(""))
panel.add_child(gui.Label("Translation Z [m]"))
tZ_slider = gui.Slider(gui.Slider.DOUBLE)
tZ_slider.set_limits(-1, 1)
tZ_slider.int_value = 0
panel.add_child(tZ_slider)
panel.add_child(gui.Label(""))
panel.add_child(gui.Label("Angle X [°]"))
rX_slider = gui.Slider(gui.Slider.INT)
rX_slider.set_limits(-180, 180)
rX_slider.int_value = 0
panel.add_child(rX_slider)
panel.add_child(gui.Label(""))
panel.add_child(gui.Label("Angle Y [°]"))
rY_slider = gui.Slider(gui.Slider.INT)
rY_slider.set_limits(-180, 180)
rY_slider.int_value = 0
panel.add_child(rY_slider)
panel.add_child(gui.Label(""))
panel.add_child(gui.Label("Angle Z [°]"))
rZ_slider = gui.Slider(gui.Slider.INT)
rZ_slider.set_limits(-180, 180)
rZ_slider.int_value = 0
panel.add_child(rZ_slider)
panel.add_child(gui.Label(""))
reset_btn = gui.Button("Reset View")
panel.add_child(reset_btn)
panel.add_child(gui.Label(""))
save_btn = gui.Button("Save Mesh")
panel.add_child(save_btn)
# Add panel to window
window.add_child(panel)

# Regulate scene
setup_scene(scale)

# Associate slider to its relative callback
fe_slider.set_on_value_changed(on_fe_changed)
lb_slider.set_on_value_changed(on_lb_changed)
ar_slider.set_on_value_changed(on_ar_changed)
tX_slider.set_on_value_changed(on_X_changed)
tY_slider.set_on_value_changed(on_Y_changed)
tZ_slider.set_on_value_changed(on_Z_changed)
rX_slider.set_on_value_changed(on_alpha_changed)
rY_slider.set_on_value_changed(on_beta_changed)
rZ_slider.set_on_value_changed(on_gamma_changed)

# Associate button to its relative callback
reset_btn.set_on_clicked(on_reset_clicked)
save_btn.set_on_clicked(on_save)

# Set window layout
window.set_on_layout(on_layout)

# Set mouse events
scene.set_on_mouse(on_mouse_event)

# Run GUI
app.run()