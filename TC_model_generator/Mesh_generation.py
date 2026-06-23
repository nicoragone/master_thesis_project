import copy
import nibabel as nib
import numpy as np
from skimage import measure
import open3d as o3d
from scipy import ndimage

# Model bodies
lumbars = ["L5" , "L4", "L3", "L2", "L1"]
torso = ["T12", "T11", "T10", "T9", "T8", "T7", "T6", "T5", "T4", "T3", "T2", "T1"]

nifti_path = r"C:\Users\olocc\OneDrive\Desktop\Thesis_new\TC_images\case_0025.nii\case_0025.nii"
output_folder =  r"C:\Users\olocc\OneDrive\Desktop\Thesis_new\Dataset\Meshes\BasePose25"

# Read 3D nifti images
img = nib.load(nifti_path)
# pixel labels
data = img.get_fdata()
voxel_size = img.header.get_zooms()
mesh_visual = []
# Initial threshold
threshold = np.max(data)
first = True
total_mesh = o3d.geometry.TriangleMesh()
start = 0
idx_separators = np.zeros((len(lumbars) + len(torso), 2))
min_voxels = 250

for i, body in enumerate(lumbars):
    mask = data == threshold
    labeled_mask, num = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled_mask, range(num + 1))
    mask = np.isin(labeled_mask, np.where(sizes > min_voxels)[0])
    verts, faces, normals, _ = measure.marching_cubes(~mask, level=0.5)
    normals = normals.astype(np.float64)
    # Create an Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    mesh.triangle_normals = o3d.utility.Vector3dVector(normals)
    mesh = mesh.filter_smooth_simple(number_of_iterations=3)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    mesh.orient_triangles()
    mesh_visual.append(mesh)
    end = start + len(mesh.vertices) - 1
    idx_separators[i, 0] = start
    idx_separators[i, 1] = end
    total_mesh += mesh
    threshold -= 1
    start += len(mesh.vertices)
    first = False

torso_mesh = o3d.geometry.TriangleMesh()
for i, body in enumerate(torso):
    data = img.get_fdata()
    mask = data == threshold
    labeled_mask, num = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled_mask, range(num + 1))
    mask = np.isin(labeled_mask, np.where(sizes > min_voxels)[0])
    verts, faces, normals, _ = measure.marching_cubes(~mask, level=0.5)
    normals = normals.astype(np.float64)
    # Create an Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    mesh.triangle_normals = o3d.utility.Vector3dVector(normals)
    mesh = mesh.filter_smooth_simple(number_of_iterations=3)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    mesh.orient_triangles()
    end = start + len(mesh.vertices) - 1
    idx_separators[i + len(lumbars), 0] = start
    idx_separators[i + len(lumbars), 1] = end
    torso_mesh += mesh
    threshold -= 1
    start += len(mesh.vertices)

print(idx_separators)

total_mesh += torso_mesh
mesh_visual.append(torso_mesh)
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=75.0, origin=[0,0,0])

# View model mesh
mesh_visual.append(coordinate_frame)
o3d.visualization.draw_geometries(
    mesh_visual,
    window_name="3D Segmentation Model",
    mesh_show_back_face=True,
    width=900,
    height=700
)

# Save meshes into a PLY file
T_align = np.array([[0, 1, 0, 0],
              [0, 0, 1, 0],
              [1, 0, 0, 0],
              [0, 0, 0, 1]])
body_names = copy.deepcopy(lumbars)
body_names.append("Torso")
for body, mesh in zip(body_names, mesh_visual):
    filename_out = output_folder + "/" + body + ".ply"
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    mesh.orient_triangles()
    o3d.io.write_triangle_mesh(filename_out, mesh)
filename_out = output_folder + "/" + "Merged.ply"
total_mesh.transform(T_align)
total_mesh.compute_vertex_normals()
total_mesh.compute_triangle_normals()
total_mesh.orient_triangles()
o3d.io.write_triangle_mesh(filename_out, total_mesh)
# Save separators indexes for thoracic vertebra
filename_separators = output_folder + "/" + "Infos.txt"
with open(filename_separators, "a") as f:
    for i, body in enumerate(lumbars):
        f.write("\n" + str(body) + "\t" + str(int(idx_separators[i, 0])) + "\t" +
                str(int(idx_separators[i, 1])))
    for i, body in enumerate(torso):
        f.write("\n" + str(body) + "\t" + str(int(idx_separators[i + len(lumbars), 0])) + "\t" +
                str(int(idx_separators[i + len(lumbars), 1])))