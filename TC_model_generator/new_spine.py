import open3d as o3d
import glob
import numpy as np

# Compute mean points distances with respect to mesh centroid
def compute_centroid_distance(pc):
    C = pc.get_center()
    pts = np.asarray(pc.points)
    distances = []
    for i in range(pts.shape[0]):
        d = np.sqrt((pts[i, 0] - C[0]) ** 2 + (pts[i, 1] - C[1]) ** 2 + (pts[i, 2] - C[2]) ** 2)
        distances.append(d)
    distance = np.mean(distances)
    return distance

# Load OpenSim Model
mesh_files = glob.glob(r"C:\Users\olocc\OneDrive\Desktop\Thesis_new\OpenSim_base_pose\*.ply")
pcs = []
combined_pc = o3d.geometry.PointCloud()
for filename in mesh_files:
    if "lumbar" in filename or "torso" in filename:
        pc = o3d.io.read_point_cloud(filename)
        combined_pc += pc

# Load segmented model
PC_IMG = o3d.geometry.PointCloud()

img_files = glob.glob(r"C:\Users\olocc\OneDrive\Desktop\Thesis_new\Dataset\Base_meshes\Model27\*.ply")
for filename in img_files:
    if "Merged" not in filename:
        pc = o3d.io.read_point_cloud(filename)
        PC_IMG += pc #-90x #-90y
T = np.array([[0, 1, 0, 0],
              [0, 0, 1, 0],
              [1, 0, 0, 0],
              [0, 0, 0, 1]])
PC_IMG.transform(T)
PC_IMG.estimate_normals()
pcs.append(PC_IMG)

# Estimate scale conversion factor
d1 = compute_centroid_distance(combined_pc)
d2 = compute_centroid_distance(PC_IMG)
print("Scale conversion factor = " + str(d2/d1))
combined_pc.scale(d2/d1, combined_pc.get_center())
combined_pc.translate(PC_IMG.get_center() - combined_pc.get_center() + [0, 25 , 150])
pcs.append(combined_pc)
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=75.0, origin=[0,0,0])
pcs.append(coordinate_frame)

o3d.visualization.draw_geometries(pcs)

# Save scale factor into the infos file
filename_infos = r"C:\Users\olocc\OneDrive\Desktop\Thesis_new\Dataset\Meshes\BasePose27\Infos.txt"
with open(filename_infos, 'a') as f:
    f.write("\n" + "Scale factor = " + str(int(d2/d1)))