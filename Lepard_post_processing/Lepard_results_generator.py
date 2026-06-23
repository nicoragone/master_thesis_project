import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import copy
from scipy.spatial.distance import pdist

labels = ['low', 'medium', 'high']
num_configs = 30

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

# Plot point clouds
def show_point_clouds(src, tgt, title):
    vis = o3d.visualization.Visualizer()
    vis.create_window(title, 1400, 1050)
    src.paint_uniform_color([0.0, 0.07, 0.604])
    vis.add_geometry(src)
    tgt.paint_uniform_color([0.941, 0.525, 0.314])
    vis.add_geometry(tgt)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=75.0, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.25, 0.25, 0.25])
    vis.run()
    vis.destroy_window()

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

def compute_couple_metrics(src, tgt):
    src_points = np.asarray(src.points)
    tgt_points = np.asarray(tgt.points)
    # Find distances between the source point and the related target point
    dists = np.linalg.norm(src_points - tgt_points, axis=1)
    index = np.mean(dists)
    return index


folder_src = "C:\\Users\\olocc\\OneDrive\\Desktop\\Thesis_new\\Dataset\\For_tests\\Lepard\\"
folder_tgt = "C:\\Users\\olocc\\OneDrive\\Desktop\\Thesis_new\\Dataset\\Clouds_for_lepard_high_res\\Model27\\"

# Load original source point cloud
filename_SRC = r"C:\Users\olocc\OneDrive\Desktop\Thesis_new\Dataset\For_tests\Model27\base.ply"
SRC = o3d.io.read_point_cloud(filename_SRC)
SRC.estimate_normals()

# Compute single vertebra separators
sep_filename = r"C:\Users\olocc\OneDrive\Desktop\Thesis_new\Dataset\Base_meshes\Model27\Infos.txt"
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

# Write header into the output filename
filename_out = r"C:\Users\olocc\OneDrive\Desktop\Thesis_new\TC_model_registration\Results\Lepard.txt"
with open(filename_out, "w") as f:
    f.write("Error out" + "\t" + "Deformation" + "\t" + "Dist min" + "\t" + "Dist max" +
            "\t" + "Dist mean" + "\t" + "std" + "\t" + "Range" + "\t" + "RMSE")

for label in labels:
    for i in range(num_configs):
        # Read deformed source point clouds
        filename_src = folder_src + label + str(i) + ".ply"
        src = o3d.io.read_point_cloud(filename_src)
        src.estimate_normals()

        # Read target point clouds
        filename_tgt = folder_tgt + label + str(i) + ".ply"
        tgt = o3d.io.read_point_cloud(filename_tgt)
        tgt.estimate_normals()


        # Computing and saving registration results
        dist_min, dist_max, dist_mean, std_p2p, range_p2p, rmse_p2p = compute_distance_metrics(src, tgt)
        error_def = distance_preservation_error(separators, SRC, src)
        err_couple_out = compute_couple_metrics(src, tgt)

        # Save results into a text file
        with open(filename_out, "a") as f:
            f.write("\n" + str(err_couple_out) + "\t")
            f.write(str(error_def) + "\t")
            f.write(str(dist_min) + "\t")
            f.write(str(dist_max) + "\t")
            f.write(str(dist_mean) + "\t")
            f.write(str(std_p2p) + "\t")
            f.write(str(range_p2p) + "\t")
            f.write(str(rmse_p2p))