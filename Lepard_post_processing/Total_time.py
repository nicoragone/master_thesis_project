import numpy as np
from scipy.io import loadmat

# Create dictionary for N-ICP times
filename = r"C:\Users\olocc\OneDrive\Desktop\Thesis_new\Dataset\Lepard_nr_icp_results\times_nicp.mat"
data = loadmat(filename, squeeze_me=True)
times = data["times"]  # Nx2
n_icp_time = {}
for i, row in enumerate(times):
    # Create key for times
    key = "prediction_" + str(i)
    value = float(row[1])  # time value [s]
    n_icp_time[key] = value

total_time = {}
for i in range(90):
    key = "prediction_" + str(i)
    filename_lep = ("C:\\Users\\olocc\\OneDrive\\Desktop\\Thesis_new\\LepardTests\\Times\\test_results\\" +
                    key + "_time.npz")
    data = np.load(filename_lep)
    lep_time = data['registration_time']
    nicp_time = n_icp_time[key]
    time = lep_time + nicp_time
    total_time[key] = time

filename_corr = "C:\\Users\\olocc\\OneDrive\\Desktop\\Thesis_new\\LepardTests\\2711\\correspondences.npz"
data = np.load(filename_corr) # num_pred -> config
corr = dict(data['correspondences'])
config_times = {}
for i in range(90):
    key_time = "prediction_" + str(i)
    key_corr = str(i)
    key = corr[key_corr]
    config_times[key] = total_time[key_time]

labels = ['low', 'medium', 'high']
lep_n_icp_times = np.zeros(90)
for i, label in enumerate(labels):
    for j in range(30):
        config = label + str(j)
        lep_n_icp_times[30 * i + j] = config_times[config]

filename_out = r"C:\Users\olocc\OneDrive\Desktop\Thesis_new\LepardTests\post_nicp_times"
np.savez(filename_out,
         times = lep_n_icp_times)