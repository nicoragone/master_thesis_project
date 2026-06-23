import numpy as np
import copy
import os
import matplotlib.pyplot as plt
import pandas as pd

labels = ['Low', 'Medium', 'High']
num_configs = 30

BASE_DIR = r"C:\Users\olocc\OneDrive\Desktop\Thesis_new\TC_model_registration\Results"
file_paths = {
    "Pre": os.path.join(BASE_DIR, "Pre.txt"),
    "FPFH+RANSAC+ICP": os.path.join(BASE_DIR, "RANSAC+ICP.txt"),
    "CPD": os.path.join(BASE_DIR, "CPD.txt"),
    "GMMReg": os.path.join(BASE_DIR, "GMMReg.txt"),
    "SVR": os.path.join(BASE_DIR, "SVR.txt"),
    "Lepard + N-ICP": os.path.join(BASE_DIR, "Lepard.txt")
}

data = {}
method_metrics = {}
for name, filename in file_paths.items():
    data_table = pd.read_csv(filename, sep = "\t")
    method_metrics[name] = list(data_table.columns)
    data[name] = np.asarray(data_table)

plot_labels = {}
for metric_name in method_metrics["FPFH+RANSAC+ICP"]:
    if metric_name == "Deformation index":
        plot_labels[metric_name] = "Value"
    elif metric_name == "Time":
        plot_labels[metric_name] = "Value [s]"
    else:
        plot_labels[metric_name] = "Value [mm]"

plot_scales = {}
for metric_name in method_metrics["FPFH+RANSAC+ICP"]:
    if metric_name not in ["Deformation index", "Dist min", "Time"]:
        plot_scales[metric_name] = 'symlog'
    else:
        plot_scales[metric_name] = 'linear'

metric_labels = {}
for metric_name in method_metrics["FPFH+RANSAC+ICP"]:
    if metric_name in ["RMSE", "Range", "std"]:
        metric_labels[metric_name] = metric_name + " Point to Point"
    else:
        metric_labels[metric_name] = metric_name

metrics = method_metrics["Pre"]
methods = data.keys()
colors = ['lightgray', 'plum', 'skyblue', 'lightgreen', 'sandybrown', 'plum']
for metric in metrics:
    plt.figure(figsize = (16,8))
    plt_label = plot_labels[metric]
    positions = []
    data_to_plot = []
    plt_labels = []
    box_colors = []
    for i, label in enumerate(labels):
        for method in methods:
            start = num_configs * i
            end = num_configs * (i + 1)
            idx = method_metrics[method].index(metric)
            data_to_plot.append(data[method][start : end, idx])
        if i > 0:
            sep_position = i * len(methods) + 0.5
            plt.axvline(sep_position, linewidth = 2.0)
        plt_labels.extend(methods)
        box_colors.extend(colors)
    ax = plt.gca()
    bp = plt.boxplot(data_to_plot, patch_artist=True, tick_labels=plt_labels)
    ax.tick_params(axis='y', labelsize=12)
    ymax = ax.get_ylim()[1]
    if plot_scales[metric] == 'linear':
        ytop = ymax * 1.005
    else:
        ytop = ymax * 1.02
        plt.ylim(bottom=- 0.1)
    for i, label in enumerate(labels):
        group_center = i * len(methods) + len(methods) / 2 + 0.5
        plt.text(group_center, ytop, label + " Deformations",
                 ha='center', va='bottom',
                 fontsize=11, fontweight='bold')
    for patch, median, color in zip(bp['boxes'], bp['medians'], box_colors):
        patch.set_facecolor(color)
        median.set_color('firebrick')
    plt.yscale(plot_scales[metric])
    plt.xticks(rotation=25, fontsize=8.5)
    plt.ylabel(plt_label, labelpad=12, fontsize=12)
    plt.suptitle(metric_labels[metric].upper() + " comparison: Pre vs FPFH + RANSAC + ICP vs CPD vs GMMReg vs SVR vs Lepard + N-ICP",
                 fontweight='bold', fontsize = 14)
    plt.subplots_adjust(wspace=0.4)
    plt.subplots_adjust(left=0.08)
    plt.subplots_adjust(right=0.95)
    plt.subplots_adjust(top=0.92)
    plt.subplots_adjust(bottom=0.11)
    out_filename = os.path.join(BASE_DIR, "BP_" + metric + ".jpg")
    plt.savefig(out_filename)
    plt.show()
    plt.close()


metric = "Time"
# Add Lepard+NICP times
filename_lep = r"C:\Users\olocc\OneDrive\Desktop\Thesis_new\LepardTests\post_nicp_times.npz"
times = np.load(filename_lep)
lepard_times = times['times'].squeeze().reshape(-1, 1)
data["Lepard + N-ICP"] = lepard_times
method_metrics["Lepard + N-ICP"] = ["Time"]
data.pop("Pre", None)
method_metrics.pop("Pre", None)
colors = ['plum', 'skyblue', 'lightgreen', 'sandybrown', 'plum']
plt.figure(figsize = (16,8))
plt_label = plot_labels[metric]
data_to_plot = []
plt_labels = []
box_colors = []
for i, label in enumerate(labels):
    start = num_configs * i
    end = num_configs * (i + 1)
    for method in methods:
        idx = method_metrics[method].index(metric)
        data_to_plot.append(data[method][start : end, idx])
    if i > 0:
        sep_position = i * len(methods) + 0.5
        plt.axvline(sep_position, linewidth=2.0)
    plt_labels.extend(methods)
    box_colors.extend(colors)
ax = plt.gca()
bp = plt.boxplot(data_to_plot, patch_artist=True , tick_labels = plt_labels)
ax.tick_params(axis='y', labelsize=12)
ymax = ax.get_ylim()[1]
if plot_scales[metric] == 'linear':
    ytop = ymax * 1.005
else:
    ytop = ymax * 1.02
    plt.ylim(bottom=- 0.1)
for i, label in enumerate(labels):
    group_center = i * len(methods) + len(methods) / 2 + 0.5
    plt.text(group_center, ytop, label + " Deformations",
             ha='center', va='bottom',
             fontsize=11, fontweight='bold')
for patch, median, color in zip(bp['boxes'], bp['medians'], box_colors):
    patch.set_facecolor(color)
    median.set_color('firebrick')
plt.xticks(rotation=25, fontsize=8.5)
plt.ylabel(plt_label, labelpad=12, fontsize=12)
plt.suptitle(metric.upper() + " comparison: FPFH + RANSAC + ICP vs CPD vs GMMReg vs SVR vs Lepard + N-ICP",
                 fontweight='bold', fontsize = 14)
plt.subplots_adjust(wspace=0.4)
plt.subplots_adjust(left=0.08)
plt.subplots_adjust(right=0.95)
plt.subplots_adjust(top=0.92)
plt.subplots_adjust(bottom=0.11)
out_filename = os.path.join(BASE_DIR, "BP_" + metric + ".jpg")
plt.savefig(out_filename)
plt.show()
plt.close()