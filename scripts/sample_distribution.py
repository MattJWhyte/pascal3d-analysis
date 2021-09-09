import statistics

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
from extraction import *
from summary_bias import *


def color_map_color(value, cmap_name='RdGy', vmin=0, vmax=1):
    # norm = plt.Normalize(vmin, vmax)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)  # PiYG
    rgb = cmap(norm(abs(value)))[:3]  # will return rgba, we take only first 3 so we get rgb
    color = matplotlib.colors.rgb2hex(rgb)
    return color


def quadrant_symmetry_measure(azimuth_list, num_bins):
    degrees = np.random.randint(0, 360, size=200)
    bin_size = 360.0/num_bins
    bin_freq, b = np.histogram(azimuth_list, bins=np.arange(0, 360 + bin_size, bin_size))

    mu = sum(bin_freq)/float(num_bins)
    std = statistics.stdev(bin_freq, mu)
    return mu,std


def plot_azimuth_distribution(ax, azimuth_list):
    bin_size = 30
    a, b = np.histogram(azimuth_list, bins=np.arange(0, 360 + bin_size, bin_size))

    centers = np.deg2rad(np.ediff1d(b) // 2 + b[:-1])
    ax.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0, color="0.8", edgecolor='k')
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)


def plot_azimuth_info(azimuth_list, ax1, ax2):
    plot_azimuth_distribution(ax1, azimuth_list)
    ax2.set_ylim([0, 1])
    x, y = get_degree_bias(azimuth_list)
    theta = np.arctan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2)
    ax2.plot([0, theta], [0, r], "r")
    ax2.set_title("Azimuth bias (avg. angle, magnitude)")


def get_azimuth_analysis(category, path):
    # Normal azimuth information
    azimuth_list, _ = extract_annotations_by_condition(get_imageset("imagenet", category, "val"), category, "azimuth",
                                                       lambda x: True)
    fig = plt.figure(figsize=(14,14))
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    ax1.set_title("Azimuth distribution")
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    ax2.set_title("Azimuth bias (avg. angle, magnitude)")
    plot_azimuth_info(azimuth_list, ax1, ax2)
    plt.savefig(path+"azimuth.png")

    plt.clf()

    # Separated by occluded field
    occ_list, not_occ_list = extract_annotations_by_condition(get_imageset("imagenet", category, "val"), category, "azimuth",
                                                       lambda x: x["occluded"] == 1)
    #fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(2, 2, 1, projection="polar")
    ax1.set_title("Azimuth distribution, occluded=1")
    ax2 = fig.add_subplot(2, 2, 2, projection="polar")
    ax2.set_title("Azimuth bias (avg. angle, magnitude)")
    plot_azimuth_info(occ_list, ax1, ax2)

    ax1 = fig.add_subplot(2, 2, 3, projection="polar")
    ax1.set_title("Azimuth distribution, occluded=0")
    ax2 = fig.add_subplot(2, 2, 4, projection="polar")
    ax2.set_title("Azimuth bias (avg. angle, magnitude)")
    plot_azimuth_info(not_occ_list, ax1, ax2)

    plt.savefig(path + "azimuth-by-occluded.png")

    plt.clf()

    # Separated by occluded field
    trunc_list, not_trunc_list = extract_annotations_by_condition(get_imageset("imagenet", category, "val"), category,
                                                              "azimuth",
                                                              lambda x: x["truncated"] == 1)
    #fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(2, 2, 1, projection="polar")
    ax1.set_title("Azimuth distribution, truncated=1")
    ax2 = fig.add_subplot(2, 2, 2, projection="polar")
    ax2.set_title("Azimuth bias (avg. angle, magnitude)")
    plot_azimuth_info(trunc_list, ax1, ax2)

    ax1 = fig.add_subplot(2, 2, 3, projection="polar")
    ax1.set_title("Azimuth distribution, truncated=0")
    ax2 = fig.add_subplot(2, 2, 4, projection="polar")
    ax2.set_title("Azimuth bias (avg. angle, magnitude)")
    plot_azimuth_info(not_trunc_list, ax1, ax2)

    plt.savefig(path + "azimuth-by-truncated.png")


def get_total_azimuth_distribution(path):
    azimuth_list = []
    for cat in CATEGORIES:
        cat_azimuth_list, _ = extract_annotations_by_condition(get_imageset("imagenet", cat, "train"), cat,
                                                           "azimuth",
                                                           lambda x: True)
        azimuth_list += cat_azimuth_list

    fig = plt.figure(figsize=(14, 14))
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    ax1.set_title("Azimuth distribution")
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    ax2.set_title("Azimuth bias (avg. angle, magnitude)")
    plot_azimuth_info(azimuth_list, ax1, ax2)
    plt.savefig(path + "total-azimuth.png")


def get_azimuth_distributions(dataset, path):
    for cat in CATEGORIES:
        get_azimuth_analysis(cat, path+cat+"/")
