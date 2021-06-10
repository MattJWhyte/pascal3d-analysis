import statistics
import sys

import scipy.io as sio
import os
import numpy as np
import matplotlib.pyplot as plt
from summary_bias import *
from sample_distribution import *

def get_az_by_diff(img_list, category, include_coarse=False):
    diff_list = []
    not_diff_list = []
    for img in img_list:
        annotation_list = get_image_annotations(img, category)
        for annotation_dict in annotation_list:
            if annotation_dict["viewpoint"]["num_anchor"] > 0:
                if annotation_dict["truncated"] == 1:
                    diff_list.append(annotation_dict["viewpoint"]["azimuth"])
                else:
                    not_diff_list.append(annotation_dict["viewpoint"]["azimuth"])
            elif include_coarse:
                if annotation_dict["truncated"] == 1:
                    diff_list.append(annotation_dict["viewpoint"]["azimuth_coarse"])
                else:
                    not_diff_list.append(annotation_dict["viewpoint"]["azimuth_coarse"])
    return diff_list, not_diff_list


def get_annotations(category, dataset, annotation):
    samples = [f.path for f in os.scandir(annotation_dir+category+"_"+dataset)]
    ct = 0
    vals = []
    for sample in samples:
        new_vals, d = get_annotation(sample, category, annotation)
        ct += d
        vals += new_vals
    print("0 for non-zero anchors {}".format(ct))
    return vals


#get_avg_azimuth("pascal", lambda x: True)

'''

fig = plt.figure(figsize=(16,8))
cat = ["boat", "bicycle", "boat", "bottle", "bus", "car", "chair", "diningtable", "motorbike", "sofa", "train", "tvmonitor"]

samples = [f.path for f in os.scandir(annotation_dir+"tvmonitor"+"_"+"pascal")]
diff, not_diff = get_az_by_diff(samples, "tvmonitor")

ls = [diff, not_diff]
nm = ["az-diff", "az-notdiff"]


for i in range(2):
    degrees = np.random.randint(0, 360, size=200)
    radians = np.deg2rad(degrees)

    bin_size = 15
    a, b = np.histogram(ls[i], bins=np.arange(0, 360 + bin_size, bin_size))
    centers = np.deg2rad(np.ediff1d(b) // 2 + b[:-1])

    ax = fig.add_subplot(int("12{}".format(i+1)), projection='polar')
    ax.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0, color='.8', edgecolor='k')
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_title("azimuth for difficult=1 (including coarse)")

plt.savefig("{}.png".format("az-by-diff"))
plt.clf()
'''

'''
for i in range(6):
    print(cat[i])
    vals = get_annotations(cat[6+i], "pascal", "azimuth")
    degrees = np.random.randint(0, 360, size=200)
    radians = np.deg2rad(degrees)

    bin_size = 15
    a, b = np.histogram(vals, bins=np.arange(0, 360 + bin_size, bin_size))
    centers = np.deg2rad(np.ediff1d(b) // 2 + b[:-1])

    ax = fig.add_subplot(int("26{}".format(i+1)), projection='polar')
    ax.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0, color='.8', edgecolor='k')
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
'''

#summarise_azimuth_bias("imagenet", "analysis/imagenet/")
get_azimuth_distributions("imagenet", "analysis/imagenet/")


'''
bicycle_img = get_imageset("imagenet", "car", "val")
az_list,_ = extract_annotations_by_condition(bicycle_img, "car", "azimuth", lambda x: True)

bin_size = 15
bin_freq, b = np.histogram(az_list, bins=np.arange(0, 360 + bin_size, bin_size))

initial_sum = sum(bin_freq)

while sum(bin_freq) > 0.75*initial_sum:
    min_std = statistics.stdev(bin_freq)
    min_idx = None
    for i in range(24):
        mu = sum(bin_freq) / 24.0
        if bin_freq[i] > mu:
            bin_freq[i] -= 1
            new_std = statistics.stdev(bin_freq)
            if new_std < min_std:
                min_std = new_std
                min_idx = i
            bin_freq[i] += 1
    if min_idx is None:
        break
    bin_freq[min_idx] -= 1


fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection='polar')

centers = np.deg2rad(np.ediff1d(b) // 2 + b[:-1])
ax.bar(centers, bin_freq, width=np.deg2rad(bin_size), bottom=0.0, color=[color_map_color(np.random.random()) for i in range(len(centers))], edgecolor='k')
ax.set_theta_zero_location("E")
ax.set_theta_direction(1)

plt.savefig("reduced-car-azimuth.png")

'''





'''
bias
    azimuth
    azimuth-by-difficult
    azimuth-by-occluded
    azimuth-by-truncated
elevation

'''
