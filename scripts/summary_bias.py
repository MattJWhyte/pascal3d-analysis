
from extraction import *
import numpy as np
import matplotlib.pyplot as plt


def unit_vector(theta):
    rad = np.radians(theta)
    return np.cos(rad), np.sin(rad)


def get_degree_bias(deg_list):
    avg_x = 0.0
    avg_y = 0.0
    ct = 0.0
    for deg in deg_list:
        x, y = unit_vector(deg)
        avg_x += x
        avg_y += y
        ct += 1
    avg_x = avg_x/ct
    avg_y = avg_y/ct
    return [avg_x, avg_y]


def get_avg_value(dataset, field, condition, include_coarse=False):
    true_avg_list = []
    false_avg_list = []
    for cat in CATEGORIES:
        img_list = get_imageset(dataset, cat, "val")
        true_cond_list, false_cond_list = extract_annotations_by_condition(img_list, cat, field, condition,
                                                    include_coarse=include_coarse)
        if len(true_cond_list) > 0:
            true_avg_list.append(sum(true_cond_list)/len(true_cond_list))
        else:
            true_avg_list.append(None)
        if len(false_cond_list) > 0:
            false_avg_list.append(sum(false_cond_list)/len(false_cond_list))
        else:
            false_avg_list.append(None)

    return true_avg_list, false_avg_list


def get_avg_azimuth(dataset, condition, include_coarse=False):
    true_bias_list = []
    false_bias_list = []
    for cat in CATEGORIES:
        img_list = get_imageset(dataset, cat, "val")
        true_cond_list, false_cond_list = extract_annotations_by_condition(img_list, cat, "azimuth", condition,
                                                    include_coarse=include_coarse)
        if len(true_cond_list) > 0:
            true_bias_list.append(get_degree_bias(true_cond_list))
        else:
            true_bias_list.append(None)
        if len(false_cond_list) > 0:
            false_bias_list.append(get_degree_bias(false_cond_list))
        else:
            false_bias_list.append(None)

    return true_bias_list, false_bias_list


def get_azimuth(dataset, subset):
    az_list = []
    for cat in CATEGORIES:
        img_list = get_imageset(dataset, cat, subset)
        true_cond_list, _ = extract_annotations_by_condition(img_list, cat, "azimuth", lambda x: True,
                                                    include_coarse=False, single=True)
        az_list += true_cond_list
    return az_list


def get_azimuth_by_cat(dataset, subset):
    az_list = {}
    for cat in CATEGORIES:
        img_list = get_imageset(dataset, cat, subset)
        true_cond_list, _ = extract_annotations_by_condition(img_list, cat, "azimuth", lambda x: True,
                                                    include_coarse=False, single=True)
        az_list[cat] = [float(a) for a in true_cond_list]
    return az_list


def plot_avg_azimuth(ax, bias_list):
    for i in range(len(CATEGORIES)):
        bias = bias_list[i]
        if bias is not None:
            ax.plot(np.linspace(0,bias[0],10), np.linspace(0,bias[1],10))
            theta = np.round(np.rad2deg(np.arctan2(bias[1], bias[0])),1)
            r = np.round(np.sqrt(bias[0] ** 2 + bias[1] ** 2),3)
            ax.text(bias[0], bias[1], "{} ({}, {})".format(CATEGORIES[i],theta,r))
        else:
            ax.plot([0,0], [0,0])


def summarise_value(dataset, value, path):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    avg_list, _ = get_avg_value(dataset, value, lambda x: True, include_coarse=False)
    ax.bar(CATEGORIES, avg_list)
    ax.set_title("Average {}".format(value))
    plt.savefig(path+"{}.png".format(value.lower()))

    fields = ["difficult", "occluded", "truncated"] if dataset == "pascal" else ["occluded", "truncated"]

    for f in fields:
        diff_list, not_diff_list = get_avg_value(dataset, value, lambda x: x[f] == 1, include_coarse=False)

        x = np.arange(len(CATEGORIES))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, diff_list, width, label='{f}=1')
        rects2 = ax.bar(x + width / 2, not_diff_list, width, label='{f}=0')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_title('Mean {}'.format(value))
        ax.set_xticks(x)
        ax.set_xticklabels(CATEGORIES)
        ax.legend()
        #ax.bar_label(rects1, padding=3)
        #ax.bar_label(rects2, padding=3)

        plt.savefig(path+"{}-by-{}.png".format(value,f))

def summarise_azimuth_bias(dataset, path):
    # Normal azimuth bias
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    bias_list, _ = get_avg_azimuth(dataset, lambda x: True, include_coarse=False)
    plot_avg_azimuth(ax, bias_list)
    ax.set_title("Azimuth Bias (avg. degree, asymmetry)")
    plt.savefig(path+"azimuth.png")

    fields = ["difficult", "occluded", "truncated"] if dataset == "pascal" else ["occluded", "truncated"]
    for f in fields:
        fig = plt.figure(figsize=(20, 10))
        diff_list, not_diff_list = get_avg_azimuth(dataset, lambda x: x[f] == 1, include_coarse=False)
        ax1 = fig.add_subplot(121)
        plot_avg_azimuth(ax1, diff_list)
        ax2 = fig.add_subplot(122)
        plot_avg_azimuth(ax2, not_diff_list)
        ax1.set_title(str("Azimuth Bias, " + f + "=1 (avg. degree, asymmetry)"))
        ax2.set_title(str("Azimuth Bias, " + f + "=0 (avg. degree, asymmetry)"))
        plt.savefig(path+"azimuth-by-{}.png".format(f))
