
import cv2
import scipy.io as sio
import numpy as np


def get_bb(path):
    mat = sio.loadmat(path)
    objs = mat["record"][0, 0]["objects"]
    return objs[0]["bbox"][0][0]


def img_size(path):
    mat = sio.loadmat(path)
    dim = mat["record"][0, 0]["imgsize"][0]
    return dim[0],dim[1]


def img_aspect_ratio(path):
    sz = img_size(path)
    return sz[0] / sz[1]


def adjust_aspect_ratio(img_size, bb, asp_rat):

    width,height = img_size
    x_min = int(np.round(bb[0]))
    x_max = int(np.round(bb[2]))
    y_min = int(np.round(bb[1]))
    y_max = int(np.round(bb[3]))

    def get_ratio():
        return (x_max-x_min) / (y_max-y_min)

    alt = True
    break_constraints = False
    if get_ratio() > asp_rat: # Want to increase height
        while get_ratio() > asp_rat:
            can_go_low = y_min >= 1
            can_go_high = y_max <= height - 1
            if (can_go_high and can_go_low) or break_constraints:
                if alt:
                    y_min -= 1
                else:
                    y_max += 1
                alt = not alt
            elif can_go_high:
                y_max += 1
            elif can_go_low:
                y_min -= 1
            else:
                break_constraints = True
    else:
        while get_ratio() < asp_rat:
            can_go_low = x_min >= 1
            can_go_high = x_max <= width-1
            if (can_go_high and can_go_low) or break_constraints:
                if alt:
                    x_min -= 1
                else:
                    x_max += 1
                alt = not alt
            elif can_go_high:
                x_max += 1
            elif can_go_low:
                x_min -= 1
            else:
                break_constraints = True

    return [x_min, y_min, x_max, y_max]


def resize_img(img, img_size, roi):
    print("Img size")
    print(img_size)
    print(roi)
    width,height = img_size
    x_min = roi[0]
    x_max = roi[2]
    y_min = roi[1]
    y_max = roi[3]
    roi_width = x_max-x_min
    roi_height = y_max-y_min
    print("{} {}".format(roi_width, roi_height))
    if min(roi) >= 0 and x_max < width and y_max < height: # Life is easy, no padding necessary:
        return img[y_min:y_max+1,x_min:x_max+1]
    else: # We need to add additional padding
        blank_image = np.zeros((roi_height, roi_width, 3), np.uint8)
        ver_pad = int(max(roi_height-height, 0))//2
        hor_pad = int(max(roi_width-width, 0))//2
        print("PADDING {} {}".format(ver_pad,hor_pad))
        new_x_min = max(x_min,0)
        new_x_max = min(x_max,width)
        new_y_min = max(y_min,0)
        new_y_max = min(y_max,height)
        roi_img = blank_image[ver_pad:ver_pad+min(roi_height,height),hor_pad:hor_pad+min(roi_width,width)]
        print(roi_img.shape)
        ext_img = img[new_y_min:new_y_max,new_x_min:new_x_max]
        print(ext_img.shape)
        blank_image[ver_pad:ver_pad + min(ext_img.shape[0], height), hor_pad:hor_pad + min(ext_img.shape[1], width)] = ext_img
        return blank_image


# Width / Height
def bb_aspect_ratio(bb):
    return (bb[2]-bb[0]) / (bb[3]-bb[1])


def draw_bounding_boxes(img, bb, color_arr):
    x_min = int(np.round(bb[0]))
    x_max = int(np.round(bb[2]))
    y_min = int(np.round(bb[1]))
    y_max = int(np.round(bb[3]))
    for i in range(x_min, x_max):
        img[y_min, i] = color_arr
        img[y_max, i] = color_arr
    for i in range(y_min, y_max):
        img[i, x_min] = color_arr
        img[i, x_max] = color_arr
