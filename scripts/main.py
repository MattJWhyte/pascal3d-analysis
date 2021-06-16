
import draw_bb
import cv2
import matplotlib.pyplot as plt
import numpy as np
import extraction
import os
import json


def asCartesian(rthetaphi):
    #takes list rthetaphi (single coord)
    r = rthetaphi[0]
    theta = rthetaphi[1]* np.pi/180 # to radian
    phi = rthetaphi[2]* np.pi/180
    x = r * np.sin( theta ) * np.cos( phi )
    y = r * np.sin( theta ) * np.sin( phi )
    z = r * np.cos( theta )
    return [x,y,z]


for cat in extraction.CATEGORIES:
    os.mkdir("../cnn_512_320_val/{}_imagenet/".format(cat))


out_dict = {}
for cat in extraction.CATEGORIES:
    out_dict[cat] = {}
    with open("../PASCAL3D+_release1.1/Image_sets/{}_imagenet_val.txt".format(cat), "r") as f:
        for img_name in f.readlines():
            img_name = img_name.replace("\n", "")
            ann = extraction.get_image_annotations("../PASCAL3D+_release1.1/Annotations/{}_imagenet/{}.mat".format(cat, img_name), cat)[0]
            az = ann["viewpoint"]["azimuth"]
            el = ann["viewpoint"]["elevation"]
            di = ann["viewpoint"]["distance"]
            coords = asCartesian([di,el,az])
            out_dict[cat][img_name] = coords

with open("../cnn_512_320_val/annotation.json", "w") as f:
    f.write(json.dumps(out_dict))


for cat in extraction.CATEGORIES:
    with open("../PASCAL3D+_release1.1/Image_sets/{}_imagenet_val.txt".format(cat), "r") as f:
        for img_name in f.readlines():
            img_name = img_name.replace("\n", "")
            img = cv2.imread("../PASCAL3D+_release1.1/Images/{}_imagenet/{}.JPEG".format(cat,img_name))
            sz = draw_bb.img_size("../PASCAL3D+_release1.1/Annotations/{}_imagenet/{}.mat".format(cat, img_name))
            bb = draw_bb.get_bb("../PASCAL3D+_release1.1/Annotations/{}_imagenet/{}.mat".format(cat, img_name))
            new_bb = draw_bb.adjust_aspect_ratio(sz,bb,1.6)
            adj_img = draw_bb.resize_img(img, sz, new_bb)
            adj_img = cv2.resize(adj_img, (512, 320))
            cv2.imwrite("../cnn_512_320_val/{}_imagenet/{}.png".format(cat,img_name), adj_img)



'''
count = np.zeros((21,))

asp = np.linspace(1,2,21)

avg_wid = 0.0
avg_ht = 0.0
ct = 0.0

one_25_count = 0.0

for cat in extraction.CATEGORIES:
    with open("../PASCAL3D+_release1.1/Image_sets/{}_imagenet_train.txt".format(cat), "r") as f:
        for img_name in f.readlines():
            img_name = img_name.replace("\n", "")
            sz = draw_bb.img_size("../PASCAL3D+_release1.1/Annotations/{}_imagenet/{}.mat".format(cat,img_name))
            avg_wid += sz[0]
            avg_ht += sz[1]
            ct += 1
            low = draw_bb.bb_aspect_ratio(draw_bb.get_bb("../PASCAL3D+_release1.1/Annotations/{}_imagenet/{}.mat".format(cat,img_name)))
            hi = draw_bb.img_aspect_ratio("../PASCAL3D+_release1.1/Annotations/{}_imagenet/{}.mat".format(cat,img_name))
            if low <= 1.6 <= hi or hi <= 1.6 <= low:
                one_25_count += 1
            for i in range(len(count)):
                if low <= asp[i] <= hi or hi <= asp[i] <= low:
                    count[i] += 1

plt.plot(asp, count)
plt.savefig("bb_asp.png")

print(np.max(count))
print(asp[np.argmax(count)])
print(asp[np.argmax(count)]/ct)
print(avg_wid/ct)
print(avg_ht/ct)

print("1.25 asp : {}%".format(one_25_count/ct*100.0))
'''