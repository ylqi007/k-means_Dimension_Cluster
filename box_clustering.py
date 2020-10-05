import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Labels of VOC
LABELS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
          'bus', 'car', 'cat', 'chair', 'cow',
          'diningtable', 'dog', 'horse', 'motorbike', 'person',
          'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# VOC Dataset directory
VOC2007_TRAIN = "./DATA/VOC2007/train"
VOC2007_TRAIN_Images = "./DATA/VOC2007/train/JPEGImages/"  # Since this is a dir, ended with `/`
VOC2007_TRAIN_Annot = "./DATA/VOC2007/train/Annotations/"

print(os.listdir(VOC2007_TRAIN))


def parse_annotation(ann_dir, img_dir, labels=[]):
    """
    :param ann_dir:
    :param img_dir:
    :param labels:
    :return:
        - all_imgs: a list, information of images. Each element in `all_imgs` is a dict and has
            * filename: path_to_image
            * width: width of this image
            * height: height of this image
            * object:
                * name: name of this object,
                * xmin: xmin of this object,
                * ymin: ymin of this object,
                * xmax: xmax of this object,
                * ymax: ymax of this object,
        - seen_labels:
    """
    all_imgs = []
    seen_labels = {}
    for ann in sorted(os.listdir(ann_dir)):
        if "xml" not in ann:
            continue
        img = {'object': []}
        # print(ann_dir + ann)

        tree = ET.parse(ann_dir + ann)
        for elem in tree.iter():
            if 'filename' in elem.tag:
                path_to_image = img_dir + elem.text
                img['filename'] = path_to_image
                if not os.path.exists(path_to_image):
                    assert False, "File does not exist!\n{}".format(path_to_image)
            if 'width' in elem.tag:     # image width
                img['width'] = int(elem.text)
            if 'height' in elem.tag:    # image height
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}    # Info of one object
                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text
                        if len(labels) > 0 and obj['name'] not in labels:   # lables do not contains this obj
                            break
                        else:
                            img['object'] += [obj]

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:  # if there has at least one object in current annotation file
            all_imgs += [img]

    return all_imgs, seen_labels


def visualize_lables(seen_labels, train_imgs):
    y_pos = np.arange(len(seen_labels))
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.barh(y_pos, list(seen_labels.values()))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(seen_labels.keys()))
    ax.set_title("The total number of object = {} in {} images".format(
        np.sum(list(seen_labels.values())), len(train_imgs)
    ))
    fig.savefig('fig1.pdf')
    plt.show()
    fig.savefig('statistic_of_labels.png')


train_images, seen_train_labels = parse_annotation(VOC2007_TRAIN_Annot, VOC2007_TRAIN_Images, labels=LABELS)
visualize_lables(seen_train_labels, train_images)