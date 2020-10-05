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
    fig.savefig('statistic_of_labels.pdf')
    plt.show()
    fig.savefig('statistic_of_labels.png')


def normalize_bounding_box(train_images):
    wh = []     # a list
    for anno in train_images:
        aw = float(anno['width'])   # width of an image
        ah = float(anno['height'])  # height of an image
        for obj in anno['object']:
            w = (obj['xmax'] - obj['xmin']) / aw    # make the width range between [0, GRID_W]
            h = (obj['ymax'] - obj['ymin']) / ah    # make the height range between [0, GRID_H]
            tmp = [w, h]
            wh.append(tmp)
    wh = np.array(wh)
    print("Clustering feature data is ready. Shape = (N object, width and height) = {}".format(wh.shape))
    return wh


def visulize_clustering_data(wh):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(wh[:, 0], wh[:, 1], alpha=0.3)
    ax.set_title("Clusters", fontsize=20)
    ax.set_xlabel("Normalized width", fontsize=20)
    ax.set_ylabel("Normalized height", fontsize=20)
    fig.savefig('clustering_data.pdf')
    plt.show()
    fig.savefig('clustering_data.png')


def iou(box, clusters):
    """
    :param box: np.array of shape (2,) containing w and h
    :param clusters: np.array of shape (N cluster, 2)
    :return:
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    intersection = x * y
    box_area = x * y
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)
    return iou_


def kmeans(boxes, k, dist=np.median, seed=1):
    """
    Calculate k-means clustering with the IoU metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters.
    :param dist: distance function
    :param seed:
    :return: numpy array of shape (k,2)
    """
    rows = boxes.shape[0]
    distances = np.empty((rows, k))  # N rows x N cluster
    last_cluster = np.zeros((rows,))

    np.random.seed(seed)

    # Initialize the cluster centers to be k items
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        # Step 1: allocate each item to the closest cluster centers
        for icluster in range(k):
            distances[:, icluster] = 1 - iou(clusters[icluster], boxes)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_cluster == nearest_clusters).all():
            break

        # Step 2: Calculate the cluster centers as mean (or median) of all the cases in the clusters
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_cluster = nearest_clusters

    return clusters, nearest_clusters, distances


train_images, seen_train_labels = parse_annotation(VOC2007_TRAIN_Annot, VOC2007_TRAIN_Images, labels=LABELS)
visualize_lables(seen_train_labels, train_images)
wh = normalize_bounding_box(train_images)
visulize_clustering_data(wh=wh)

kmax = 10
dist = np.mean
results = {}

for k in range(2, 3):
    clusters, nearest_clusters, distances = kmeans(wh, k, seed=2, dist=dist)
    WithinClusterMeanDist = np.mean(distances[np.arange(distances.shape[0]), nearest_clusters])
    result = {"clusters": clusters,
              "nearest_clusters": nearest_clusters,
              "distances": distances,
              "WithinClusterMeanDist": WithinClusterMeanDist}
    print(":2.0f clusters: mean IoU = {:5.4f}".format(k, 1-result["WithinClusterMeanDist"]))
    results[k] = result


