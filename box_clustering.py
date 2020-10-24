"""
https://github.com/YunYang1994/tensorflow-yolov3/blob/master/docs/Box-Clustering.ipynb
"""
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
current_palette = list(sns.xkcd_rgb.values())


from UNT import UNT
from VOC import VOC
from COCO import COCO

OUTPUT_DIR = "./data/"
FIG_SIZE = (10, 10)


def parse_annotation(ann_dir, img_dir, labels, dataset_name):
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
    for ann in sorted(os.listdir(ann_dir)):     # Iterate over all annotations
        # img["filename"], i.e path to image
        # img["width"], img["height"], i.e. the size of image
        # img["object"] is a list of objs,
        #   obj["name"], the name of this object
        #   obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]
        if "xml" not in ann:
            continue
        img = {'object': []}    # info of an image

        tree = ET.parse(ann_dir + ann)
        folder = ""
        for elem in tree.iter():
            # Step 1. Get image path.
            if 'folder' in elem.tag:    # folder is only needed in UNT Aerial Dataset.
                folder = elem.text
            if 'filename' in elem.tag:
                if dataset_name == "UNT_Aerial_Dataset":
                    path_to_image = img_dir + folder + '_' + elem.text
                elif dataset_name == "VOC2007":
                    path_to_image = img_dir + elem.text
                else:
                    raise ValueError("Not acceptable dataset!")
                img['filename'] = path_to_image
                if not os.path.exists(path_to_image):
                    assert False, "File does not exist!\n{}".format(path_to_image)

            # Step 2. Get image size.
            if 'width' in elem.tag:     # image width
                img['width'] = int(elem.text)
            if 'height' in elem.tag:    # image height
                img['height'] = int(elem.text)

            # Step 3. Object class and its bounding box
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
            all_imgs += [img]       # Only consider images with at least one object.

    return all_imgs, seen_labels


def parse_annotation_coco(anno_dir, labels, dataset_name):
    """

    :param anno_dir:
    :param labels:
    :param dataset_name:
    :return:
    """
    # TODO: name
    # TODO: width + height
    if not os.path.exists(anno_dir):
        raise ValueError("Annotation file doest not exits.")
    all_imgs = []
    seen_labels = {}
    with open(anno_dir, 'r') as f:
        data = f.readlines()
        img = {"object": []}
        for anno in data:
            line = anno.split()
            img['filename'] = line[0]
            size = line[1].split(",")
            img["width"] = int(size[0])
            img["height"] = int(size[1])

            bboxes = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in line[2:]])
            for bbox in bboxes:
                obj = {}
                obj["name"] = COCO.LABELS[int(bbox[4])]
                obj["xmin"] = bbox[0]
                obj["ymin"] = bbox[1]
                obj["xmax"] = bbox[2]
                obj["ymax"] = bbox[3]
                img["object"] += [obj]

                if obj['name'] in seen_labels:
                    seen_labels[obj['name']] += 1
                else:
                    seen_labels[obj['name']] = 1
        if len(img["object"]) > 0:
            all_imgs += [img]
    return all_imgs, seen_labels


def visualize_lables(seen_labels, train_imgs, output_dir):
    y_pos = np.arange(len(seen_labels))
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.barh(y_pos, list(seen_labels.values()))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(seen_labels.keys()))
    ax.set_title("The total number of object = {} in {} images".format(
        np.sum(list(seen_labels.values())), len(train_imgs)
    ))
    # fig.savefig(output_dir + 'statistic_of_labels.pdf')
    plt.show()
    fig.savefig(output_dir + 'statistic_of_labels.png')


def normalize_bounding_box(images):
    wh = []     # a list
    for img in images:
        aw = float(img['width'])   # width of an image
        ah = float(img['height'])  # height of an image
        for obj in img['object']:
            w = (obj['xmax'] - obj['xmin']) / aw    # make the width range between [0, GRID_W]
            h = (obj['ymax'] - obj['ymin']) / ah    # make the height range between [0, GRID_H]
            tmp = [w, h]
            wh.append(tmp)
    wh = np.array(wh)
    print("The bboxes are normalized. Shape = (N object, width and height) = {}".format(wh.shape))
    return wh


def visulize_clustering_data(wh, file_name):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(wh[:, 0], wh[:, 1], alpha=0.3)
    ax.set_title("Clusters", fontsize=20)
    ax.set_xlabel("Normalized width", fontsize=20)
    ax.set_ylabel("Normalized height", fontsize=20)
    # fig.savefig('UNT_clustering_data.pdf')
    plt.show()
    fig.savefig(file_name)


def iou(cluster, boxes):
    """
    Calculate distance between boxes with the specific cluster center.
    :param cluster: np.array of shape (2,) containing w and h, i.e. the centroid of a cluster
    :param boxes: np.array of shape (N boxes, 2)
    :return:
    """
    x = np.minimum(cluster[0], boxes[:, 0])
    y = np.minimum(cluster[1], boxes[:, 1])

    intersection = x * y
    cluster_area = cluster[0] * cluster[1]
    boxes_area = boxes[:, 0] * boxes[:, 1]

    iou_ = intersection / (cluster_area + boxes_area - intersection)
    return iou_


def kmeans(boxes, k, dist=np.median, seed=1):
    """
    Calculate k-means clustering with the IoU metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows, i.e. number of objects
    :param k: number of clusters.
    :param dist: distance function
    :param seed:
    :return: numpy array of shape (k, 2), i.e. k clusters, and each cluster is a (w,h) boxes
        - clusters: the centers of each cluster
        - nearest_clusters: the center index of each box
        - distances: the distance of each box to its cluster center
    """
    rows = boxes.shape[0]   # i.e. the amount of objects

    distances = np.zeros((rows, k))  # N rows x k cluster, distance between each point to its center
    last_cluster = np.zeros((rows,))

    np.random.seed(seed)

    # Initialize the cluster centers to be k items
    clusters = boxes[np.random.choice(rows, k, replace=False)]  # Randomly choose k boxes as initial cluster center
    while True:
        # Step 1: allocate each item to the closest cluster centers
        for cluster_id in range(k):
            distances[:, cluster_id] = 1 - iou(clusters[cluster_id], boxes) # distance to i-th cluster center

        nearest_clusters = np.argmin(distances, axis=1)     # (15662,)
        if (last_cluster == nearest_clusters).all():
            break

        # Step 2: Calculate the cluster centers as mean (or median) of all the cases in the clusters
        for cluster_id in range(k):    # dist = np.median
            clusters[cluster_id] = dist(boxes[nearest_clusters == cluster_id], axis=0)  # (N, 2) --> (2),

        last_cluster = nearest_clusters
    return clusters, nearest_clusters, distances


def plot_cluster_result(plt, clusters, nearest_clusters, withinClusterSumDist, wh, k, output_dir):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for icluster in np.unique(nearest_clusters):
        pick = (nearest_clusters == icluster)
        c = current_palette[icluster]
        ax.plot(wh[pick, 0], wh[pick, 1],
                "p", color=c, alpha=0.5,
                label="cluster = {}, N = {:6.0f}".format(icluster, np.sum(pick)))
        ax.text(clusters[icluster, 0],
                clusters[icluster, 1],
                "c{}\n({:3.2f}, {:3.2f})".format(icluster, clusters[icluster, 0], clusters[icluster, 1]),
                fontsize=20,
                color="red")
    ax.set_title("Clusters = %d" % k)
    ax.set_xlabel("Normalized width")
    ax.set_ylabel("Normalized height")
    ax.legend(title="Mean IoU = {:5.4f}".format(withinClusterSumDist))
    fig.savefig(output_dir + "k-means_with_k={}.png".format(k))


def resToString(res):
    res = ""
    for k in res:
        res += (k + ":" + res[k]) + "\n";


def write_to_file(results, scales, file_name, output_dir):
    file_name = output_dir + file_name
    with open(file_name, 'w') as f: # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        for k in results:
            f.write("k = {}\n".format(k))
            for anchor in results[k]["clusters"]:
                f.write("{:4f},{:4f},".format(anchor[0], anchor[1]))
            if k == 3:
                f.write("\n")
                for anchor in results[k]["clusters"]:
                    for i in range(3):
                        f.write("{:4f},{:4f},".format(anchor[0] * scales[i], anchor[1] * scales[i]))
            f.write("\n")
            # f.write("clusters:\n" + results[k]["clusters"])
            # f.write("nearest_clusters:\n" + results[k]["nearest_clusters"])
            # f.write("distance:\n" + results[k]["distance"])
            # f.write("WithinClusterMeanDist:\n" + results[k]["WithinClusterMeanDist"])
            # f.write("\n")


def k_means_on_Dataset(dataset, output_dir):
    """
    :param anno_dir: annotation folder dir
    :param img_dir: image folder dir
    :param dataset: dataset type, VOC or UNT
    :param output_dir:
    :return:
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Step 1. parse annotations and normalize object sizes
    print("Step 1. parse annotations and normalize object sizes")
    if dataset.NAME == "COCO2017":
        train_images, seen_train_labels = parse_annotation_coco(dataset.Annos_dir, dataset.LABELS, dataset.NAME)
    else:
        train_images, seen_train_labels = parse_annotation(dataset.Annos_dir, dataset.Images_dir, dataset.LABELS, dataset.NAME)

    # Step 2. Visualize labels
    print("Step 2. statistic and visualize labels")
    visualize_lables(seen_train_labels, train_images, output_dir)

    # Step 3. Normalize bboxes and visualize
    print("Step 3. normalize bounding boxes of each object and visualize")
    file_name = output_dir + "{}_Dataset_Visualization.png".format(dataset.NAME)
    wh = normalize_bounding_box(train_images)
    statistic_2D_to_1D(wh)

    # pass
    visulize_clustering_data(wh=wh, file_name=file_name)

    # Step 4. Do k-means for different k value, i.e. different clusters
    print("Step 4. Do k-means for different k value, i.e. different clusters")
    kmax = 10
    dist = np.mean
    results = {}
    for k in range(2, kmax):
        clusters, nearest_clusters, distances = kmeans(wh, k, dist=dist, seed=2)
        WithinClusterMeanDist = np.mean(distances[np.arange(distances.shape[0]), nearest_clusters])
        result = {"clusters": clusters,
                  "nearest_clusters": nearest_clusters,
                  "distances": distances,
                  "WithinClusterMeanDist": WithinClusterMeanDist}
        results[k] = result

    # Step 5. Visualize k-means results of different k
    print("Step 5. Visualize k-means results of different k")
    for k in range(2, kmax):
        result = results[k]
        clusters = result["clusters"]
        nearest_clusters = result["nearest_clusters"]
        withinClusterMeanDist = result["WithinClusterMeanDist"]

        plt.rc('font', size=8)
        plot_cluster_result(plt, clusters, nearest_clusters, 1 - withinClusterMeanDist, wh, k, output_dir)
        plt.show()

    # Step 6. Write the whole results into a file
    scales = [8, 16, 32]
    file_name = "{}_Dataset_Clustering_Results.txt".format(dataset.NAME)
    write_to_file(results, scales, file_name, output_dir)


def read_classes(filename):
    if not os.path.exists(filename):
        raise ValueError("Annotation file does not exits.")
    classes = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            classes += [line.strip('\n')]
    return classes

# =========================================================================== #
# Statistic
# 2D to 1D: area, width, height
# =========================================================================== #
def statistic_2D_to_1D(wh):
    area = wh[:, 0] * wh[:, 1]
    print(area[:5])
    print(wh[:5])
    plt.hist(wh[:, 0], bins=np.arange(0, 1, 0.1))
    plt.title("histogram")
    plt.show()


# =========================================================================== #
if __name__ == "__main__":
    k_means_on_Dataset(COCO, OUTPUT_DIR + "COCO/")

    # TODO
    # imgs, seen_labels = parse_annotation_coco(COCO.Annos_dir, COCO.LABELS, COCO.NAME)
    # print(seen_labels)
    # Readl COCO classes
    # classes = read_classes("./DATA/coco.names")
    # print(classes)