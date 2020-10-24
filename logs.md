[TOC]

# 20201023
- [x] 1. COCO, without class names, width, height
- [x] 2. uitls.fun to get anmes
- [x] 3. how to add width and height


```
Step 1. parse annotations and normalize object sizes
<class 'list'> 5011
<class 'dict'> 20
<class 'dict'> 4
{'object': [{'name': 'chair', 'xmin': 263, 'ymin': 211, 'xmax': 324, 'ymax': 339}, {'name': 'chair', 'xmin': 165, 'ymin': 264, 'xmax': 253, 'ymax': 372}, {'name': 'chair', 'xmin': 5, 'ymin': 244, 'xmax': 67, 'ymax': 374}, {'name': 'chair', 'xmin': 241, 'ymin': 194, 'xmax': 295, 'ymax': 299}, {'name': 'chair', 'xmin': 277, 'ymin': 186, 'xmax': 312, 'ymax': 220}], 'filename': './DATA/VOC2007/train/JPEGImages/000005.jpg', 'width': 500, 'height': 375}
```

In `box_clustering.py/kmeans()`:
```python
nearest_clusters = np.argmin(distances, axis=1)     # (15662,)
```
* `distances` is 2D array with `shape = (N, k)`, i.e each row has k elements representing `boxes[i]` to `clusters[j]` correspondingly.
```python
a = np.arange(6).reshape(2,3) + 10

array([[10, 11, 12],
       [13, 14, 15]])

# without axis, i.e. the index of min element of all elements
np.argmin(a)
0

# a.shape = (2, 3), axis = 0, i.e. remove 2, the shape of a will be (3,)
np.argmin(a, axis=0)
array([0, 0, 0])

# a.shape = (2m 3), remove axis=1, i.e. the shape will be (2,)
np.argmin(a, axis=1)
array([0, 0])
```

* Update the newest clusters
```python
clusters[cluster_id] = dist(boxes[nearest_clusters == cluster_id], axis=0)  # dist = np.mean
```
    * (N, 2) -> (2,), i.e. (np.mean(w), np.mean(h))