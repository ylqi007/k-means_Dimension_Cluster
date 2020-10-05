# k-means_Dimension_Cluster

## Dimension Clusters
> [YOLO9000:Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)         
> **Dimension Clusters.** We encounter two issues with anchor boxes where using them with YOLO. 
> The first is that the box dimensions are hand picked. The network can learn to adjust the boxes
> appropriately but if we pick better priors for the network to start with we can make it easier
> for network to learn to predict good detections.            
> Instead of choosing priors by hand, we run k-means clustering on the training set bounding
> boxes to automatically find good priors. If we use standard k-means with Euclidean distance
> larger boxes generate more error than smaller boxes. However, what we really want are priors
> that lead to good IOU scores, which is independent of the size of the box. Thus for our 
> distance metric we use:           
> d(box, centroid) = 1 - IOU(box, centroid)
> 
> k-means algorithm 的输入数据是 ground truth bounding box 的尺寸(i.e. width and height)。
> 因为每张 image 的 size 是不同的，图片中每个 object 的 size 也是不同的，因此也不具有可比性，因此最好不要用 object 原始的 bounding
> box 的尺寸进行聚类。因此，有必要标准化 bounding box 的宽度和高度与 image 的宽度和高度。


## Intersection over Union(IOU)
我们会用它来衡量两个 boundingbox 之间的距离。           
我们计算两个bounding box的iou时，只需要使用它们的4个位置参数(xmin,ymin, width, height)，这里引用别人一张图:         
![]()

<img src="https://farm8.staticflickr.com/7813/46412972842_6d2af063e9_h.jpg" width="300" height="400" alt="bbx">

iou的计算公式为:      
<img src="https://latex.codecogs.com/gif.latex?\begin{array}{rcl}&space;IoU&space;&=&&space;\frac{intersection}{union-intersection}\\&space;intersection&space;&=&&space;Min(w_{1},w_{2})\cdot&space;Min(h_{1},h_{2})\\&space;union&space;&=&&space;w_{1}h_{1}&plus;w_{2}h_{2}&space;\end{array}" />
  
<img src="https://latex.codecogs.com/gif.latex?\begin{array}{rcl}&space;IoU&space;&=&&space;\frac{intersection}{union-intersection}\\&space;intersection&space;&=&&space;Min(w_{1},w_{2})\cdot&space;Min(h_{1},h_{2})\\&space;union&space;&=&&space;w_{1}h_{1}&plus;w_{2}h_{2}&space;\end{array}" title="\begin{array}{rcl} IoU &=& \frac{intersection}{union-intersection}\\ intersection &=& Min(w_{1},w_{2})\cdot Min(h_{1},h_{2})\\ union &=& w_{1}h_{1}+w_{2}h_{2} \end{array}" />

h<sub>&theta;</sub>(x) = &theta;<sub>o</sub> x + &theta;<sub>1</sub>x


## [box_clustering.py](box_clustering.py)
* `def parse_annotation(ann_dir, img_dir, labels=[])`
* `def visualize_lables(seen_labels, train_imgs)`



## Reference:
[tensorflow-yolov3/docs/Box-Clustering.ipynb ](https://github.com/YunYang1994/tensorflow-yolov3/blob/master/docs/Box-Clustering.ipynb)
[codecogs](https://www.codecogs.com/latex/eqneditor.php)

