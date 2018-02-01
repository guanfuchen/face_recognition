# LFW人脸数据集

是VGG Face Dataset，数据集由2,622个身份组成。
每个身份都有一个关联的文本文件，其中包含图像的URL和相应的人脸检测。

Each file in files folder provides images for that identity.
Each line in the file provides an image url, a bounding box and some other attributes.

Each line represents:

id uid url left top right bottom pose detection_score curation

id: Integer id for an image.

uid: Unique reference id for an image.

url: The weblink for the image.

[left top right bottom] the bounding box for an image.
pose: frontal/profile (pose>2 signifies a frontal face while
pose<=2 represents left and right profile detection).

detection score: Score of a DPM detector.

curation: Whether this image was a part of final curated dataset

文件夹中的每一个文件都是一个实体的身份，提供了图像url和bounding box
每一行提供一个图像的id，一个图像参考的唯一ID，图像weblink，[left top right bottom]的人脸bounding box

```bash
00000001 http://aplive.net/wp-content/uploads/2014/07/aamir_khan_86215.jpg 119.19 99.49 295.47 275.78 3.00 2.31 1
```

参考如下：
- [VGG Face Dataset](http://www.robots.ox.ac.uk/~vgg/data/vgg_face/)