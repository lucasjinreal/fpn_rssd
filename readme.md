# FPN RSSDs

this project aims to combine FPN and SSD for object detection. as all we know, SSD are not fully
using multi layer features to merge low resolution and high semantic feature togather to get a good
result on object detection. We applied FPN on SSD to make it more reasonable.

Beside, a MobileNetV2 backbone has been added to achieve a fast speed for SSD. **Also, we call it
RSSD because we try do RBox detection on image rather than regular box**.

*updates*:

2019.07.15: We complete the detection part combines FPN and SSD, check our codes on MANA platform: http://manaai.cn
2019.01.09: first init project, all things will on going still I successfully trained a RBox detection model. keep an eye on our progress.


## RSSD

So the RSSD is a Rotated SSD framework which is same as DRBox (there is a paper compose that). In the implementation will contains both normal detection framework and the RBox detection.
