#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys

sys.path.append("..")


# In[ ]:


import cv2
import pandas as pd
from matplotlib import pyplot as plt
from ultralytics import YOLO

# In[ ]:


PATH = "../../storage/data/coco2017/val2017/000000289343.jpg"
RESIZE = 500


# In[ ]:


# detection
model = YOLO("../../storage/model/yolov5lu.pt")
columns = ["xmin", "ymin", "xmax", "ymax", "confidence", "label"]
prediction = model(PATH, verbose=False)
prediction = pd.DataFrame(prediction[0].boxes.data.cpu(), columns=columns)
prediction["label"] = prediction["label"].map(model.names)


# In[ ]:


# akaze
akaze = cv2.AKAZE_create()
image = cv2.cvtColor(cv2.imread(PATH), cv2.COLOR_BGR2RGB)
for _, bbox in prediction.iterrows():
    label = bbox["label"]
    bbox = bbox[["xmin", "ymin", "xmax", "ymax"]].values.astype(int)
    bbox = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    if RESIZE is not None:
        bbox = cv2.resize(bbox, (RESIZE, RESIZE))
    keypoint, description = akaze.detectAndCompute(bbox, None)
    keypoint_image = cv2.drawKeypoints(bbox, keypoint, None)
    plt.imshow(keypoint_image)
    plt.title(label)
    plt.show()
    plt.close()


# In[ ]:
