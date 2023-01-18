# 物体検出

import torch
import cv2
import numpy as np
from yolov5.utils.general import non_max_suppression


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# クラス
classes = {0: 'player', 1: 'ball'}

# カスタムモデルのロード
# model = torch.hub.load("ultralytics/yolov5", "custom", path='./weights/yolov5s.pt')
model = torch.hub.load("ultralytics/yolov5", "custom", path='./weights/best.pt')

# 閾値（score > 0.5 で検出）
model.conf = 0.5

# 物体検出関数
def detect(frame):
    """
        Input : BGR image
                    
        Output:
        以下のフォーマットの辞書のリストを返却:
            {   label   :  str
                bbox    :  [(xmin,ymin),(xmax,ymax)]
                score   :  float
                cls     :  int
            }
    """
    
    img = cv2.resize(frame, (640,384))

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()/255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    result = model(img)

    # print(type(result))
    # print(result)

    # pred = result.pandas().xyxy[0]
    # pred = result.pandas().xywhn[0]
    pred = non_max_suppression(result, 0.5, 0.5, classes=None)

    items = []

    if pred[0] is not None and len(pred):
        for p in pred[0]:
            if int(p[5]) in list(classes.keys()): 
                score = np.round(p[4].cpu().detach().numpy(),2)
                label = classes[int(p[5])]
                xmin = int(p[0] * frame.shape[1] /640)
                ymin = int(p[1] * frame.shape[0] /384)
                xmax = int(p[2] * frame.shape[1] /640)
                ymax = int(p[3] * frame.shape[0] /384)

                item = {'label': label,
                        'bbox' : [(xmin,ymin),(xmax,ymax)],
                        'score': score,
                        'cls' : int(p[5])}

                items.append(item)

    # for index, p in pred.iterrows():
    #     if int(p[5]) in list(classes.keys()): 
    #         score = p[4]
    #         label = classes[int(p[5])]
    #         xmin = int(p[0] * frame.shape[1] / 640)
    #         ymin = int(p[1] * frame.shape[0] / 384)
    #         xmax = int(p[2] * frame.shape[1] / 640)
    #         ymax = int(p[3] * frame.shape[0] / 384)

    #         item = {'label': label,
    #                 'bbox' : [(xmin,ymin),(xmax,ymax)],
    #                 'score': score,
    #                 'cls' : int(p[5])}

    #         items.append(item)

    return(items)