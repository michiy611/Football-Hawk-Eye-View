# K-means法による色検知

import cv2
from sklearn.cluster import KMeans
import numpy as np

pallete = {'b': (0, 0, 128),
        'g': (0, 128, 0),
        'r': (255, 0, 0),
        'c': (0, 192, 192),
        'm': (192, 0, 192),
        'y': (192, 192, 0),
        'k': (0, 0, 0),
        'w': (255, 255, 255)}


def detect_color(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[1]*img.shape[0],3))

    kmeans = KMeans(n_clusters=2)
    s = kmeans.fit(img)

    labels = kmeans.labels_
    centroid = kmeans.cluster_centers_
    labels = list(labels)
    percent=[]
    
    for i in range(len(centroid)):
        j=labels.count(i)
        j=j/(len(labels))
        percent.append(j)

    detected_color = centroid[np.argmin(percent)]
    
    list_of_colors = list(pallete.values())
    assigned_color = closest_color(list_of_colors, detected_color)[0]
    assigned_color = (int(assigned_color[2]), int(assigned_color[1]), int(assigned_color[0]))

    if assigned_color == (0, 0, 0):
        assigned_color = (128, 128, 128)

    return assigned_color


# 検出された色に最も近い色を、あらかじめ定義されたパレットをもとに検索
def closest_color(list_of_colors, color):
    colors = np.array(list_of_colors)
    color = np.array(color)
    distances = np.sqrt(np.sum((colors-color)**2,axis=1))
    index_of_shortest = np.where(distances==np.amin(distances))
    shortest_distance = colors[index_of_shortest]

    return shortest_distance 