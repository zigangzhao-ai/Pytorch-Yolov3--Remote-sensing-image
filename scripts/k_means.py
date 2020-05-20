# coding: utf-8
# 获得anchors
from __future__ import division, print_function

import os
import os.path as path
import json
import math
import numpy as np
import generate_labels as tool


# 生成file_name文件，并追加写入name_list中内容
def write_file(file_name, name_list):
    base_name = path.basename(file_name)
    dir_name = file_name[:len(file_name)-len(base_name)]
    if not path.exists(dir_name):
        os.mkdir(dir_name)
    with open(file_name, 'a') as f:
        for name in name_list:
            f.write(name+'\n')
    return 

# 生成file_name文件，并重新写入name_list中内容
def rewrite_file(file_name, name_list):
    base_name = path.basename(file_name)
    dir_name = file_name[:len(file_name)-len(base_name)]
    if not path.exists(dir_name):
        os.mkdir(dir_name)
    with open(file_name, 'w') as f:
        for i in range(len(name_list)):
            f.write(str(i) + ' ' + name_list[i] +'\n')
    return 

# 解析一个points,得到坐标序列
def get_point(point_list):
    # point_list是一个包含两个坐标的list
    x_min = min(point_list[0][0], point_list[1][0])
    y_min = min(point_list[0][1], point_list[1][1])
    x_max = max(point_list[0][0], point_list[1][0])
    y_max = max(point_list[0][1], point_list[1][1])
    result = str(x_min) + ' ' + str(y_min) + ' ' + str(x_max) + ' ' + str(y_max)
    return result

# 解析json文件
def paras_json(json_file, class_name_file):
    if not path.exists(json_file):
        print("warning:不存在json文件" + str(json_file))
        assert(0)
    f = open(json_file)
    setting = json.loads(f.read())
    # print(setting)
    # f.close()
    shapes = setting['annotation'] 
    height = setting['annotation']['size']['height']
    width = setting['annotation']['size']['width']
    # 拿到标签坐标
    result = ""
    flag = 0
    count = 0

    if "object" not in [x for x in shapes]:
        
        result += ' ' + '0' + ' ' + get_point([[0,0],[0,0]])
        return str(width) + ' ' + str(height) +  result
    
    for shape in shapes['object']:
        count += 1
    a = [x for x in shapes['object']]
        # print(shape['name'])
    if count == 5 and a[0] == 'name' :
        flag = 1
    else:
        flag = count
    
    if flag == 1:
        shape = shapes['object']
        class_name = shape['name'] #得到分类名

        class_id = tool.get_id(class_name, class_name_file)
        point = []
        a = []
        b = []
        a.append(float(shape['bndbox']['xmin']))
        a.append(float(shape['bndbox']['ymin']))    
        b.append(float(shape['bndbox']['xmax']))
        b.append(float(shape['bndbox']['ymax']))
        point.append(a)
        point.append(b)
        # print(point)
        locate_result = get_point(point)
        
        result += ' ' + str(class_id) + ' ' + locate_result
        return str(width) + ' ' + str(height) + result

    if flag == count:
        for shape in shapes['object']:

            class_name = shape['name'] #得到分类名
            class_id = tool.get_id(class_name, class_name_file)
    
            point = []
            a = []
            b = []
            a.append(float(shape['bndbox']['xmin']))
            a.append(float(shape['bndbox']['ymin']))      
            b.append(float(shape['bndbox']['xmax']))
            b.append(float(shape['bndbox']['ymax']))

            point.append(a)
            point.append(b)
            # print(point)
            locate_result = get_point(point)
            result += ' ' + str(class_id) + ' ' + locate_result
        return str(width) + ' ' + str(height) + result
        
# 得到文件夹里面所有的图片路径
def get_pic_file(dir_path):
    result = []
    if not path.isdir(dir_path):
        print("exception: 路径%s不是文件夹" %(dir_path))
        return result
    # 读取图片路径
    for f in os.listdir(dir_path):
        curr_file = path.join(dir_path, f)
        if not path.isfile(curr_file):
            continue
        if not tool.isPic(f):
            continue
        result.append(curr_file)
    return result

# 生成k_means需要的数据
def generate_k_means_data(class_name='train.names', train_dir='./JPEGImages'):
    train_list = []
    img_index = 0
    dir_list = os.listdir(train_dir)
    for i in range(len(dir_list)):
        f = dir_list[i]
        curr_path = os.path.join(train_dir, f)
        if not path.isdir(curr_path):
            continue
        curr_dir_imgs = get_pic_file(curr_path)
        # 判断有没有对应的json文件,并解析json文件保存到list中
        for img_file in curr_dir_imgs:
            json_file = tool.has_json(img_file)
            if json_file:
                # 有这个json文件就保存这个img
                json_inf = paras_json(json_file, class_name)
                train_list.append(str(img_index) + ' ' + str(img_file) + ' ' + str(json_inf))
                img_index += 1
        print("\r文件夹处理进度：{:2f}%".format( (i+1)*100 / dir_list.__len__() ), end='')
    print()
    return train_list


# ################ k_means #############

def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    param:
        box: tuple or array, shifted to the origin (i. e. width and height)
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = np.true_divide(intersection, box_area + cluster_area - intersection + 1e-10)
    # iou_ = intersection / (box_area + cluster_area - intersection + 1e-10)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    param:
        boxes: numpy array of shape (r, 4)
    return:
    numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        k: number of clusters
        dist: distance function
    return:
        numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters

def parse_anno(data, target_size=None):
    result = []
    for line in data:

        s = line.strip().split(' ')
    
        if len(s[2])<=4:

            # img_w = int(s[2])
            # img_h = int(s[3])
            img_w = 1280
            img_h = 659
            s = s[4:]
            box_cnt = len(s) // 5
            if len(s)%5 != 0:
                print(s)
            for i in range(box_cnt):
                x_min, y_min, x_max, y_max = float(s[i*5+1]), float(s[i*5+2]), float(s[i*5+3]), float(s[i*5+4])
                width = x_max - x_min
                height = y_max - y_min
                if width == 0 or height == 0:
                    continue
                # assert width >= 0
                # assert height >= 0
                # use letterbox resize, i.e. keep the original aspect ratio
                # get k-means anchors on the resized target image size
                if target_size is not None and img_w != 0 and img_h != 0:
                    resize_ratio = min(target_size[0] / img_w, target_size[1] / img_h)
                    width *= resize_ratio
                    height *= resize_ratio
                    result.append([width, height])
            # get k-means anchors on the original image size
                else:
                    result.append([width, height])

        # print(s)
        else:
            # img_w = int(s[3])
            # img_h = int(s[4])
            img_w = 1280
            img_h = 659
            s = s[5:]

            box_cnt = len(s) // 5   
            for i in range(box_cnt):
                x_min, y_min, x_max, y_max = float(s[i*5+1]), float(s[i*5+2]), float(s[i*5+3]), float(s[i*5+4])
                width = x_max - x_min
                height = y_max - y_min

                if width == 0 or height == 0:
                    continue
                # assert width >= 0
                # assert height >= 0
                # use letterbox resize, i.e. keep the original aspect ratio
                # get k-means anchors on the resized target image size
                if target_size is not None and img_w != 0 and img_h != 0:
                    resize_ratio = min(target_size[0] / img_w, target_size[1] / img_h)
                    width *= resize_ratio
                    height *= resize_ratio
                    result.append([width, height])
                # get k-means anchors on the original image size
                else:
                    result.append([width, height])
    result = np.asarray(result)
    return result


def get_kmeans(anno, cluster_num=9):

    anchors = kmeans(anno, cluster_num)
    ave_iou = avg_iou(anno, anchors)

    anchors = anchors.astype('int').tolist()

    anchors = sorted(anchors, key=lambda x: x[0] * x[1])

    return anchors, ave_iou

# ############################

if __name__ == "__main__":
    # target resize format: [width, height]
    # if target_resize is speficied, the anchors are on the resized image scale
    # if target_resize is set to None, the anchors are on the original image scale
    # target_size = [416, 416]
    target_size = [416, 416]
    data = generate_k_means_data()
    # print(data)
    anno_result = parse_anno(data, target_size=target_size)

    anchors, ave_iou = get_kmeans(anno_result, 9)

    anchor_string = ''
    for anchor in anchors:
        anchor_string += '{},{}, '.format(anchor[0], anchor[1])
    anchor_string = anchor_string[:-2]

    print('anchors are:')
    print(anchor_string)
    print('the average iou is:')
    print(ave_iou)
    # generate_val_txt()
    pass
