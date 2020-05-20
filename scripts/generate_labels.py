# coding:utf-8
# 生成darknet需要的标签

import os
import os.path as path
import json

# 保存所有类别对应的id的字典
class_id_dict = {}

# 判断是不是图片
def isPic(basename):
    file_type = basename.split('.')[-1]
    pic_file_list = ['png', 'jpg', 'jpeg', 'BMP', 'JPEG', 'JPG', 'JPeG', 'Jpeg', 'PNG', 'TIF', 'bmp', 'tif']
    if file_type in pic_file_list:
        return True
    return False

# 判断这个图片有没有对应的json文件
def has_json(img_file):
    # 得到json文件名
    base_name = path.basename(img_file)
    dir_name = img_file[:len(img_file) - len(base_name)]
    json_name = base_name.split('.')[0]
    json_name = json_name + '.json'
    json_name = path.join(dir_name, json_name)      
    if path.isfile(json_name):
        return json_name
    return None

# 生成file_name的label文件，并重新写入 content_list 中内容
def rewrite_labels_file(file_name, content_list):
    with open(file_name, 'w') as f:
        for line in content_list:
            curr_line_str = ''
            for element in line:
                curr_line_str += str(element) + ' '
            f.write(curr_line_str + '\n')
    return

# 生成file_name的训练图片路径文件
def rewrite_train_name_file(file_name, content_list):
    with open(file_name, 'w') as f:
        for line in content_list:
            f.write(str(line) + '\n')
    return 

# 读取文件
def read_file(file_name):
    if not path.exists(file_name):
        print("warning:不存在文件"+str(file_name))
        return None
    with open(file_name, 'r', encoding='utf-8') as f:
        result = []
        for line in f.readlines():
            result.append(line.strip('\n'))
        return result

# 加载class_id
def load_class_id(class_name_file):
    global class_id_dict
    class_list = read_file(class_name_file)
    for i in range(len(class_list)):
        class_id_dict[str(class_list[i])] = i
    return class_id_dict

# 得到分类的id,未分类是-1
def get_id(class_name, class_name_file):
    global class_id_dict
    if len(class_id_dict) < 1:
        class_id_dict = load_class_id(class_name_file)
        print("分类 id 加载完成")
    # 补丁:替换掉汉字 "局""段"
    class_name = get_id_patch(class_name)
    if class_name in class_id_dict.keys():
        return class_id_dict[class_name]
    return -1
# 去掉汉字'段'和'局'
def get_id_patch(class_name):
    if class_name.strip() == '段':
        return 'duan'
    if class_name.strip() == '局':
        return 'ju'
    return class_name

# 解析一个points,得到坐标序列
def get_relative_point(img_width, img_height, point_list):
    # point_list是一个包含两个坐标的list

    dh = 1.0/ img_height
    x_min = min(point_list[0][0], point_list[1][0])
    y_min = min(point_list[0][1], point_list[1][1])
    x_max = max(point_list[0][0], point_list[1][0])
    y_max = max(point_list[0][1], point_list[1][1])
    dw = 1.0 / img_width
    dh = 1.0/ img_height
    # 中心坐标
    x = (x_min + x_max)/2.0
    y = (y_min + y_max)/2.0
    w = x_max - x_min
    h = y_max - y_min
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return [x, y, w, h]

# 解析json文件
def paras_json(json_file, class_name_file):
    if not path.exists(json_file):
        print("warning:不存在json文件" + str(json_file))
        assert(0)
    # 读取json文件拿到基本信息, encoding要注意一下
    # try:
    #     f = open(json_file, encoding="gbk")
    #     setting = json.loads(f.read())
    # except:
    #     f = open(json_file, encoding='utf-8')
    #     setting = json.loads(f.read())
    f = open(json_file)
    setting = json.loads(f.read())
    # print(setting)
    # f.close()
    shapes = setting['annotation'] 
    height = float(setting['annotation']['size']['height']) ##1280
    width = float(setting['annotation']['size']['width'])   ##659
    # 拿到标签坐标
    result = []
    flag = 0
    count = 0

    if "object" not in [x for x in shapes]:
        return [[0,0,0,0,0]]
    
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
        class_id = get_id(class_name, class_name_file)

        point = []
        a = []
        b = []
        a.append(float(shape['bndbox']['xmin']))
        a.append(float(shape['bndbox']['ymin']))
        
        b.append(float(shape['bndbox']['xmax']))
        b.append(float(shape['bndbox']['ymax']))
        point.append(a)
        point.append(b)
        print(point)
        locate_result = get_relative_point(1280, 659, point)
        locate_result.insert(0, class_id)
        result.append(locate_result)
        return result

    if flag == count:

        for shape in shapes['object']:

            class_name = shape['name'] #得到分类名
            class_id = get_id(class_name, class_name_file)
    
            point = []
            a = []
            b = []
            a.append(float(shape['bndbox']['xmin']))
            a.append(float(shape['bndbox']['ymin']))
        
            b.append(float(shape['bndbox']['xmax']))
            b.append(float(shape['bndbox']['ymax']))

            point.append(a)
            point.append(b)
            print(point)

            locate_result = get_relative_point(1280, 659, point)
                # 插入id
            locate_result.insert(0, class_id)
            result.append(locate_result)
        print(result)
        return result
    # else:
    #     return None

# 得到文件夹下所有的图片文件
def get_pic_file_from_dir(dir_name):
    '''
        return:所有的图片文件名
    '''
    if not path.isdir(dir_name):
        print("warning:路径 %s 不是文件夹" %dir_name)
        return []
    result = []
    for f in os.listdir(dir_name):
        curr_file = path.join(dir_name, f)
        if not path.isfile(curr_file):
            continue
        if not isPic(curr_file):
            continue
        result.append(f)
    return result

def main(class_name="classes.names", img_dir="images/", train_txt='train.txt', labels_dir='labels'):
    
    cwd = os.getcwd()
    img_dir = path.join(cwd, img_dir)
    
    # print(img_dir)
    labels_dir = path.join(cwd, labels_dir)

    if not path.exists(img_dir):
        print("error:没有发现图片文件夹 ", img_dir)

    if not path.exists(labels_dir):
        os.mkdir(labels_dir)
    
    count = 0                                                 
    dir_len = len(os.listdir(img_dir))  # 进度条
    # print(dir_len)
    imgs = []
    for f in os.listdir(img_dir): 
        # print(f)
        curr_path = path.join(img_dir, f)
        # print(curr_path)
        if not path.isdir(curr_path):   # 不是文件夹就先跳过
            continue
        curr_train_dir = curr_path
        # print(curr_train_dir)
        # 是文件夹就创建labels对应的文件夹
        curr_labels_dir = path.join(labels_dir, f)
        if not path.isdir(curr_labels_dir):
            os.mkdir(curr_labels_dir)
        # 拿到文件夹下所有的图片文件
        curr_dir_imgs = get_pic_file_from_dir(curr_train_dir)
        # print(curr_dir_imgs)
        # 解析这些图片的json文件
        for img_file in curr_dir_imgs:
            curr_img_file = path.join(curr_train_dir, img_file)
            # print(curr_img_file)
            json_file = has_json(curr_img_file)
            print(json_file)
            if json_file:
                # 保存图片路径
                imgs.append(curr_img_file)
                # 得到json信息 list
                json_inf = paras_json(json_file, class_name)
                # print(json_inf)
                # 标签文件名
                label_name = img_file.split('/')[-1].split('.')[0] + '.txt'
                curr_labels_file = path.join(curr_labels_dir, label_name)
                # 写入标签
                rewrite_labels_file(curr_labels_file, json_inf)
        count += 1
        print("\r当前进度: {:02f} %".format(count/dir_len * 100.0), end='')
    print("\n 保存训练图片路径到: ", train_txt)
    rewrite_train_name_file(train_txt, imgs)
    return 

if __name__ == "__main__":
    main()