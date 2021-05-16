import os
import cv2
import json
import xml.dom.minidom                   # 这两条代码是引入对xml文件进行处理的包
import xml.etree.ElementTree as ET

data_dir = '../mmdetection-master/data/1'                # 根目录文件，其中包含image文件夹和box文件夹（根据自己的情况修改这个路径）

image_file_dir = os.path.join(data_dir, 'image')   # 相当于image_file_dir='./data/train/image'，图片存放的路径（根据自己的情况修改）
xml_file_dir = os.path.join(data_dir, 'box')       # 相当于image_file_dir='./data/train/box'，目标框真值文件（根据自己的情况修改）

annotations_info = {'images': [], 'annotations': [], 'categories': []}

categories_map = {'holothurian': 1, 'echinus': 2, 'scallop': 3, 'starfish': 4}
# json文件的categories信息，请按照这个格式修改成您的目标类别（根据自己的情况修改）

for key in categories_map:
    categoriy_info = {"id":categories_map[key], "name":key}
    annotations_info['categories'].append(categoriy_info)

file_names = [image_file_name.split('.')[0] for image_file_name in os.listdir(image_file_dir)]
# 上面这条代码是得到每张图片去掉后缀名的名字

ann_id = 1
for i, file_name in enumerate(file_names):  # 进入一个循环，对所有图片进行处理


    image_file_name = file_name + '.jpg'    # 得到一张图片的名字
    print(image_file_name)
    xml_file_name = file_name + '.xml'      # 得到图片对应的真值文件的名字
    image_file_path = os.path.join(image_file_dir, image_file_name)  # 得到图片的绝对路径
    print(image_file_path)
    xml_file_path = os.path.join(xml_file_dir, xml_file_name)        # 得到图片的真值文件的绝对路径
    image_info = dict()
    image = cv2.cvtColor(cv2.imread(image_file_path), cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    image_info = {'file_name': image_file_name, 'id': i+1,           # json文件的image信息
                  'height': height, 'width': width}
    annotations_info['images'].append(image_info)

    DOMTree = xml.dom.minidom.parse(xml_file_path)                   # 打开目标框真值文件，您可以根据您的真值文件格式进行修改
    collection = DOMTree.documentElement                             # 得到目标框真值文件的信息，您可以根据您的真值文件格式进行修改

    names = collection.getElementsByTagName('name')                  # 找到所有目标框，对应于我们上面提到的xml文件中的<name>
    names = [name.firstChild.data for name in names]

    xmins = collection.getElementsByTagName('xmin')                  # 这8行代码是得到一张图片所有目标框的位置信息
    xmins = [xmin.firstChild.data for xmin in xmins]
    ymins = collection.getElementsByTagName('ymin')
    ymins = [ymin.firstChild.data for ymin in ymins]
    xmaxs = collection.getElementsByTagName('xmax')
    xmaxs = [xmax.firstChild.data for xmax in xmaxs]
    ymaxs = collection.getElementsByTagName('ymax')
    ymaxs = [ymax.firstChild.data for ymax in ymaxs]

    object_num = len(names)                                          # object_num记录一张图片中目标框到的个数

    for j in range(object_num):                                      # 对每个目标框的信息进行处理后再写入json文件
        if names[j] in categories_map:
            image_id = i + 1                                         # 目标框对应的图片的id:image_id
            x1,y1,x2,y2 = int(xmins[j]),int(ymins[j]),int(xmaxs[j]),int(ymaxs[j])  # 这四行代码是将目标框的位置信息转换成
            x1,y1,x2,y2 = x1,y1,x2,y2                                              # json格式要求的[x,y,w,h]格式
            x,y = x1,y1
            w,h = x2 - x1,y2 - y1
            category_id = categories_map[names[j]]                    # 目标框对应的类别的id:categories_id

            segmentation =  [x, y, w + x, w, w + x, h + y, x, h + y]  # segmentation是目标检测不需要的信息，随便填充的
            area = w * h                                              # 目标框的面积信息：area
            annotation_info = {"segmentation": segmentation,"id": ann_id, "image_id":image_id, "bbox":[x, y, w, h], "category_id": category_id, "area": area,"iscrowd": 0}                # 记录每个目标框的信息
            annotations_info['annotations'].append(annotation_info)
            ann_id += 1
with  open('./data/1/annotations/train.json', 'w')  as f:         # 将imagesannotations和categories信息写入json文件
    json.dump(annotations_info, f, indent=4)