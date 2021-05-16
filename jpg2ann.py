import os
import cv2
import json

data_dir = './data/test/image/test-B-image'                # 根目录文件，其中包含image文件夹和box文件夹（根据自己的情况修改这个路径）
image_file_dir = os.path.join(data_dir)   # 相当于image_file_dir='./data/train/image'，图片存放的路径（根据自己的情况修改）
print(image_file_dir)

annotations_info = {'images': [],'annotations': [], 'categories': [{"id": 1,"name": "holothurian"},
                                                                   {"id": 2,"name": "echinus"},
                                                                   {"id": 3,"name": "scallop"},
                                                                   {"id": 4,"name": "starfish"}]
                    }
file_names = [image_file_name.split('.')[0] for image_file_name in os.listdir(image_file_dir)]
file_names.sort()

for i, file_name in enumerate(file_names):  # 进入一个循环，对所有图片进行处理
    image_file_name = file_name + '.jpg'    # 得到一张图片的名字
    print(image_file_name)

    image_info = {'file_name': image_file_name, 'id': i+1}
    annotations_info['images'].append(image_info)

with  open('./data/test/annotations/test.json', 'w')  as f:         # 将imagesannotations和categories信息写入json文件
    json.dump(annotations_info, f, indent=4)