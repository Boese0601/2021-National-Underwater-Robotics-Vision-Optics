# 2021-National-Underwater-Robotics-Vision-Optics
2021年全国水下机器人算法大赛-光学赛道 -B榜精度18名（A榜map@50:95   56.36         B榜map@50:65 56.7）
## 请按照mmdetection官方文档配置环境，并运行Config文件对应的模型

## 代码内容和trick：

+ 基本网络模型
  + cascade rcnn
  + resnext101 pretrained on COCO
  + soft nms
  + 基于mmdetection
  + mmcv==0.2.16
+ 添加的trick
  + dcn
  + global_context(gcb)
  + RandomRotate90
  + cutout
  + Mixup
  +  边框抖动
  +  高斯噪声椒盐噪声
  +  Libra RCNN
  +  GIoU/CIoU/DIoU Loss
  +  Attention Block
  +  Multi-scale Training and Testing

# Contact me

Email：2862588711cd@gmailcom
