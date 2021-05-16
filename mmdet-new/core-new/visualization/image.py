# # Copyright (c) Open-MMLab. All rights reserved.
# import cv2
# import numpy as np
#
# from mmcv.image import imread, imwrite
# from .color import color_val
#
# # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# # txt文件用的
# # file_handle=open('/home/zjy/mmdetection/result/result.txt',mode='w')
# # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# # csv
# import csv
# import os
#
# f = open('/home/zjy/mmdetection/result/result.csv', 'w', encoding='utf-8')
# csv_writer = csv.writer(f)
#
# csv_writer.writerow(['name', 'image_id', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'])
#
#
# ################################################################################
#
# def imshow(img, win_name='', wait_time=0):
#     """Show an image.
#
#     Args:
#         img (str or ndarray): The image to be displayed.
#         win_name (str): The window name.
#         wait_time (int): Value of waitKey param.
#     """
#     cv2.imshow(win_name, imread(img))
#     if wait_time == 0:  # prevent from hangning if windows was closed
#         while True:
#             ret = cv2.waitKey(1)
#
#             closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
#             # if user closed window or if some key pressed
#             if closed or ret != -1:
#                 break
#     else:
#         ret = cv2.waitKey(wait_time)
#
#
# def imshow_bboxes(img,
#                   bboxes,
#                   colors='green',
#                   top_k=-1,
#                   thickness=1,
#                   show=True,
#                   win_name='',
#                   wait_time=0,
#                   out_file=None):
#     """Draw bboxes on an image.
#
#     Args:
#         img (str or ndarray): The image to be displayed.
#         bboxes (list or ndarray): A list of ndarray of shape (k, 4).
#         colors (list[str or tuple or Color]): A list of colors.
#         top_k (int): Plot the first k bboxes only if set positive.
#         thickness (int): Thickness of lines.
#         show (bool): Whether to show the image.
#         win_name (str): The window name.
#         wait_time (int): Value of waitKey param.
#         out_file (str, optional): The filename to write the image.
#
#     Returns:
#         ndarray: The image with bboxes drawn on it.
#     """
#     img = imread(img)
#
#     if isinstance(bboxes, np.ndarray):
#         bboxes = [bboxes]
#     if not isinstance(colors, list):
#         colors = [colors for _ in range(len(bboxes))]
#     colors = [color_val(c) for c in colors]
#     assert len(bboxes) == len(colors)
#
#     for i, _bboxes in enumerate(bboxes):
#         _bboxes = _bboxes.astype(np.int32)
#         if top_k <= 0:
#             _top_k = _bboxes.shape[0]
#         else:
#             _top_k = min(top_k, _bboxes.shape[0])
#         for j in range(_top_k):
#             left_top = (_bboxes[j, 0], _bboxes[j, 1])
#             right_bottom = (_bboxes[j, 2], _bboxes[j, 3])
#             cv2.rectangle(
#                 img, left_top, right_bottom, colors[i], thickness=thickness)
#
#     if show:
#         imshow(img, win_name, wait_time)
#     if out_file is not None:
#         imwrite(img, out_file)
#     return img
#
#
# def imshow_det_bboxes(img,
#                       bboxes,
#                       labels,
#                       class_names=None,
#                       score_thr=0.5,  # 本来0.3
#                       bbox_color='green',
#                       text_color='green',
#                       thickness=1,
#                       font_scale=0.5,
#                       show=True,
#                       win_name='',
#                       wait_time=0,
#                       out_file=None):
#     """Draw bboxes and class labels (with scores) on an image.
#
#     Args:
#         img (str or ndarray): The image to be displayed.
#         bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
#             (n, 5).
#         labels (ndarray): Labels of bboxes.
#         class_names (list[str]): Names of each classes.
#         score_thr (float): Minimum score of bboxes to be shown.
#         bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
#         text_color (str or tuple or :obj:`Color`): Color of texts.
#         thickness (int): Thickness of lines.
#         font_scale (float): Font scales of texts.
#         show (bool): Whether to show the image.
#         win_name (str): The window name.
#         wait_time (int): Value of waitKey param.
#         out_file (str or None): The filename to write the image.
#
#     Returns:
#         ndarray: The image with bboxes drawn on it.
#     """
#
#     assert bboxes.ndim == 2
#     assert labels.ndim == 1
#     assert bboxes.shape[0] == labels.shape[0]
#     assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
#     # print('\n',out_file[-10:-4],' ',)  # image_id在他妈这里！！！！！！！！！！！！！！！！！！！！！
#
#     img = imread(img)
#     if score_thr > 0:
#         assert bboxes.shape[1] == 5
#         scores = bboxes[:, -1]
#         inds = scores > score_thr
#         bboxes = bboxes[inds, :]
#         labels = labels[inds]
#
#     bbox_color = color_val(bbox_color)
#     text_color = color_val(text_color)
#     img = np.ascontiguousarray(img)
#     for bbox, label in zip(bboxes, labels):
#         # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#         # # txt文件用的
#         # # print('\n', out_file[-10:-4], ' ', label+1, ' ',bbox[4],' ', bbox[0], ' ', bbox[1], ' ', bbox[2], ' ', bbox[3])
#         # file_handle.write(str(str(out_file[-10:-4])+' '+str(label+1)+' '+str(bbox[4])+' '+str(bbox[0])+' '+
#         #                       str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+'\n'))
#         # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#         # # csv文件---光学版
#         # if label==0:
#         #     name='holothurian'
#         # elif label==1:
#         #     name='echinus'
#         # elif label==2:
#         #     name='scallop'
#         # else:
#         #     name='starfish'
#         # 声学版
#         if label == 0:
#             name = 'cube'
#         elif label == 1:
#             name = 'ball'
#         elif label == 2:
#             name = 'cylinder'
#         elif label == 3:
#             name = 'human body'
#         elif label == 4:
#             name = 'tyre'
#         elif label == 5:
#             name = 'circle cage'
#         elif label == 6:
#             name = 'square cage'
#         else:
#             name = 'metal bucket'
#         csv_writer.writerow(
#             [name, str(os.path.basename(out_file)[:-4]), bbox[4], int(bbox[0]), int(bbox[1]), int(bbox[2]),
#              int(bbox[3])])
#         # csv_writer.writerow([name,str(out_file[-10:-4]),bbox[4],int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])])
#         ###########################################################################################
#         bbox_int = bbox.astype(np.int32)
#         left_top = (bbox_int[0], bbox_int[1])  # xmin,ymin
#         right_bottom = (bbox_int[2], bbox_int[3])  # xmax,ymax
#         cv2.rectangle(
#             img, left_top, right_bottom, bbox_color, thickness=thickness)
#         label_text = class_names[
#             label] if class_names is not None else f'cls {label}'
#         if len(bbox) > 4:
#             label_text += f'|{bbox[-1]:.02f}'
#         cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
#                     cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
#
#     if show:
#         imshow(img, win_name, wait_time)
#     if out_file is not None:
#         imwrite(img, out_file)
#     return img

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pycocotools.mask as mask_util
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from ..utils import mask2ndarray
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# txt文件用的
# file_handle=open('/home/zjy/mmdetection/result/result.txt',mode='w')
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# csv
import csv
import os

# f = open('./work_dirs/B-newnew.csv', 'w', encoding='utf-8')
# csv_writer = csv.writer(f)
#
# csv_writer.writerow(['name', 'image_id', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'])


################################################################################

EPS = 1e-2

# file_handle = open("./work_dirs/result4.26newdata.txt", mode='w')

def color_val_matplotlib(color):
    """Convert various input in BGR order to normalized RGB matplotlib color
    tuples,

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[float]: A tuple of 3 normalized floats indicating RGB channels.
    """
    color = mmcv.color_val(color)
    color = [color / 255 for color in color[::-1]]
    return tuple(color)


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      segms=None,
                      class_names=None,
                      score_thr=0,
                      bbox_color='red',
                      text_color='red',
                      mask_color=None,
                      thickness=2,
                      font_size=25,
                      win_name='',
                      show=False,
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        segms (ndarray or None): Masks, shaped (n,h,w) or None
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.  Default: 0
        bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: 'green'
        text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: 'green'
        mask_color (str or tuple(int) or :obj:`Color`, optional):
           Color of masks. The tuple of color should be in BGR order.
           Default: None
        thickness (int): Thickness of lines. Default: 2
        font_size (int): Font size of texts. Default: 13
        show (bool): Whether to show the image. Default: True
        win_name (str): The window name. Default: ''
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes.shape[0] == labels.shape[0], \
        'bboxes.shape[0] and labels.shape[0] should have the same length.'
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    img = mmcv.imread(img).astype(np.uint8)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    mask_colors = []
    if labels.shape[0] > 0:
        if mask_color is None:
            # random color
            np.random.seed(42)
            mask_colors = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
        else:
            # specify  color
            mask_colors = [
                np.array(mmcv.color_val(mask_color)[::-1], dtype=np.uint8)
            ] * (
                max(labels) + 1)

    bbox_color = color_val_matplotlib(bbox_color)
    text_color = color_val_matplotlib(text_color)

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    polygons = []
    color = []
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):

        #         # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #         # # txt文件用的
        #         # # print('\n', out_file[-10:-4], ' ', label+1, ' ',bbox[4],' ', bbox[0], ' ', bbox[1], ' ', bbox[2], ' ', bbox[3])
        #         # file_handle.write(str(str(out_file[-10:-4])+' '+str(label+1)+' '+str(bbox[4])+' '+str(bbox[0])+' '+
        #         #                       str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+'\n'))
        #         # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # csv文件---光学版
        # if label==0:
        #     name='holothurian'
        # elif label==1:
        #     name='echinus'
        # elif label==2:
        #     name='scallop'
        # else:
        #     name='starfish'
        #         # 声学版
        # if label == 0:
        #     name = 'cube'
        # elif label == 1:
        #     name = 'ball'
        # elif label == 2:
        #     name = 'cylinder'
        # elif label == 3:
        #     name = 'human body'
        # elif label == 4:
        #     name = 'tyre'
        # elif label == 5:
        #     name = 'circle cage'
        # elif label == 6:
        #     name = 'square cage'
        # else:
        #     name = 'metal bucket'
        # csv_writer.writerow(
        #     [name, str(os.path.basename(out_file)[:-4]), bbox[4], int(bbox[0]), int(bbox[1]), int(bbox[2]),
        #      int(bbox[3])])
        # csv_writer.writerow([name,str(out_file[-10:-4]),bbox[4],int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])])
        #         ###########################################################################################
        # csv_writer.writerow([name,str(out_file[-10:-4]),bbox[4],bbox[0],bbox[1],bbox[2],bbox[3]])

        # csv_writer.writerow([name,str(out_file[-10:-4]),bbox[4],round(bbox[0]),round(bbox[1]),round(bbox[2]),round(bbox[3])])

        bbox_int = bbox.astype(np.int32)



        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(bbox_color)
        label_text = class_names[
            label] if class_names is not None else f'class {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        ax.text(
            bbox_int[0],
            bbox_int[1],
            f'{label_text}',
            bbox={
                'facecolor': 'black',
                'alpha': 0.8,
                'pad': 0.7,
                'edgecolor': 'none'
            },
            color=text_color,
            fontsize=font_size,
            verticalalignment='top',
            horizontalalignment='left')
        if segms is not None:
            color_mask = mask_colors[labels[i]]
            mask = segms[i].astype(bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5

    plt.imshow(img)

    p = PatchCollection(
        polygons, facecolor='none', edgecolors=color, linewidths=thickness)
    ax.add_collection(p)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        # mmcv.imwrite(img, out_file)
        pass

    plt.close()

    return img


def imshow_gt_det_bboxes(img,
                         annotation,
                         result,
                         class_names=None,
                         score_thr=0,
                         gt_bbox_color=(255, 102, 61),
                         gt_text_color=(255, 102, 61),
                         gt_mask_color=(255, 102, 61),
                         det_bbox_color=(72, 101, 241),
                         det_text_color=(72, 101, 241),
                         det_mask_color=(72, 101, 241),
                         thickness=2,
                         font_size=13,
                         win_name='',
                         show=False,
                         wait_time=0,
                         out_file=None):
    """General visualization GT and result function.

    Args:
      img (str or ndarray): The image to be displayed.)
      annotation (dict): Ground truth annotations where contain keys of
          'gt_bboxes' and 'gt_labels' or 'gt_masks'
      result (tuple[list] or list): The detection result, can be either
          (bbox, segm) or just bbox.
      class_names (list[str]): Names of each classes.
      score_thr (float): Minimum score of bboxes to be shown.  Default: 0
      gt_bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: (255, 102, 61)
      gt_text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: (255, 102, 61)
      gt_mask_color (str or tuple(int) or :obj:`Color`, optional):
           Color of masks. The tuple of color should be in BGR order.
           Default: (255, 102, 61)
      det_bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: (72, 101, 241)
      det_text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: (72, 101, 241)
      det_mask_color (str or tuple(int) or :obj:`Color`, optional):
           Color of masks. The tuple of color should be in BGR order.
           Default: (72, 101, 241)
      thickness (int): Thickness of lines. Default: 2
      font_size (int): Font size of texts. Default: 13
      win_name (str): The window name. Default: ''
      show (bool): Whether to show the image. Default: True
      wait_time (float): Value of waitKey param. Default: 0.
      out_file (str, optional): The filename to write the image.
         Default: None

    Returns:
        ndarray: The image with bboxes or masks drawn on it.
    """
    assert 'gt_bboxes' in annotation
    assert 'gt_labels' in annotation
    assert isinstance(
        result,
        (tuple, list)), f'Expected tuple or list, but get {type(result)}'

    gt_masks = annotation.get('gt_masks', None)
    if gt_masks is not None:
        gt_masks = mask2ndarray(gt_masks)

    img = mmcv.imread(img)

    img = imshow_det_bboxes(
        img,
        annotation['gt_bboxes'],
        annotation['gt_labels'],
        gt_masks,
        class_names=class_names,
        bbox_color=gt_bbox_color,
        text_color=gt_text_color,
        mask_color=gt_mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=False)

    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None

    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        segms = mask_util.decode(segms)
        segms = segms.transpose(2, 0, 1)

    img = imshow_det_bboxes(
        img,
        bboxes,
        labels,
        segms=segms,
        class_names=class_names,
        score_thr=score_thr,
        bbox_color=det_bbox_color,
        text_color=det_text_color,
        mask_color=det_mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=show,
        wait_time=wait_time,
        out_file=out_file)
    return img
