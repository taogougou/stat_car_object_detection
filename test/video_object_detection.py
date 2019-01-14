#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import colorsys
import os
import sys
import random
import copy
now_dir = os.path.dirname(__file__)
parent_path = os.path.dirname(now_dir)
sys.path.append(os.path.join(parent_path, "lib"))
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from model.config import cfg
from model.nms_wrapper import nms
from model.test import im_detect
from nets.resnet_v1 import resnetv1
from nets.vgg16 import vgg16
from utils.timer import Timer

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}


def detectObjectsFromVideo(sess, net, input_file_path="", output_file_path="", log_progress=True,detection_object_list = ["person","car"]):
    if (input_file_path == "" or output_file_path == ""):
        raise ValueError(
            "You must set 'input_file_path' to a valid video file, and the 'output_file_path' to path you want the detected video saved.")
    else:

        input_video = cv2.VideoCapture(input_file_path)
        output_video_filepath = output_file_path + '.avi'

        frame_width = int(input_video.get(3))
        frame_height = int(input_video.get(4))
        output_video = cv2.VideoWriter(output_video_filepath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 27,
                                       (frame_width, frame_height))

        counting = 1
        # 根据detection_object_list中的数据创建变量名，并初始化为0
        # createCount = locals()
        # for n in detection_object_list:
        #     createCount[n + "_count"] = 0


        # Exit if video not opened.
        if not input_video.isOpened():
            print("视频打开出错！！！")
            sys.exit()

        # 读取第一帧图片.
        ok1, frame1 = input_video.read()
        if not ok1:
            print('视频读取失败！！！')
            sys.exit()

        # 定义一个默认的统计区域框,//是向下取整
        stat_area = (0, frame_height//3, frame_width, frame_height//3 + 90)

        # 自己选择的统计框
        # stat_area = cv2.selectROI(frame1, False)
        # # 左下x，左下y，宽，高
        # x_left_bottom, y_left_bottom, width, height = stat_area
        # # 计算右上x和右上y
        # x_right_top, y_right_top = x_left_bottom + width, y_left_bottom - height
        x_left_top, y_left_top,x_right_bottom, y_right_bottom = stat_area


        print("选择的统计区域为:{}".format(stat_area))
        detected_copy1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)


        # 获取需要检测的类的dict

        det_dict1,inds1 = get_detection_boxes_scores(sess, net, detected_copy1, detection_object_list)
        #上一帧字典。建立box中心点坐标字典,格式为{'person': [center_point1,center_point], 'car': [center_point1,center_point]}
        dict_center_point_pre = {}
        # 根据detection_object_list中的数据创建变量名，并初始化为0
        createCount = locals()
        for n in detection_object_list:
            # createCount[n + "_count"] = 0
            det = det_dict1[n]
            #dict_center_point_pre内部的list，格式为[center_point1,center_point]
            dict_center_point_inner_pre = []
            for i in inds1:
                box = det[i, :4]


                # 计算中心点坐标,x,y

                #  x_left_top, y_left_top,x_right_bottom, y_right_bottom
                x_center_point_tmp,y_center_point_tmp = get_center_point(box)
                # draw_caption(detected_copy1, box, "{}".format((x_center_point_tmp,y_center_point_tmp)))
                # 画出统计区域
                draw_box(detected_copy1, stat_area, color=(0, 255, 0))

                if x_left_top <= x_center_point_tmp <= x_right_bottom and y_left_top <= y_center_point_tmp <= y_right_bottom:
                    dict_center_point_inner_pre.append((x_center_point_tmp,y_center_point_tmp))
                    draw_caption(detected_copy1, box, "{}".format((x_center_point_tmp, y_center_point_tmp)))
                # dict_center_point[n] = dict_center_point_inner
                #     createCount[n + "_count"] = createCount[n + "_count"] + 1
                    # 画框
                    draw_box(detected_copy1, box, color=(0, 255, 0))
                # plt_show(detected_copy1)
                #     plt_show(detected_copy1)
            dict_center_point_pre[n] = dict_center_point_inner_pre

            createCount[n + "_count"] = len(dict_center_point_pre[n])
        #距离的阈值，两帧的目标计算距离，由于帧速很快，所以同一目标在相邻的两帧中不可能相距太远的距离
        distance_thresh = 30


        while True:
            # 读取下一帧图片
            ok, frame = input_video.read()
            if not ok:
                break
            counting += 1
            # 当前帧字典。建立box中心点坐标字典,格式为{'person': [center_point1,center_point], 'car': [center_point1,center_point]}
            dict_center_point_now = copy.deepcopy(dict_center_point_pre)
            if (log_progress == True):
                print("Processing Frame :{} ".format(str(counting)))
            # detected_copy = frame.copy()
            # detected_copy_original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #保存一张原始帧，后面需要和处理后的图片就行上下拼接
            detected_copy_original = detected_copy.copy()

            #画出统计区域
            draw_box(detected_copy,stat_area,color=(0, 255, 0))

            # 获取要检测的图片中的scores和boxes

            # scores, boxes = im_detect(sess, net, detected_copy)

            # 获取需要检测的类的dict
            det_dict,inds = get_detection_boxes_scores(sess, net, detected_copy, detection_object_list)

            #显示统计的数量
            caption_count = ''

            for object_calss in detection_object_list:
                if object_calss in det_dict.keys():
                # if det_dict.has_key(object_calss):

                    det = det_dict[object_calss]
                    for i in inds:
                        box = det[i, :4]
                        # score = det[i, -1]
                        # 每个目标都打印出坐标
                        # draw_caption(detected_copy, box, "{}".format(get_center_point(box)))

                        # 计算中心点坐标
                        x_center_point, y_center_point = get_center_point(box)
                        # 每个目标都打印出中心的坐标
                        # tmp_center_point = get_center_point(box)
                        # draw_caption(detected_copy, box, "{}".format((x_center_point, y_center_point)))
                        # plt_show(detected_copy)



                        # 在统计的区域内才标注画框，区域外不标注画框
                        if x_left_top <= x_center_point <= x_right_bottom and y_left_top <= y_center_point <= y_right_bottom:

                            # 画框
                            draw_box(detected_copy, box, color=(0, 255, 0))
                            # 写类名字和得分
                            # caption = '{} {:.2f}'.format(object_calss, score) if score else object_calss
                            caption = '{}'.format(object_calss)
                            draw_caption(detected_copy, box, caption)
                            # plt_show(detected_copy)

                            #本帧图片中的目标中心点和上帧的目标中心点作比较，距离在阈值distance_thresh之内，则认为是同一目标
                            list_center_point_pre = dict_center_point_pre[object_calss]
                            #计算这个目标与上一帧图片所有目标距离大于阈值的数量，按理如果这个目标在上帧统计区域存在，则这个数量应该小于上帧目标总数量
                            # 如果数量等于上一帧的目标，则是新目标，需要计数 +1
                            # number_exceed_thresh = 0
                            #将所有求得的距离都放入distance_list中
                            distance_list = []
                            for j,center_point_pre in enumerate(list_center_point_pre) :
                                center_point_x_pre, center_point_y_pre = center_point_pre
                                #计算距离

                                distance = np.linalg.norm(
                                        np.array([center_point_x_pre, center_point_y_pre]) - np.array([x_center_point, y_center_point]))
                                distance_list.append(distance)

                            if min(distance_list) <= distance_thresh:
                                # 更新上一帧的中心点坐标
                                dict_center_point_now[object_calss][distance_list.index(min(distance_list))] = x_center_point,y_center_point
                            else:

                                #说明是一个新进入统计框的目标，则将其值添加到dict_center_point_now中
                                dict_center_point_now[object_calss].append((x_center_point, y_center_point))
                                # 统计 +1
                                createCount[object_calss + "_count"] = createCount[object_calss + "_count"] + 1

                        # # 每个目标都打印出中心的坐标
                        # tmp_center_point = get_center_point(box)
                        # draw_caption(detected_copy, box, "{}".format(tmp_center_point))
                        # plt_show(detected_copy)
                    dict_center_point_pre.clear()
                    dict_center_point_pre = copy.deepcopy(dict_center_point_now)
                    # # 统计dict_center_point[object_calss]中的元素个数即可
                    # createCount[object_calss + "_count"] = len(dict_center_point_now[object_calss])

                caption_count = caption_count  + '{}:{}'.format(object_calss + "_count",createCount[object_calss + "_count"])




            #将统计结果显示
            cv2.putText(detected_copy,caption_count,(40,100),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            plt_show(detected_copy)

            #将原视频和处理后的视频拼接到一起,这里拼接后的宽和高必须与原视频是一样的，否则合视频会出错！！
            detected_copy = np.concatenate([detected_copy_original[70:-241], detected_copy[70:-241]], axis=0)
            # plt_show(detected_copy)

            output_video.write(detected_copy)
        input_video.release()
        output_video.release()

        return output_video_filepath


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args
def get_center_point(box):
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    # 计算中心点坐标
    x_center_point, y_center_point = (x1 + x2) / 2, (y1 + y2) / 2
    return x_center_point, y_center_point
def get_detection_boxes_scores (sess,net,image,detection_object_list):
    # # 获取得分和框
    scores, boxes = im_detect(sess, net, image)
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    dict = {}

    for cls_ind, cls in enumerate(CLASSES[1:]):
        # 之检测需要检测的目标类
        if cls not in detection_object_list:
            continue
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        # keep = nms(cls_boxes, NMS_THRESH)
        dets = dets[keep, :]

        # 将框和标签画到图像上
        # 选取候选框score大于阈值CONF_THRESH的dets
        #返回的是满足条件元素的下标，并不是满足条件的元素值
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0:
            continue


        #符合条件的dets返回
        dict[cls] = dets

    return dict,inds

# 在一幅图片上进行目标检测并描框和打标签
def detect_image(sess, net, image,detection_object_list):
    """

    :param sess:
    :param net:
    :param image:
    :param detection_object_list: 要检测的目标，list  ["person","car"]
    :return:
    """
    # # 获取得分和框
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, image)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    #
    # image = image[:, :, (2, 1, 0)]
    # [(cls_ind, cls) for cls_ind, cls in enumerate(CLASSES[1:]) if cls == ""]
    for cls_ind,cls in enumerate(CLASSES[1:]):
        #之检测需要检测的目标类
        if cls not in detection_object_list:
            continue
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        # 将框和标签画到图像上
        # 选取候选框score大于阈值CONF_THRESH的dets
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0:
            continue

        # image = image[:, :, (2, 1, 0)]
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            # x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

            # 画框
            draw_box(image,bbox,color=(0,255,0))
            # 写类名字和得分
            caption = '{} {:.2f}'.format(cls, score) if score else cls
            draw_caption(image, bbox, caption)

    return image


# 随机产生颜色
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def draw_box(image, box, color, thickness=2):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    cv2.rectangle(image, (x1, y1), (x2, y2), color=color, lineType=cv2.LINE_AA)


def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.

    # Arguments
        image   : The image to draw on.
        box     : A list of 4 elements (x1, y1, x2, y2).
        caption : String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 3)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)


def plt_show(masked_image):
    _, ax = plt.subplots(1, figsize=(32, 32))
    height, width = masked_image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title("")
    ax.imshow(masked_image.astype(np.uint8))

    plt.show()


if __name__ == '__main__':

    # now_dir = os.path.dirname(__file__)
    # parent_path = os.path.dirname(now_dir)
    # 视频的输入地址
    video_input = parent_path + "/videos/test.mp4"
    # 视频的输出地址
    white_output = parent_path + '/videos/test_out'
    # detectObjectsFromVideo(video_input, white_output, log_progress=True)

    print("视频地址：{}".format(video_input))
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    dir_now = os.path.dirname(__file__)
    dir_root = os.path.dirname(dir_now)
    tfmodel = os.path.join(dir_root + '/output', demonet, DATASETS[dataset][0], 'default', NETS[demonet][0])
    print("tfmodel:{}".format(tfmodel))
    # tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',NETS[demonet][0])
    # tfmodel = "E:/machine-learning/tf-faster-rcnn-master/data/voc_2007_trainval+voc_2012_trainval/res101_faster_rcnn_iter_110000.ckpt"
    #
    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", 21, tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    #要检测的类
    detection_object_list = ["car"]

    detectObjectsFromVideo(sess, net, input_file_path=video_input, output_file_path=white_output,detection_object_list = detection_object_list)
    # plt.savefig(im_name + "_out.png")
    # plt.show()
