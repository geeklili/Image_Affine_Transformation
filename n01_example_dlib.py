# -*- coding: utf-8 -*-

import sys

import dlib

from skimage import io

# 使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()

# 使用dlib提供的图片窗口
win = dlib.image_window()

# sys.argv[]是用来获取命令行参数的，sys.argv[0]表示代码本身文件路径，所以参数从1开始向后依次获取图片路径
for f in sys.argv[1:]:

    # 输出目前处理的图片地址
    print("Processing file: {}".format(f))

    # 使用skimage的io读取图片
    img = io.imread(f)

    # 使用detector进行人脸检测 dets为返回的结果
    dets = detector(img, 1)

    # dets的元素个数即为脸的个数
    print("Number of faces detected: {}".format(len(dets)))

    # 使用enumerate 函数遍历序列中的元素以及它们的下标
    # 下标i即为人脸序号
    # left：人脸左边距离图片左边界的距离 ；right：人脸右边距离图片左边界的距离
    # top：人脸上边距离图片上边界的距离 ；bottom：人脸下边距离图片上边界的距离
    for i, d in enumerate(dets):
        print("dets{}".format(d))
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}"
              .format(i, d.left(), d.top(), d.right(), d.bottom()))

    # 也可以获取比较全面的信息，如获取人脸与detector的匹配程度
    dets, scores, idx = detector.run(img, 1)
    for i, d in enumerate(dets):
        print("Detection {}, dets{},score: {}, face_type:{}".format(i, d, scores[i], idx[i]))

        # 绘制图片(dlib的ui库可以直接绘制dets)
    win.set_image(img)
    win.add_overlay(dets)

    # 等待点击
    dlib.hit_enter_to_continue()
