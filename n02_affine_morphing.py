import os
import imageio
import numpy as np
import dlib
import cv2
import sys


def find_index_in_landmark(np_array, val):
    for i in range(0, np_array.shape[0]):
        if np_array[i, 0] == int(val[0]) and np_array[i, 1] == int(val[1]):
            return i
    return -1


def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


def get_trianle_image(img_from, tr1, tr2):
    imgTo = np.zeros_like(img_from)
    r1 = cv2.boundingRect(tr1)
    r2 = cv2.boundingRect(tr2)
    tri1_cropped = []
    tri2_cropped = []
    for i in range(0, 3):
        tri1_cropped.append(((tr1[0][i][0] - r1[0]), (tr1[0][i][1] - r1[1])))
        tri2_cropped.append(((tr2[0][i][0] - r2[0]), (tr2[0][i][1] - r2[1])))
    img1_cropped = img_from[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    warp_mat = cv2.getAffineTransform(np.float32(tri1_cropped), np.float32(tri2_cropped))
    img2_cropped = cv2.warpAffine(img1_cropped, warp_mat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri2_cropped), (1.0, 1.0, 1.0), 16, 0);
    img2_cropped = img2_cropped * mask
    imgTo[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = imgTo[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
            (1.0, 1.0, 1.0) - mask)
    imgTo[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = imgTo[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2_cropped
    return imgTo


def img_add_no_zero(img1, img2):
    img3 = img1 + img2
    img3_overlay_pixels = np.transpose(np.nonzero(img1 * img2))
    for i in img3_overlay_pixels:
        img3[i[0]][i[1]] = img1[i[0]][i[1]] / 2 + img2[i[0]][i[1]] / 2
    return img3


def get_landmarks(img_path, dlib_detector, dlib_predictor):
    """获取人脸识别后图片的特征点
    """
    img = cv2.imread(img_path)
    # 使用detector进行人脸检测 dets为返回的结果
    dets = dlib_detector(img, 1)
    # dets的元素个数即为脸的个数, 列表里是脸的左上角位置和右下角的位置
    # print(dets)

    # 使用enumerate 函数遍历序列中的元素以及它们的下标
    # 下标i即为人脸序号
    # left：人脸左边距离图片左边界的距离 ；right：人脸右边距离图片左边界的距离
    # top：人脸上边距离图片上边界的距离 ；bottom：人脸下边距离图片上边界的距离
    # for i, d in enumerate(dets):
    #     print("dets{}".format(d))
    #     print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}"
    #           .format(i, d.left(), d.top(), d.right(), d.bottom()))
    if len(dets) != 1:
        print("Face num not 1!")
        return []

    # 特征点全部保存在了shape里面，d是dlib.rectangle()，里面保存着人脸检测矩形的左上和右下坐标，shape.part(i)是第i个特征点
    shape = dlib_predictor(img, dets[0])
    # print(shape.part(1))

    # 特征点列表
    key_points = []
    for j in range(0, shape.num_parts):
        key_points.append((shape.part(j).x, shape.part(j).y))
    A = img.shape
    # print(key_points)
    key_points.append((0, 0))
    key_points.append((0, A[0] - 1))
    key_points.append((A[1] - 1, 0))
    key_points.append((A[1] - 1, A[0] - 1))
    key_points.append((0, (A[0] - 1) / 2))
    key_points.append(((A[1] - 1) / 2, 0))
    key_points.append((A[1] - 1, (A[0] - 1) / 2))
    key_points.append(((A[1] - 1) / 2, A[0] - 1))
    # print(key_points)
    return key_points


def generate_morphing_image(img1, img2, lm1, lm2, alpha):
    lm3 = lm1 * alpha + lm2 * (1 - alpha)
    lm3 = lm3.astype("uint32")
    for i in range(0, len(lm3)):
        lm3[i] = (lm3[i, 0], lm3[i, 1])
    A = img1.shape
    rect = (0, 0, A[1], A[0])
    subdiv = cv2.Subdiv2D(rect)
    for i in lm3:
        subdiv.insert((i[0], i[1]))
    triangles3 = subdiv.getTriangleList()
    triangleAsId = []
    for i in triangles3:
        pt1 = (i[0], i[1])
        pt2 = (i[2], i[3])
        pt3 = (i[4], i[5])
        if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
            triangleAsId.append(
                [find_index_in_landmark(lm3, pt1), find_index_in_landmark(lm3, pt2), find_index_in_landmark(lm3, pt3)])
    # Test affine transform
    img_morphing_from1 = np.zeros_like(img1)
    img_morphing_from2 = np.zeros_like(img2)
    for i in range(0, len(triangleAsId)):
        tr11 = np.float32([[[lm1[triangleAsId[i][0]][0], lm1[triangleAsId[i][0]][1]],
                            [lm1[triangleAsId[i][1]][0], lm1[triangleAsId[i][1]][1]],
                            [lm1[triangleAsId[i][2]][0], lm1[triangleAsId[i][2]][1]]]])
        tr33 = np.float32([[[lm3[triangleAsId[i][0]][0], lm3[triangleAsId[i][0]][1]],
                            [lm3[triangleAsId[i][1]][0], lm3[triangleAsId[i][1]][1]],
                            [lm3[triangleAsId[i][2]][0], lm3[triangleAsId[i][2]][1]]]])
        img_morphing_from1 = img_add_no_zero(img_morphing_from1, get_trianle_image(img1, tr11, tr33))
        tr22 = np.float32([[[lm2[triangleAsId[i][0]][0], lm2[triangleAsId[i][0]][1]],
                            [lm2[triangleAsId[i][1]][0], lm2[triangleAsId[i][1]][1]],
                            [lm2[triangleAsId[i][2]][0], lm2[triangleAsId[i][2]][1]]]])
        img_morphing_from2 = img_add_no_zero(img_morphing_from2, get_trianle_image(img2, tr22, tr33))
    img_result = cv2.addWeighted(img_morphing_from1, alpha, img_morphing_from2, 1 - alpha, 0)
    return img_result


def combine_to_one_pic():
    # 1 加载一张图片
    image_origin1 = cv2.imread('./img/img_affine/1.png')
    image_origin2 = cv2.imread('./img/img_affine/2.png')
    image_origin3 = cv2.imread('./img/img_affine/3.png')
    image_origin4 = cv2.imread('./img/img_affine/4.png')
    image_origin5 = cv2.imread('./img/img_affine/5.png')
    image_origin6 = cv2.imread('./img/img_affine/6.png')
    image_origin7 = cv2.imread('./img/img_affine/7.png')
    image_origin8 = cv2.imread('./img/img_affine/8.png')
    image_origin9 = cv2.imread('./img/img_affine/9.png')

    # 将两张图片的元组叠加在一起
    merged_img = np.hstack([image_origin1, image_origin2, image_origin3,
                            image_origin4, image_origin5, image_origin6,
                            image_origin7, image_origin8, image_origin9,
                            ])
    # 展示窗口
    cv2.imshow('image', merged_img)
    # 窗口等待
    cv2.waitKey(0)
    # 销毁窗口e
    cv2.destroyAllWindows()


def convert_to_git():
    images = []
    filename1 = sorted((i for i in os.listdir('./img/img_affine/') if i.endswith('.png')), reverse=True)
    filename2 = sorted((i for i in os.listdir('./img/img_affine/') if i.endswith('.png')), reverse=False)
    filename = filename1 + filename2
    print(filename)
    for filename in filename:
        images.append(imageio.imread('./img/img_affine/' + filename))
    imageio.mimsave('./img/img_affine/combine_girl.gif', images, duration=0.2)
    print('Program conversion completed...')


if __name__ == '__main__':
    print('准备数据...')
    # 人脸识别模型路径
    predictor_path = "./model/shape_predictor_68_face_landmarks.dat"
    face1path = "./img/girl.png"
    face2path = "./img/girl2.png"

    # 获使用dlib自带的frontal_face_detector作为我们的特征提取器
    detector = dlib.get_frontal_face_detector()
    # 加载模型
    predictor = dlib.shape_predictor(predictor_path)

    # 获取图片人脸的特征点
    landmark1 = get_landmarks(face1path, detector, predictor)
    # print(landmark1)
    landmark2 = get_landmarks(face2path, detector, predictor)

    if len(landmark1) == 0 or len(landmark2) == 0:
        print("Error!")

    L1 = np.array(landmark1)
    L2 = np.array(landmark2)
    # print(L1)i
    face1 = cv2.imread(face1path)
    # print(face1)
    face2 = cv2.imread(face2path)

    print('开始计算...')

    for i in range(1, 10):
        img3 = generate_morphing_image(face1, face2, L1, L2, i / 10)

        print("\r进度:%.2f%%" % (i * (100 / 9)), end='')

        cv2.imwrite("./img/img_affine/" + str(i) + ".png", img3)
    print('图片转换完毕...')

    print('正在转换成gif图...')
    convert_to_git()

    print('将9个图片转换成一个图片...')
    combine_to_one_pic()

    print('运行结束...')
