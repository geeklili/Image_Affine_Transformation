import numpy as np
import dlib
import cv2
import sys
import math


def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr = "{0:." + str(decimals) + "f}"
    percents = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '|' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def getLandmarks(imgPath, dlibDetector, dlibPredictor):
    img = cv2.imread(imgPath)
    dets = dlibDetector(img, 1)
    if len(dets) != 1:
        print("Face num not 1!")

        return []
    shape = dlibPredictor(img, dets[0])
    keyPoints = []
    for i in range(0, shape.num_parts):
        keyPoints.append((shape.part(i).x, shape.part(i).y))
    A = img.shape
    keyPoints.append((0, 0))
    keyPoints.append((0, A[0] - 1))
    keyPoints.append((A[1] - 1, 0))
    keyPoints.append((A[1] - 1, A[0] - 1))
    keyPoints.append((0, (A[0] - 1) / 2))
    keyPoints.append(((A[1] - 1) / 2, 0))
    keyPoints.append((A[1] - 1, (A[0] - 1) / 2))
    keyPoints.append(((A[1] - 1) / 2, A[0] - 1))
    return keyPoints


def pldist(Xprime, Pprime, Qprime):
    return math.fabs(
        (Qprime[1] - Pprime[1]) * Xprime[0] - (Qprime[0] - Pprime[0]) * Xprime[1] + Qprime[0] * Pprime[1] - Qprime[1] *
        Pprime[0]) / math.sqrt((Qprime[1] - Pprime[1]) ** 2 + (Qprime[0] - Pprime[0]) ** 2)


def findX(Xprime, Pprime, Qprime, P, Q):
    u = pldist(Xprime, Pprime, Qprime)
    pxprime = [Xprime[0] - Pprime[0], Xprime[1] - Pprime[1]]
    pqprime = [Qprime[0] - Pprime[0], Qprime[1] - Pprime[1]]
    vprime = (pxprime[0] * pqprime[0] + pxprime[1] * pqprime[1]) / math.sqrt(pqprime[0] ** 2 + pqprime[1] ** 2)
    pq = [Q[0] - P[0], Q[1] - P[1]]
    ratio = vprime / math.sqrt(pqprime[0] ** 2 + pqprime[1] ** 2)
    v1 = [P[0] + pq[0] * ratio, P[1] + pq[1] * ratio]
    vu = [0, 0]
    if pxprime[0] * pqprime[1] - pxprime[1] * pqprime[0] > 0:
        vu = [pq[1], -pq[0]]
    else:
        vu = [-pq[1], pq[0]]
    lenvu = math.sqrt(vu[0] ** 2 + vu[1] ** 2)
    vu1 = [vu[0] / lenvu * u, vu[1] / lenvu * u]
    x = [int(v1[0] + vu1[0]), int(v1[1] + vu1[1])]
    return x


def generateMorphingImage(img1, img2, L1, L2, alpha):
    lineVect = [[0, 16], [8, 27], [20, 23], [31, 35], [48, 54]]
    resLine = []
    for i in lineVect:
        resLine.append(
            [L1[i[0]][0] * alpha + L2[i[0]][0] * (1 - alpha), L1[i[0]][1] * alpha + L2[i[0]][1] * (1 - alpha),
             L1[i[1]][0] * alpha + L2[i[1]][0] * (1 - alpha), L1[i[1]][1] * alpha + L2[i[1]][1] * (1 - alpha)])
    h, w, c = img1.shape
    a = 5
    b = 0.9
    p = 0
    morph1 = np.zeros_like(img1)
    for i in range(0, h):
        for j in range(0, w):
            weight = []
            vtmp = 0
            for k in range(0, len(lineVect)):
                x = findX([j, i], [resLine[k][0], resLine[k][1]], [resLine[k][2], resLine[k][3]],
                          [L1[lineVect[k][0]][0], L1[lineVect[k][0]][1]],
                          [L1[lineVect[k][1]][0], L1[lineVect[k][1]][1]])
                if (x[0] > 0) and (x[0] < w) and (x[1] > 0) and (x[1] < h):
                    length = math.sqrt((resLine[k][2] - resLine[k][0]) ** 2 + (resLine[k][3] - resLine[k][1]) ** 2)
                    dist = pldist([i, j], [resLine[k][0], resLine[k][1]], [resLine[k][2], resLine[k][3]])
                    weight.append((length ** p / (a + dist)) ** b)
                    vtmp += img1[x[1]][x[0]] * ((length ** p / (a + dist)) ** b)
            vtmp /= np.sum(weight)
            morph1[i][j] = vtmp
    morph2 = np.zeros_like(img2)
    for i in range(0, h):
        for j in range(0, w):
            weight = []
            vtmp = 0
            for k in range(0, len(lineVect)):
                x = findX([j, i], [resLine[k][0], resLine[k][1]], [resLine[k][2], resLine[k][3]],
                          [L2[lineVect[k][0]][0], L2[lineVect[k][0]][1]],
                          [L1[lineVect[k][1]][0], L1[lineVect[k][1]][1]])
                if (x[0] > 0) and (x[0] < w) and (x[1] > 0) and (x[1] < h):
                    length = math.sqrt((resLine[k][2] - resLine[k][0]) ** 2 + (resLine[k][3] - resLine[k][1]) ** 2)
                    dist = pldist([i, j], [resLine[k][0], resLine[k][1]], [resLine[k][2], resLine[k][3]])
                    weight.append((length ** p / (a + dist)) ** b)
                    vtmp += img2[x[1]][x[0]] * ((length ** p / (a + dist)) ** b)
            vtmp /= np.sum(weight)
            morph2[i][j] = vtmp
    morph = cv2.addWeighted(morph1, alpha, morph2, (1 - alpha), 0)
    return morph


if __name__ == '__main__':

    predictor_path = "./model/shape_predictor_68_face_landmarks.dat"
    face1path = "./img/obama.png.png"
    face2path = "./img/zjm.png.png"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    landmark1 = getLandmarks(face1path, detector, predictor)
    landmark2 = getLandmarks(face2path, detector, predictor)
    if len(landmark1) == 0 or len(landmark2) == 0:
        print("Error!")

    L1 = np.array(landmark1)
    L2 = np.array(landmark2)

    face1 = cv2.imread(face1path)
    face2 = cv2.imread(face2path)

    for i in range(1, 10):
        img3 = generateMorphingImage(face1, face2, L1, L2, np.float32(i) / 10)
        printProgress(i, 9, barLength=50)
        cv2.imwrite("./img/img_bn/" + str(i) + ".png", img3)

'''
lineVect = [[0,16], [8,27], [20,23], [31,35], [48,54]]
#lineVect = [[0,16], [8,27]]
p = 0.5

resLine = []

for i in lineVect:
    resLine.append([L1[i[0]][0]*0.5+L2[i[0]][0]*0.5, L1[i[0]][1]*0.5+L2[i[0]][1]*0.5, L1[i[1]][0]*0.5+L2[i[1]][0]*0.5, L1[i[1]][1]*0.5+L2[i[1]][1]*0.5])
    cv2.line(face1, (L1[i[0]][0], L1[i[0]][1]), (L1[i[1]][0], L1[i[1]][1]), (0,0,255))

h,w,c = face1.shape
a=5
b=0.9
p=0

morph = np.zeros_like(face1)

for i in range(0,h):
    for j in range(0,w):
        weight = []
        vtmp = 0
        for k in range(0,len(lineVect)):
            x = findX([j, i], [resLine[k][0], resLine[k][1]], [resLine[k][2], resLine[k][3]], [L1[lineVect[k][0]][0], L1[lineVect[k][0]][1]], [L1[lineVect[k][1]][0], L1[lineVect[k][1]][1]])
            if x[0]>0 and x[0]<h and x[1]>0 and x[1]<h:
                length = math.sqrt((resLine[k][2]-resLine[k][0])**2+(resLine[k][3]-resLine[k][1])**2)
                dist = pldist([i,j], [resLine[k][0], resLine[k][1]], [resLine[k][2], resLine[k][3]])
                weight.append((length**p/(a+dist))**b)
                vtmp += face1[x[1]][x[0]]*((length**p/(a+dist))**b)
        vtmp /= np.sum(weight)
        morph[i][j] = vtmp

for l in resLine:
    cv2.line(morph, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0,255,0))

cv2.imshow('morph.png',morph)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
