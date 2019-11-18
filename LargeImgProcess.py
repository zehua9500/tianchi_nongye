import cv2
import numpy as np
import os
#from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image
import imageio
from osgeo import gdal
import random

def splitLargeImage(ImgPath, LabelPath, ImgDst, LabelDst, ImgNum):
    """
    :param ImgPath:  原图地址
    :param LabelPath:  原始label图
    :param ImgDst:  遥感图分割后存储文件夹
    :param LabelDst:  label分割后存储文件夹
    :param ImgNum:  1 或 2  两张遥感图号码
    :return:
    """
    Image.MAX_IMAGE_PIXELS = None
    ds = gdal.Open(ImgPath)
    step = 128
    outsize = 256
    if(LabelPath != None):
        dslb = gdal.Open(LabelPath)
        step = 1536
        outsize = 2048
    wx = ds.RasterXSize
    wy = ds.RasterYSize
    cx = 0
    cy = 0

    while cx + outsize < wx:
        cy = 0
        print('Now...cx = ', cx)
        while cy + outsize < wy:
            img = ds.ReadAsArray(cx, cy, outsize, outsize)  #loc1为横坐标 loc2为纵坐标
            img2 = img[:3, :, :].transpose(1, 2, 0)
            if (img2[:, :, 0] == 0).sum() > outsize * outsize * 0.6:
                cy += step
                continue

            img2 = Image.fromarray(img2, 'RGB')
            img2.save(os.path.join(ImgDst, '%d_%d_%d.bmp'% (ImgNum,cx, cy)))

            if (LabelPath != None):
                img = dslb.ReadAsArray(cx, cy, outsize, outsize)
                img = Image.fromarray(img).convert('L')
                img.save(os.path.join(LabelDst, '%d_%d_%d.bmp'% (ImgNum,cx, cy)))
            cy += step
        cx += step

def mergeLargeImg(srcPathFold, dstPathFold, imageNum=3):
    """
    :param imageNum: 3 或 4  两张测试图标号
    :param srcPathFold:
    :param dstPathFold:
    :return:
    """
    Image.MAX_IMAGE_PIXELS = None
    res = Image.open(r"E:\jingwei\jingwei_round1_submit_20190619\image_%d_predict.png" % imageNum)
    res = np.array(res)
    #path = r"E:\jingwei\pred%d\work\nongye\new_test\pred%d" % (srcPathFold, srcPathFold)
    #path = r"E:\jingwei\res1\work\result\res1"
    for imgname in os.listdir(srcPathFold):
        if (imgname[5] != str(imageNum)):
            continue
        # print("%s"%pred)
        pred = cv2.imread(os.path.join(srcPathFold, imgname))
        #pred = pred[128:896, 128:896, 0]
        pred = pred[64:448, 64:448, 0]
        if pred is None:
            print("%s is None" % imgname)
        index1 = imgname.find('_')
        index2 = imgname.find('_', index1 + 1)
        index3 = imgname.find('.')
        x = int(imgname[index1 + 1:index2])
        y = int(imgname[index2 + 1:index3])

        #res[y + 128:y + 896, x + 128:x + 896] = pred
        res[y + 64:y + 448, x + 64:x + 448] = pred

    if not os.path.exists(dstPathFold):
        os.makedirs(dstPathFold)
    imageio.imwrite(os.path.join(dstPathFold, r"image_old%d_predict.png" % imageNum), res)
    del res
    postProcess(os.path.join(dstPathFold, r"image_old%d_predict.png" % imageNum), os.path.join(dstPathFold, r"image_%d_predict.png" % imageNum))

    #imageio.imwrite(r"E:\jingwei\pred%d\ans\image_%d_predict.png" % (dstPathFold, imageNum), res)

def postProcess(labelpath, savepath):
    Image.MAX_IMAGE_PIXELS = None
    label = Image.open(labelpath)
    label = np.array(label)
    h, w = label.shape
    cx = 0
    cy = 0
    step = 30
    size = 60
    start = 15 #(size - step)/2
    end = 45 #size - start
    while cy + size < h:
        cx = 0
        count = np.array([0,0,0])
        while(cx + size < w):
            temp = label[cy : cy + size, cx : cx + size]
            if(np.sum(temp == 0) > size*size*0.8):
                label[cy + start: cy + end, cx + start: cx + end] = 0
                cx += step
                continue
            mask = (label[cy + start: cy + end, cx + start: cx + end] > 0)
            for i in range(3):
                count[i] = np.sum(temp == (i+1))
            finalLabel = np.argmax(count) + 1
            label[cy + start: cy + end, cx + start: cx + end]  = finalLabel * mask
            cx += step
        cy += step
    imageio.imwrite(savepath, label)

def displayCover(testFold = 3, labelPath = r"E:\jingwei\res\res1"):
    path = r"E:\jingwei\new_train\data%d"%testFold
    labelList = os.listdir(labelPath)
    random.shuffle(labelList)
    for imgname in labelList[:40]:
        if (imgname[0] != str(testFold)):
            continue
        # print("%s"%pred)
        label = cv2.imread(os.path.join(labelPath, imgname))[:, :, 0]
        # for k in range(4):
        #   print("label = %d have %s"%(k,np.sum(label == k)))
        index1 = imgname.find('_')
        index2 = imgname.find('_', index1 + 1)
        index3 = imgname.find('.')
        x = int(imgname[index1 + 1:index2])
        y = int(imgname[index2 + 1:index3])
        testImg = cv2.imread(os.path.join(path, "%s_%s_%s.bmp" % (testFold, x, y)))

        cv2.imshow("src", testImg[:768, :768])

        mask = (label == 3)
        temp = mask * 255 + testImg[:, :, 0]
        testImg[:, :, 0] = np.where(temp > 255, 255, temp)

        mask = (label == 1)
        temp = mask * 255 + testImg[:, :, 1]
        testImg[:, :, 1] = np.where(temp > 255, 255, temp)

        mask = (label == 2)
        temp = mask * 255 + testImg[:, :, 2]
        testImg[:, :, 2] = np.where(temp > 255, 255, temp)

        cv2.imshow("img", testImg[:768, :768])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':

    epoch = 5 #res文件夹序号
    srcPath = r"E:\jingwei\epoch=15\work\nongye\7_19\epoch=15"#%(epoch, epoch)
    dstPath = r"E:\jingwei\epoch=15\work\nongye\7_19\ans=15"#%(epoch, epoch)
    mergeLargeImg(imageNum=3, srcPathFold=srcPath, dstPathFold=dstPath)
    mergeLargeImg(imageNum=4, srcPathFold=srcPath, dstPathFold=dstPath)
    """
    
    ImgNum = 2
    ImgPath = r"E:\jingwei\jingwei_round1_train_20190619\image_%d.png"%ImgNum
    
    LabelPath = r"E:\jingwei\jingwei_round1_train_20190619\image_%d_label.png"%ImgNum
    ImgDst = r'E:/jingwei/new_train/data%d'%ImgNum
    LabelDst = r'E:/jingwei/new_train//label%d'%ImgNum
    
    postProcess(r"E:\jingwei\epoch=10\work\nongye\new_test\ans=10\image_old3_predict.png",
                r"E:\jingwei\epoch=10\work\nongye\new_test\ans=10\image_3_predict.png")
    postProcess(r"E:\jingwei\epoch=10\work\nongye\new_test\ans=10\image_old4_predict.png",
                r"E:\jingwei\epoch=10\work\nongye\new_test\ans=10\image_4_predict.png")
    """
    #mergeLargeImg(imageNum=3, srcPathFold=5, dstPathFold=5)
    #mergeLargeImg(imageNum=4, srcPathFold=5, dstPathFold=5)
    #displayCover(3, labelPath = r"E:\jingwei\res\res_epoch=3\result\res2_epoch=3")
    #splitLargeImage(ImgPath, LabelPath, ImgDst, LabelDst, ImgNum)
    #postProcess(labelpath=r"E:\jingwei\res\ans1\image_3_predict.png",
    #            savepath=r"E:\jingwei\res\ans1\change\image_3_predict.png")
    #postProcess(labelpath = r"E:\jingwei\res\ans1\image_4_predict.png",
    #           savepath = r"E:\jingwei\res\ans1\change\image_4_predict.png")

    #displayCover(testFold=2, labelPath=r"E:\jingwei\new_train\label2")
    """
    ImgNum = 3
    ImgPath = r"E:\jingwei\jingwei_round1_test_a_20190619\image_%d.png"%ImgNum
    LabelPath = None
    ImgDst = r'E:\jingwei\256test\test%d'%ImgNum
    LabelDst = None
    splitLargeImage(ImgPath, LabelPath, ImgDst, LabelDst, ImgNum)

    ImgNum = 4
    ImgPath = r"E:\jingwei\jingwei_round1_test_a_20190619\image_%d.png"%ImgNum
    LabelPath = None
    ImgDst = r'E:\jingwei\256test\test%d'%ImgNum
    LabelDst = None
    splitLargeImage(ImgPath, LabelPath, ImgDst, LabelDst, ImgNum)
"""
