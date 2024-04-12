# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 20:10:15 2024

@author: lilulu
"""
'''ref1: https://blog.csdn.net/qq_41701723/article/details/135449035
   ref1: https://blog.csdn.net/qq_42131051/article/details/131430740
    
    
    '''
import copy
import cv2
import os
import shutil
import numpy as np

# mask images path
#path = "C:/work/datasets/yolo_graspnet_txt/img_mask_png_label/"
path = r"C:\work\datasets\graspnet\test_novel"  # for graspnet

camera = r'realsense'

sceneIds = list(range(160, 190 ))
sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in sceneIds]
print(sceneIds)

depthpath = []

for x in sceneIds:
    depthpath = os.path.join(path, x, camera, 'label')
    txt_labels_path = os.path.join(path, x, camera, "txt_labels")

    if not os.path.exists(txt_labels_path):
        # 如果目录不存在，则创建目录
        os.makedirs(txt_labels_path)
        print("目录 'txt_labels' 创建成功")
    else:
        print(txt_labels_path, "目录 'txt_labels' 已存在")    

    print('Current path', depthpath)
    files = os.listdir(depthpath)
    for file in files:
        name = file.split('.')[0]
        file_path = os.path.join(depthpath, name+'.png')
        img = cv2.imread(file_path)
        # img = cv2.imread(path)
        H,W=img.shape[0:2]
        # print(H,W) # print width and height of img
    
        #img1 = cv2.imread("F:/Deep_Learning/Model/YOLOv8_Seg/Dataset/images/20160222_080933_361_1.jpg")
    
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    
        # 使用 np.unique 函数获取图像数组中的唯一值和它们的计数
        unique_values, counts = np.unique(gray_img, return_counts=True) #像素-1=标签：含背景0从小到大排列
        num_objs = len(unique_values) #算背景0 的目标种类个数 从小到达排列
        # print(num_objs)
        # print(unique_values)
        
        #循环一张图标签数量，创建对应标签的mask
        #f = open("C:/work/datasets/yolo_graspnet_txt/txt_labels/{}.txt".format(file.split(".")[0]), "a+")
        txt_file_path = os.path.join(txt_labels_path, "{}.txt".format(file.split(".")[0]))
        try:
            f = open(txt_file_path, "w+")
            print("File '{}' ...".format(txt_file_path))
        except Exception as e:
            print("创建文件时出现错误:", e)
        masks_arr = []
        for label_num in range(1, num_objs):
            # 创建一个与图像数组相同大小的零数组作为 mask 数组
            mask_array = np.zeros_like(gray_img)
            #获取像素标签 gt标签=像素标签-1
            label_pixel_value = unique_values[label_num]
            # print('lll:', label_pixel_value, str(label_pixel_value-1))
            
            # 将图像数组中值为 87 的位置在 mask 数组中设置为 87
            mask_array[gray_img == label_pixel_value] = label_pixel_value
            # masks_arr[label_num-1] = mask_array
    
            # 将大于 71 和小于 71 的像素值设置为 0，仅保留值为 71 的像素
            # ret, bin_img = cv2.threshold(gray_img, 71, 255, cv2.THRESH_BINARY) # ret 变量将返回确定的阈值，bin_img 将存储二值化后的图像。
            # print('ret', ret)
            # bin_img = np.zeros_like(np.array(img))
        
            # bin_img[gray_img != 71] = 255
            #  # 显示二值图像
            # cv2.imshow('Binary Image', bin_img)
            # cv2.waitKey(0)  # 等待用户按下任意键
            # cv2.destroyAllWindows()  # 关闭窗口
        
        
            # cnt,hit = cv2.findContours(mask_array,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS) #检索轮廓树，即检索所有的轮廓并重构轮廓之间的嵌套关系。指定了轮廓的近似方法Teh-Chinl-89 近似算法
            # cnts, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#仅返回最外层的轮廓，以及 cv2.CHAIN_APPROX_SIMPLE 作为轮廓近似方法，它将轮廓的所有点返回而不做近似
            cnt, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            #ret,bin_img = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
           
            
           
        
            #cv2.drawContours(img1,cnt,-1,(0,255,0),5)
        
            cnt = list(cnt)
            
            for j in cnt:
                result = []
                pre = j[0]
                for i in j:
                    if abs(i[0][0] - pre[0][0]) > 20 or abs(i[0][1] - pre[0][1]) > 20: # 在这里可以调整间隔点，我设置为1
                        pre = i
                        temp = list(i[0])
                        temp[0] /= W
                        temp[1] /= H
                        result.append(temp)
        
                        #cv2.circle(img1,i[0],1,(0,0,255),2)
        
                #print(result)
                #print(len(result))
        
                # if len(result) != 0:
        
                if len(result) != 0:
                    f.write(str(label_pixel_value-1) + " ") #写入gt标签
                    for line in result:
                        line = str(line)[1:-2].replace(",","")
                        # print(line)
                        f.write(line+" ")
                    f.write("\n")
        f.close()



