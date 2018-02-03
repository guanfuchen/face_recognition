#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import collections
import random

import torch
import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class vggfaceInfo:
    def __init__(self, id_name, uid, image_url, left_boundingbox, top_boundingbox, right_boundingbox, bottom_boundingbox, score, curation):
        self.id_name = id_name
        self.uid = uid
        self.image_url = image_url
        self.left_boundingbox = left_boundingbox
        self.top_boundingbox = top_boundingbox
        self.right_boundingbox = right_boundingbox
        self.bottom_boundingbox = bottom_boundingbox
        self.score = score
        self.curation = curation
        pass

class vggfaceLoader(Dataset):
    def __init__(self, root, split="train"):
        file_list = os.listdir(root)
        self.m_vggfaceInfos = []
        for filename in file_list:
            file = open(os.path.join(root, filename))
            for line in file.readlines():
                line_split = line.split(' ')
                uid = line_split[0]
                image_url = line_split[1]
                left_boundingbox = line_split[2]
                top_boundingbox = line_split[3]
                right_boundingbox = line_split[4]
                bottom_boundingbox = line_split[5]
                score = line_split[6]
                curation = line_split[7]
                # print('uid:', uid)
                # print('image_url:', image_url)
                # print('left_boundingbox:', left_boundingbox)
                # print('top_boundingbox:', top_boundingbox)
                # print('right_boundingbox:', right_boundingbox)
                # print('bottom_boundingbox:', bottom_boundingbox)
                # print('score:', score)
                # print('curation:', curation)
                id_name = filename[:filename.rfind('.')]
                l_vggfaceInfo = vggfaceInfo(id_name=id_name, uid=uid, image_url=image_url, left_boundingbox=left_boundingbox, top_boundingbox=top_boundingbox, right_boundingbox=right_boundingbox, bottom_boundingbox=bottom_boundingbox, score=score, curation=curation)
                self.m_vggfaceInfos.append(l_vggfaceInfo)

    def __getitem__(self, index):
        l_vggfaceInfo = self.m_vggfaceInfos[index]
        return l_vggfaceInfo.id_name, l_vggfaceInfo.uid, l_vggfaceInfo.image_url, l_vggfaceInfo.left_boundingbox, l_vggfaceInfo.top_boundingbox, l_vggfaceInfo.right_boundingbox, l_vggfaceInfo.bottom_boundingbox, l_vggfaceInfo.score, l_vggfaceInfo.curation

    def __len__(self):
        return len(self.m_vggfaceInfos)

if __name__ == '__main__':
    local_path = os.path.expanduser('~/Data/face/vgg_face_dataset/files')
    local_image_path = os.path.expanduser('~/Data/face/vgg_face_dataset/images')
    batch_size = 1
    dst = vggfaceLoader(local_path)
    trainloader = DataLoader(dst, batch_size=batch_size)
    for i, (id_names, uids, image_urls, left_boundingboxs, top_boundingboxs, right_boundingboxs, bottom_boundingboxs, scores, curations) in enumerate(trainloader):
        print(i)
        if i == 100000:
            exit()
        # print(uids)
        id_names_0 = id_names[0]
        uids_0 = uids[0]
        image_urls_0 = image_urls[0]
        left_boundingboxs_0 = left_boundingboxs[0]
        top_boundingboxs_0 = top_boundingboxs[0]
        right_boundingboxs_0 = right_boundingboxs[0]
        bottom_boundingboxs_0 = bottom_boundingboxs[0]
        # print(image_urls_0)
        unique_image_urls_path_0 = os.path.join(local_image_path, id_names_0, uids_0+'.jpg')

        # x_0 = float(left_boundingboxs_0)
        # y_0 = float(top_boundingboxs_0)
        # width_0 = abs(float(left_boundingboxs_0)-float(right_boundingboxs_0))
        # height_0 = abs(float(top_boundingboxs_0)-float(bottom_boundingboxs_0))
        # if os.path.exists(unique_image_urls_path_0):
        #     imgs_0 = misc.imread(unique_image_urls_path_0)
        #     ax = plt.subplot(111)
        #     ax.imshow(imgs_0)
        #     ax.add_patch(patches.Rectangle((x_0, y_0), width_0, height_0, fill=False))
        #     plt.show()
        print(unique_image_urls_path_0)
        image_path = os.path.join(local_image_path, id_names_0)
        print(image_path)
        if not os.path.exists(image_path):
            os.mkdir(image_path)
        # if not os.path.exists(unique_image_urls_path_0):
        os.system("proxychains4 wget -c {} -O {} -t 1".format(image_urls_0, unique_image_urls_path_0))
        # exit()
