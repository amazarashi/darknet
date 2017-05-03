import pickle
import glob
import os
import numpy as np
current = dir_path = os.path.dirname(os.path.realpath('__file__')) + "/"

import amaz_util as amaz_Util
import amaz_augumentation

from os import system
from PIL import Image
from bs4 import BeautifulSoup as Soup
import cv2

from multiprocessing import Pool


class ImageNetInspector(object):

    def __init__(self):
        self.trainImageBasePath = "/home/codenext2/Downloads/ILSVRC/"
        self.annotationsPath = self.trainImageBasePath + "Annotations/CLS-LOC/"
        self.dataPath = self.trainImageBasePath + "Data/CLS-LOC/"
        self.imgSetsPath = self.trainImageBasePath + "ImageSets/CLS-LOC/"
        self.final_dataset_file = "imagenet.pkl"
        self.utility = amaz_Util.Utility()
        self.category_num = 0
        self.meta = []

    def arrangement(self):
        #load data for Train
        """
         * load data
        """
        #get all categories meta
        alllist = os.listdir(self.dataPath + "train/")
        metalist = alllist
        self.meta = metalist
        category_num = len(metalist)
        print("category_num: ",category_num)
        self.category_num = category_num

        #get annotation info
        trainImageSetPath = self.imgSetsPath + "train_loc.txt"
        valImageSetPath = self.imgSetsPath + "val.txt"

        trainImgs = open(trainImageSetPath,"r")
        trainImgs = trainImgs.readlines()
        trainImgs = [info.split()[0] for info in trainImgs]
        valImgs = open(valImageSetPath,"r")
        valImgs = valImgs.readlines()
        valImgs = [info.split()[0] for info in valImgs]
        print("trainLength:",len(trainImgs))
        print("valLength:",len(valImgs))
