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

class ImageNet(object):

    def __init__(self):
        self.trainImageBasePath = "/media/codenext2/d/ILSVRC/"
        self.annotationsPath = self.trainImageBasePath + "Annotations/CLS-LOC/"
        self.dataPath = self.trainImageBasePath + "Data/CLS-LOC/"
        self.imgSetsPath = self.trainImageBasePath + "ImageSets/CLS-LOC/"
        self.final_dataset_file = "imagenet.pkl"
        self.category_num = 0

    def loader(self):

        self.arrangement()

        return

    def simpleLoader(self):
        """
        without download check
        """
        data = self.utility.unpickle(current + self.final_dataset_file)
        return data

    def arrangement(self):
        #load data for Train
        """
         * load data
        """
        #get all categories meta
        alllist = os.listdir(self.dataPath + "train/")
        print(alllist)
        metalist = [item for item in alllist if os.path.isdir(self.dataPath+item)]
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

        print("loading traindata ,,,,,,,")
        trainData = {}
        for trainimg in trainImgs:
            imgpath = self.dataPath + "train/" + trainimg + ".JPEG"
            annotationpath = self.annotationsPath + "train/" + trainimg + ".xml"
            label = self.loadXML(annotationpath)
            trainData[trainimg] = {"imgpath":imgpath,"label":label,"label_index":self.ctg_ind(label)}

        print("loading valdata ,,,,,,,")
        valData = {}
        for valimg in valImgs:
            imgpath = self.dataPath + "val/" + valimg + ".JPEG"
            annotationpath = self.annotationsPath + "val/" + valimg + ".xml"
            label = self.loadXML(annotationpath)
            trainData[valimg] = {"imgpath":imgpath,"label":label,"label_index":self.ctg_ind(label)}

        res = {}
        res["train"] = trainData
        res["val"] = valData
        res["meta"] = metalist

        #save on pkl file
        print("saving to pkl file ...")
        savepath = self.final_dataset_file
        self.utility.savepickle(res,savepath)
        print("data preparation was done ...")
        return self.category_num

    def ctg_ind(self,ctgname):
        # print("category: ",ctgname)
        # print("#####")
        # print("#####")
        meta = np.array(self.meta)
        print(meta)
        print(ctgname)
        print(np.where(meta==ctgname))
        ind = np.where(meta==ctgname)[0][0]
        return ind

    def loadXML(self,filepath):
        d = open(filepath).read()
        soup = Soup(d,"lxml")
        label = soup.find("folder").text
        return label

    def loadImageDataFromKey(self,sampled_key_lists,dataKeyList,train_or_test):
        if train_or_test == "train":
            batchsize = len(sampled_key_lists)
            targetKeys = dataKeyList[sampled_key_lists]
        elif train_or_test == "val":
            batchsize = len(dataKeyList)
            targetKeys = dataKeyList

        imgdatas = []
        for key in targetKeys:
            img = Image.open( self.dataPath + train_or_test + key + ".JPEG")
            img = np.asarray(img).transpose(2,0,1).astype(np.float32)/255.
            img = amaz_augumentation.Augumentation().Z_score(img)
            imgdatas.append(img)
        return imgdatas

    def loadImageAnnotationsFromKey(self,sampled_key_lists,dataKeyList,annotation_filepath,train_or_test):
        d = open(annotation_filepath,"rb")
        dd = pickle.load(d)
        d.close()
        annotations = []
        if train_or_test == "train":
            targetKeys = dataKeyList[sampled_key_lists]
        elif train_or_test == "val":
            targetKeys = dataKeyList

        t = []
        for key in targetKeys:
            anno = dd[train_or_test][key]
            ind = int(anno["label_index"])
            t.append(ind)
        return t
