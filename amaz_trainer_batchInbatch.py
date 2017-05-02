#coding : utf-8
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import time
import six
import pickle
from tqdm import tqdm
import amaz_sampling
import amaz_util
import amaz_sampling
import amaz_datashaping
import amaz_log
import amaz_augumentationCustom
import amaz_imagenet
import sys

sampling = amaz_sampling.Sampling()

xp = cuda.cupy

class Trainer(object):

    def __init__(self,model=None,batchinbatch=16,loadmodel=None,optimizer=None,dataset=None,epoch=300,batch=128,gpu=-1,dataaugumentation=amaz_augumentationCustom.Normalize32):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.epoch = epoch
        self.batch = batch
        self.train_key,self.train_len,self.test_key,self.test_len,self.meta = self.init_dataset()
        self.gpu = gpu
        self.check_gpu_status = self.check_gpu(self.gpu)
        self.xp = self.check_cupy(self.gpu)
        self.utility = amaz_util.Utility()
        self.datashaping = amaz_datashaping.DataShaping(self.xp)
        self.logger = amaz_log.Log()
        self.dataaugumentation = dataaugumentation
        self.batchinbatch = batchinbatch
        self.loadmodel = loadmodel
        self.init_model()

    def check_cupy(self,gpu):
        if gpu == -1:
            return np
        else:
            #cuda.get_device(gpu).use()
            self.model.to_gpu()
            return cuda.cupy

    def check_gpu(self, gpu):
        if gpu >= 0:
            cuda.get_device(gpu).use()
            self.model.to_gpu()
            return True
        return False

    def init_model(self):
        if self.loadmodel is None:
            print('no model to load')
        else:
            print('loading ' + self.load_model)
            serializers.load_npz(load_model, model)
        self.check_gpu(self.gpu)

    def init_dataset(self):
        d = open("imagenet.pkl","rb")
        dd = pickle.load(d)
        d.close()
        train_len = len(dd["train_key"])
        test_len = len(dd["val_key"])
        train_key = np.array(sorted(dd["train_key"]))
        test_key = np.array(sorted(dd["val_key"]))
        meta = np.array(dd["meta"])
        return (train_key,train_len,test_key,test_len,meta)

    def train_one(self,epoch):
        model = self.model
        optimizer = self.optimizer
        batch = self.batch
        meta = self.meta
        sum_loss = 0
        total_data_length = self.train_len

        progress = self.utility.create_progressbar(int(total_data_length/batch),desc='train',stride=1)
        train_data_yeilder = sampling.random_sampling(int(total_data_length/batch),batch,total_data_length)
        #epoch,batch_size,data_length
        batch_in_batch_size = self.batchinbatch
        for ii,indices in zip(progress,train_data_yeilder):
            model.cleargrads()
            train_x = amaz_imagenet.ImageNet().loadImageDataFromKey(indices,self.train_key,"train")
            train_y = amaz_imagenet.ImageNet().loadImageAnnotationsFromKey(indices,self.train_key,self.meta,"imagenet.pkl","train")

            #for ii in six.moves.range(0, len(indices), batch_in_batch_size):
            # print("#########")
            # print(ii)
            # print("#########")
            # print("11111")
            # print("11111")
            # print("11111")
            # x = train_x[ii:ii*batch_in_batch_size + batch_in_batch_size]
            # t = train_y[ii:ii*batch_in_batch_size + batch_in_batch_size]
            x = train_x
            t = train_y
            # print("222222")
            # print("222222")
            # print("222222")
            DaX = [self.dataaugumentation.train(img) for img in x]
            # print("333333")
            # print("333333")
            # print("333333")

            print("koko1")
            print("koko1")
            print("koko1")
            print(str(sys.getsizeof(DaX)))
            x = self.datashaping.prepareinput(DaX,dtype=self.xp.float32,volatile=False)
            print("koko2")
            print("koko2")
            print("koko2")
            print(str(sys.getsizeof(x)))
            t = self.datashaping.prepareinput(t,dtype=self.xp.int32,volatile=False)
            print("koko3")
            print("koko3")
            print("koko3")
            print(str(sys.getsizeof(t)))
            print(str(sys.getsizeof(model)))
            print("#######")
            print("#######")
            print("#######")
            # print("4444444")
            # print("4444444")
            # print("4444444")
            y = model(x,train=True)
            # print("555555")
            # print("555555")
            # print("555555")
            loss = model.calc_loss(y,t) / batch
            # print("555555")
            # print("555555")
            # print("555555")
            loss.backward()
            print("666666")
            print("666666")
            print("666666")
            loss.to_cpu()
            tqdm.write("777777")
            sum_loss += loss.data * batch
            tqdm.write("8888888")
            del loss,x,t
            print("before update")
            print("--------------")
            print("-------------")
            optimizer.update()
            print("after update")
            print("--------------")
            print("-------------")

        ## LOGGING ME
        print("train mean loss : ",float(sum_loss) / total_data_length)
        self.logger.train_loss(epoch,sum_loss/len(train_y))
        print("######################")

    def test_one(self,epoch):
        model = self.model
        optimizer = self.optimizer
        batch = self.batch
        meta = self.meta

        sum_loss = 0
        sum_accuracy = 0
        batch_in_batch_size = self.batchinbatch

        test_x = amaz_imagenet.ImageNet().loadImageDataFromKey(np.arange(self.test_len),self.test_key,"val")
        test_y = amaz_imagenet.ImageNet().loadImageAnnotationsFromKey(np.arange(self.test_len),self.test_key,self.meta,"imagenet.pkl","val")

        progress = self.utility.create_progressbar(int(len(test_x)),desc='test',stride=batch_in_batch_size)
        for i in progress:
            x = test_x[i:i + batch_in_batch_size]
            t = test_y[i:i + batch_in_batch_size]

            DaX = []
            for img in x:
                da_x = self.dataaugumentation.test(img)
                DaX.append(da_x)

            x = self.datashaping.prepareinput(DaX,dtype=np.float32,volatile=True)
            t = self.datashaping.prepareinput(t,dtype=np.int32,volatile=True)

            y = model(x,train=False)
            loss = model.calc_loss(y,t)
            sum_loss += batch_in_batch_size * loss.data
            sum_accuracy += F.accuracy(y,t).data * batch_in_batch_size
            categorical_accuracy = model.accuracy_of_each_category(y,t)
            del loss,x,t

        ## LOGGING ME
        print("test mean loss : ",sum_loss/len(test_y))
        self.logger.test_loss(epoch,sum_loss/len(test_y))
        print("test mean accuracy : ", sum_accuracy/len(test_y))
        self.logger.accuracy(epoch,sum_accuracy/len(test_y))
        print("######################")

    def run(self):
        epoch = self.epoch
        model = self.model
        progressor = self.utility.create_progressbar(epoch,desc='epoch',stride=1,start=0)
        for i in progressor:
            self.train_one(i)
            self.optimizer.update_parameter(i)
            self.test_one(i)
            #DUMP Model pkl
            model.to_cpu()
            self.logger.save_model(model=model,epoch=i)
            if self.check_gpu_status:
                model.to_gpu()

        self.logger.finish_log()
