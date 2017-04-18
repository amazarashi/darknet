import chainer
from chainer import optimizers

class Optimizers(object):

    def __init__(self,model,epoch=300):
        self.model = model
        self.epoch = epoch
        self.optimizer = None

    def __call__(self):
        pass

    def update(self):
        self.optimizer.update()

    def setup(self,model):
        self.optimizer.setup(model)

class OptimizerSqueeze(Optimizers):

    def __init__(self,model=None,lr=0.01,momentum=0.9,epoch=300,schedule=(150,225),weight_decay=1.0e-4):
        super(OptimizerSqueeze,self).__init__(model,epoch)
        self.lr = lr
        self.optimizer = optimizers.MomentumSGD(self.lr,momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        self.optimizer.setup(model)
        self.optimizer.add_hook(weight_decay)
        self.schedule = schedule

    def update_parameter(self,current_epoch):
        if current_epoch in self.schedule:
            new_lr = self.lr * 0.1
            self.optimizer.lr = new_lr
            print("optimizer was changed to {0}..".format(new_lr))
