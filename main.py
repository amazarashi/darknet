import argparse
from chainer import optimizers
import squeeze
import amaz_trainer
import amaz_cifar10_dl
import amaz_augumentationCustom
import amaz_optimizer

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='cifar10')
    parser.add_argument('--epoch', '-e', type=int,
                        default=300,
                        help='maximum epoch')
    parser.add_argument('--batch', '-b', type=int,
                        default=64,
                        help='mini batch number')
    parser.add_argument('--gpu', '-g', type=int,
                        default=-1,
                        help='-1 means cpu, put gpu id here')
    parser.add_argument('--lr', '-lr', type=float,
                        default=0.1,
                        help='learning rate')

    args = parser.parse_args().__dict__
    lr = args.pop('lr')
    epoch = args.pop('epoch')

    model = squeeze.Squeeze(10)
    optimizer = amaz_optimizer.OptimizerSqueeze(model,lr=lr,epoch=epoch)
    dataset = amaz_cifar10_dl.Cifar10().loader()
    dataaugumentation = amaz_augumentationCustom.Normalize128
    args['model'] = model
    args['optimizer'] = optimizer
    args['dataset'] = dataset
    args['dataaugumentation'] = dataaugumentation
    main = amaz_trainer.Trainer(**args)
    main.run()
