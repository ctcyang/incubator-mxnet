import matplotlib
matplotlib.use('Agg')

import argparse, time, logging
import math
import re
import mxnet as mx
import numpy as np
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.data import imagenet
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, TrainingHistory, freeze_bn

# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--data-dir', type=str, default='~/data/gluon/')
parser.add_argument('--data-train', type=str, default='/media/ramdisk/pass-through/train-passthrough.rec')
parser.add_argument('--data-val', type=str, default='/media/ramdisk/pass-through/val-passthrough.rec')
parser.add_argument('--data-train-idx', type=str, default='/media/ramdisk/pass-through/train-passthrough.idx')
parser.add_argument('--data-val-idx', type=str, default='')
parser.add_argument('--recio', action='store_true')
parser.add_argument('--optimizer', type=str, default='nag')
parser.add_argument('--dummy', action='store_true',
                    help='use dummy data to test training speed. default is false.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                    help='number of preprocessing workers')
parser.add_argument('--num-epochs', type=int, default=3,
                    help='number of training epochs.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate. default is 0.1.')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum value for optimizer, default is 0.9.')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate. default is 0.0001.')
parser.add_argument('--lr-factor', type=float, default=0.1,
                    help='decay rate of learning rate. default is 0.1.')
parser.add_argument('--lr-step-epochs', type=str, default='30,60,80')
parser.add_argument('--mode', type=str, default='hybrid',
                    help='mode in which to train the model. options are symbolic, imperative, hybrid')
parser.add_argument('--model', type=str, required=True,
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--use_se', action='store_true',
                    help='use SE layers or not in resnext. default is false.')
parser.add_argument('--label-smoothing', action='store_true',
                    help='use label smoothing or not in training. default is false.')
parser.add_argument('--freeze-bn', action='store_true',
                    help='freeze batchnorm layers in the last frew training epochs or not. default is false.')
parser.add_argument('--batch-norm', action='store_true',
                    help='enable batch normalization or not in vgg. default is false.')
parser.add_argument('--use-pretrained', action='store_true',
                    help='enable using pretrained model from gluon.')
parser.add_argument('--log-interval', type=int, default=50,
                    help='Number of batches to wait before logging.')
parser.add_argument('--save-frequency', type=int, default=10,
                    help='frequency of model saving.')
parser.add_argument('--save-dir', type=str, default='params',
                    help='directory of saved models')
parser.add_argument('--logging-dir', type=str, default='logs',
                    help='directory of training logs')
parser.add_argument('--save-plot-dir', type=str, default='plots',
                    help='the path to save the history plot')
parser.add_argument('--dtype', type=str, default='float32')
parser.add_argument('--params-file', type=str, default='')
parser.add_argument('--contrast', type=float, default=0.4)
parser.add_argument('--saturation', type=float, default=0.4)
parser.add_argument('--brightness', type=float, default=0.4)
parser.add_argument('--pca-noise', type=float, default=0.1)
parser.add_argument('--max-random-aspect-ratio', type=float, default=1.33)
parser.add_argument('--min-random-aspect-ratio', type=float, default=0.75)
parser.add_argument('--max-random-scale', type=float, default=1)
parser.add_argument('--min-random-scale', type=float, default=0.533)
parser.add_argument('--max-random-area', type=float, default=1)
parser.add_argument('--min-random-area', type=float, default=0.08)
parser.add_argument('--max-rotate-angle',type=float, default=0)
parser.add_argument('--max-random-shear-ratio', type=float, default=0)
parser.add_argument('--data-nthreads', type=int, default=40)
parser.add_argument('--kv-store', type=str, default='device')
parser.add_argument('--warmup-epochs', type=int, default=0)
parser.add_argument('--log', type=str, default='')
opt = parser.parse_args()

logging_handlers = [logging.StreamHandler()]
if opt.log:
    makedirs(opt.logging_dir)
    logging_handlers.append(logging.FileHandler('%s/%s'%(opt.logging_dir, opt.log), mode='w'))

logging.basicConfig(level=logging.INFO, handlers=logging_handlers)
logging.info(opt)

batch_size = opt.batch_size
classes = 1000

num_gpus = len(opt.gpus.split(','))
batch_size *= max(1, num_gpus)
num_examples = 1281167
num_batch = math.ceil(num_examples/batch_size)
context = [mx.gpu(int(i)) for i in opt.gpus.split(',')]
num_workers = opt.num_workers
kv = mx.kvstore.create(opt.kv_store)

class LRScheduler(object):
    """Base class of a learning rate scheduler.

    A scheduler returns a new learning rate based on the number of updates that have
    been performed.

    Parameters
    ----------
    base_lr : float, optional
        The initial learning rate.
    """
    def __init__(self, base_lr=0.01):
        self.base_lr = base_lr

    def __call__(self, num_update):
        """Return a new learning rate.

        The ``num_update`` is the upper bound of the number of updates applied to
        every weight.

        Assume the optimizer has updated *i*-th weight by *k_i* times, namely
        ``optimizer.update(i, weight_i)`` is called by *k_i* times. Then::

            num_update = max([k_i for all i])

        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """
        raise NotImplementedError("must override this")

class MultiFactorScheduler(LRScheduler):
    """Reduce the learning rate by given a list of steps.

    Assume there exists *k* such that::

       step[k] <= num_update and num_update < step[k+1]

    Then calculate the new learning rate by::

       base_lr * pow(factor, k+1)

    Parameters
    ----------
    step: list of int
        The list of steps to schedule a change
    factor: float
        The factor to change the learning rate.
    """
    def __init__(self, step, factor=1):
        super(MultiFactorScheduler, self).__init__()
        assert isinstance(step, list) and len(step) >= 1
        for i, _step in enumerate(step):
            if i != 0 and step[i] <= step[i-1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if _step < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        self.step = step
        self.cur_step_ind = 0
        self.factor = factor
        self.count = 0

    def __call__(self, num_update):
        # NOTE: use while rather than if  (for continuing training via load_epoch)
        while self.cur_step_ind <= len(self.step)-1:
            if num_update > self.step[self.cur_step_ind]:
                self.count = self.step[self.cur_step_ind]
                self.cur_step_ind += 1
                self.base_lr *= self.factor
                logging.info("Update[%d]: Change learning rate to %0.5e",
                             num_update, self.base_lr)
            else:
                return self.base_lr
        return self.base_lr

class WarmupScheduler(LRScheduler):
    """Implement linear warmup

    base_lr * pow(1 - num_update/max_steps, poly)

    Parameters
    ----------
    lr_begin: float
                  learning rate at the first iteration of warmup
    warmup_steps: int
                  number of warmup steps
        scheduler: LRScheduler
                  scheduler following the warmup
    """
    def __init__(self, lr_begin, warmup_steps, scheduler, **kwargs):
        super(WarmupScheduler, self).__init__()
        self.lr_begin = lr_begin
        self.warmup_steps = warmup_steps
        self.scheduler = scheduler
        self.lrs_updates = {}
    def __call__(self, num_update):
        if num_update < self.warmup_steps:
            self.base_lr = self.scheduler.base_lr
            if num_update not in self.lrs_updates:
                l = self.lr_begin + (self.base_lr - self.lr_begin) * float(num_update)/float(self.warmup_steps)
                self.lrs_updates[num_update] = l
                #logging.info('lr for num_update ' + str(num_update) + ' is ' + str(self.lrs_updates[num_update]))
            else:
                l = self.lrs_updates[num_update]
        if num_update not in self.lrs_updates:
            self.lrs_updates[num_update] = self.scheduler(num_update - self.warmup_steps)
            #logging.info('lr for num_update ' + str(num_update) + ' is ' + str(self.lrs_updates[num_update]))
        return self.lrs_updates[num_update]


def get_lr_scheduler():
    epoch_size = int(int(num_examples / batch_size) / kv.num_workers)
   
    if 'pow' in opt.lr_step_epochs:
        lr = opt.lr
        max_up = opt.num_epochs * epoch_size
        pwr = float(re.sub('pow[- ]*', '', opt.lr_step_epochs))
        lr_sched = mx.lr_scheduler.PolyScheduler(max_up, lr, pwr)
    else:
        step_epochs = [int(l) for l in opt.lr_step_epochs.split(',')]
        lr = opt.lr
        begin_epoch = 0
        for s in step_epochs:
            if begin_epoch >= s:
                lr *= opt.lr_factor
        if lr != opt.lr:
            logging.info('Adjust learning rate to %e for epoch %d', lr, begin_epoch)
        steps = [epoch_size * (x - begin_epoch) for x in step_epochs if x - begin_epoch > 0]
        lr_sched = mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=opt.lr_factor)
    #lr_decay = opt.lr_decay
    #lr_decay_period = opt.lr_decay_period
    #lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')] + [np.inf]
    if opt.warmup_epochs > 0:
        lr_sched = mx.lr_scheduler.WarmupScheduler(0, epoch_size * opt.warmup_epochs, lr_sched)
    return lr_sched

model_name = opt.model

kwargs = {'ctx': context, 'pretrained': opt.use_pretrained, 'classes': classes}
if model_name.startswith('vgg'):
    kwargs['batch_norm'] = opt.batch_norm
elif model_name.startswith('resnext'):
    kwargs['use_se'] = opt.use_se

optimizer = opt.optimizer
optimizer_params = {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum, 'multi_precision':True, 'lr_scheduler': get_lr_scheduler()}

net = get_model(model_name, **kwargs)
net.initialize(mx.init.MSRAPrelu(), ctx=context)
net.cast(opt.dtype)

if opt.params_file:
    # net.load_params(opt.params_file, ctx=ctx)
    params = mx.nd.load(opt.params_file)
    for k, v in params.items():
        temp = params.pop(k)
        new_k = k.replace('arg:', '').replace('aux:', '')
        params[new_k] = temp
        '''
    for k, v in params.items():
        new_k = k.replace('arg:', '').replace('aux:', '')
        assert new_k in net_params
        net_params[new_k].set_data(v)
        '''
    mx.nd.save('temp.params', params)
    net.collect_params().load('temp.params', ctx=context)

if opt.use_se:
    model_name = 'se_' + model_name

acc_top1 = mx.metric.Accuracy()
train_history = TrainingHistory(['training-top1-err', 'validation-top1-err'])

save_frequency = opt.save_frequency
if opt.save_dir and save_frequency:
    save_dir = opt.save_dir
    makedirs(save_dir)
else:
    save_dir = ''
    save_frequency = 0

plot_path = opt.save_plot_dir

if plot_path:
    makedirs(plot_path)

def dataloader_transforms():
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    jitter_param = 0.0 if model_name.startswith('mobilenet') else 0.4
    lighting_param = 0.0 if model_name.startswith('mobilenet') else 0.1

    transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                 saturation=jitter_param),
    transforms.RandomLighting(lighting_param),
    transforms.ToTensor(),
    normalize
    ])

    transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
    ])
    return transform_train, transform_test

def smooth(label, classes, eta=0.1):
    if isinstance(label, nd.NDArray):
        label = [label]
    smoothed = []
    for l in label:
        ind = l.astype('int')
        res = nd.zeros((ind.shape[0], classes), ctx = l.context)
        res += eta/classes
        res[nd.arange(ind.shape[0], ctx = l.context), ind] = 1 - eta + eta/classes
        smoothed.append(res)
    return smoothed

def test(ctx, val_data):
    acc_top1.reset()
    for i, batch in enumerate(val_data):
        dataarg = batch.data[0] if opt.recio else batch[0]
        labelarg = batch.label[0] if opt.recio else batch[1]
        data = gluon.utils.split_and_load(dataarg, ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(labelarg, ctx_list=ctx, batch_axis=0)
        outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
        acc_top1.update(label, outputs)
    _, top1 = acc_top1.get()
    if opt.recio:
        val_data.reset()
    return 1-top1

def get_rec():
    rank, nworker = kv.rank, kv.num_workers
    train = mx.io.ImageRecordIter(
        path_imgrec         = opt.data_train, #"/media/ramdisk/pass-through/train-passthrough.rec",
        path_imgidx         = opt.data_train_idx, #"/media/ramdisk/pass-through/train-passthrough.idx",
        label_width         = 1,
        mean_r              = 123.68,
        mean_g              = 116.779,
        mean_b              = 103.939,
        data_name           = 'data',
        label_name          = 'softmax_label',
        data_shape          = (3,224,224),
        batch_size          = batch_size,
        rand_crop           = False,
        rand_mirror         = True,
        random_resized_crop = True,
        max_random_scale    = opt.max_random_scale,
        pad                 = 0,
        fill_value          = 127,
        min_random_scale    = opt.min_random_scale,
        max_aspect_ratio    = opt.max_random_aspect_ratio,
        min_aspect_ratio    = opt.min_random_aspect_ratio,
        max_random_area     = opt.max_random_area,
        min_random_area     = opt.min_random_area,
        random_h            = 0,
        random_s            = 0,
        random_l            = 0,
        brightness          = opt.brightness,
        saturation          = opt.saturation,
        contrast            = opt.contrast,
        pca_noise           = opt.pca_noise,
        max_rotate_angle    = opt.max_rotate_angle,
        max_shear_ratio     = opt.max_random_shear_ratio,
        preprocess_threads  = opt.data_nthreads,
        shuffle             = True,
        num_parts           = nworker,
        part_index          = rank)
    val = mx.io.ImageRecordIter(
        path_imgrec         = opt.data_val, # "/media/ramdisk/pass-through/val-passthrough.rec",
        path_imgidx         = opt.data_val_idx,
        label_width         = 1,
        mean_r              = 123.68,
        mean_g              = 116.779,
        mean_b              = 103.939,
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        resize              = 256,
        data_shape          = (3,224,224),
        preprocess_threads  = opt.data_nthreads,
        rand_crop           = False,
        rand_mirror         = False,
        num_parts           = nworker,
        part_index          = rank)
    return train,val

def train(epochs, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    if not opt.recio:
        transform_train, transform_test = dataloader_transforms()
        train_data = gluon.data.DataLoader(
          imagenet.classification.ImageNet(opt.data_dir, train=True).transform_first(transform_train),
          batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)
        val_data = gluon.data.DataLoader(
          imagenet.classification.ImageNet(opt.data_dir, train=False).transform_first(transform_test),
          batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        train_data, val_data = get_rec()
    
    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
    if opt.label_smoothing:
        L = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
    else:
        L = gluon.loss.SoftmaxCrossEntropyLoss()

#    lr_decay_count = 0
    best_val_score = 1
    for epoch in range(epochs):
        tic = time.time()
        acc_top1.reset()
        btic = time.time()
   
        #if lr_decay_period and epoch and epoch % lr_decay_period == 0:
        #    trainer.set_learning_rate(trainer.learning_rate*lr_decay)
        #elif lr_decay_period == 0 and epoch == lr_decay_epoch[lr_decay_count]:
        #    trainer.set_learning_rate(trainer.learning_rate*lr_decay)
        #    lr_decay_count += 1

        if opt.freeze_bn and epoch == epochs - 10:
            freeze_bn(net, True)

        for i, batch in enumerate(train_data):
            dataarg = batch.data[0] if opt.recio else batch[0]
            labelarg = batch.label[0] if opt.recio else batch[1]
            data = gluon.utils.split_and_load(dataarg, ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(labelarg, ctx_list=ctx, batch_axis=0)
            if opt.label_smoothing:
                label_smooth = smooth(label, classes)
            else:
                label_smooth = label
            with ag.record():
                outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
                loss = [L(yhat, y) for yhat, y in zip(outputs, label_smooth)]
            ag.backward(loss)
            trainer.step(batch_size)
            acc_top1.update(label, outputs)
          #  train_loss += sum([l.sum().asscalar() for l in loss])
            if opt.log_interval and not (i+1)%opt.log_interval:
                _, top1 = acc_top1.get()
                err_top1 = 1-top1
                logging.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\ttop1-err=%f\tlr=%f'%(
                             epoch, i, batch_size*opt.log_interval/(time.time()-btic), err_top1, trainer.learning_rate))
                btic = time.time()

        _, top1 = acc_top1.get()
        err_top1 = 1-top1
        #train_loss /= num_batch * batch_size

        if opt.recio:
            train_data.reset()

        err_top1_val = test(ctx, val_data)
        train_history.update([err_top1, err_top1_val])
        train_history.plot(['training-top1-err', 'validation-top1-err'], save_path='%s/%s_top1.png'%(plot_path, model_name))
        logging.info('[Epoch %d] training: err-top1=%f'%(epoch, err_top1))
        logging.info('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
        logging.info('[Epoch %d] validation: err-top1=%f'%(epoch, err_top1_val))

        if err_top1_val < best_val_score and epoch > 50:
            best_val_score = err_top1_val
            net.save_params('%s/%.4f-imagenet-%s-%d-best.params'%(save_dir, best_val_score, model_name, epoch))

        if save_frequency and save_dir and (epoch + 1) % save_frequency == 0:
            net.save_params('%s/imagenet-%s-%d.params'%(save_dir, model_name, epoch))

    if save_frequency and save_dir:
        net.save_params('%s/imagenet-%s-%d.params'%(save_dir, model_name, epochs-1))

def train_dummy(ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    net.initialize(mx.init.MSRAPrelu(), ctx=ctx)

    data = []
    label = []
    bs = batch_size // len(ctx)
    for c in ctx:
        data.append(mx.nd.random.uniform(shape=(bs,3,224,224), ctx = c))
        label.append(mx.nd.ones(shape=(bs), ctx = c))

    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
    if opt.label_smoothing:
        L = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
    else:
        L = gluon.loss.SoftmaxCrossEntropyLoss()

    acc_top1.reset()
    btic = time.time()
    num_batch = 1000
    warm_up = 100

    for i in range(num_batch):
        if i == warm_up:
            tic = time.time()
        if opt.label_smoothing:
            label_smooth = smooth(label, classes)
        else:
            label_smooth = label
        with ag.record():
            outputs = [net(X) for X in data]
            loss = [L(yhat, y) for yhat, y in zip(outputs, label_smooth)]
        ag.backward(loss)
        trainer.step(batch_size)
        acc_top1.update(label, outputs)
        if opt.log_interval and not (i+1)%opt.log_interval:
            logging.info('Batch [%d]\tSpeed: %f samples/sec'%(
                         i, batch_size*opt.log_interval/(time.time()-btic)))
            btic = time.time()
    total_time_cost = time.time()-tic
    logging.info('Test finished. Average Speed: %f samples/sec. Total time cost: %f'%(
                 batch_size*(num_batch-warm_up)/total_time_cost, total_time_cost))

def main():
    if opt.mode == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)
    if opt.dummy:
        train_dummy(context)
    else:
        train(opt.num_epochs, context)

if __name__ == '__main__':
    main()
