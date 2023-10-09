"""
本部分内容学习mmengine中的runner目录
runner目录下的方法列表如下：
    'BaseLoop', 'load_state_dict', 'get_torchvision_models',
    'get_external_models', 'get_mmcls_models', 'get_deprecated_model_names',
    'CheckpointLoader', 'load_checkpoint', 'weights_to_cpu', 'get_state_dict',
    'save_checkpoint', 'EpochBasedTrainLoop', 'IterBasedTrainLoop', 'ValLoop',
    'TestLoop', 'Runner', 'get_priority', 'Priority', 'find_latest_checkpoint',
    'autocast', 'LogProcessor', 'set_random_seed', 'FlexibleRunner'
"""

from mmengine.runner import *


def BaseLoop_learn():
    """
    BaseLoop为所有Loop的基类，它的所有子类都必须对其run方法进行重写。
    """

    # 对其方法进行重写
    class MyLoop(BaseLoop):
        def run(self):
            print('loop start running')

    myloop = MyLoop(None, None)
    myloop.run()

def load_state_dict_learn():
    """
    load_state_dict函数用于将模型参数绑定到模型中，此方法由torch.nn.Module.load_state_dict改写而来。
    参数包括module, state_dict, strict=False, logger=None。
    其中strict参数用于是否强制限制模型的state dict和读取的权重一致。
    """
    import torch
    from mmdet import models
    from mmdet.registry import MODELS

    resnet_cfg = dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=None)

    weights = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
    model = MODELS.build(resnet_cfg)
    # 非严格模式读取，不会报错，但会提示哪些权重缺失
    load_state_dict(model, weights, strict=False)
    # 严格模式读取，报错
    load_state_dict(model, weights, strict=True)

def get_torchvision_models_learn():
    """
    get_torchvision_models用于从当前安装的torchvision中获取其包含的所有模型的url链接。
    """
    models_urls = get_torchvision_models()
    for k, v in models_urls.items():
        print(k, v)
    # 下载对应权重
    import torch
    weights = torch.hub.load_state_dict_from_url(models_urls['alexnet'])
    print(weights)
    # 打印权重下载目录
    print(torch.hub.get_dir())

def get_external_models_learn():
    """
    get_external_models用于从读取openmmlab官网提供的一系列模型权重的对应url链接。（主要为caffe模型）
    对应模型的urls存放地址为hub目录下的openmmlab.json文件中
    """
    openmmlab_models_urls = get_external_models()
    print(openmmlab_models_urls)
    for k, v in openmmlab_models_urls.items():
        print(k, v)

def get_mmcls_models_learn():
    """
    get_mmcls_models用于读取mmclassification中的一系列分类模型的权重urls。
    """
    mmcls_models_urls = get_mmcls_models()
    for k, v in mmcls_models_urls.items():
        print(k, v)

def get_deprecated_model_names_learn():
    """
    get_deprecated_model_names用于读取被移除的模型的name
    """
    deprecated_models_urls = get_deprecated_model_names()
    for k, v in deprecated_models_urls.items():
        print(k, v)

def CheckpointLoader_learn():
    """
    CheckpointLoader用来管理所有的读取权重方案，包括从本地读取和从url中读取权重。
    """
    checkpoint_loader = CheckpointLoader()
    # 读取在线url
    weights = checkpoint_loader.load_checkpoint('https://download.pytorch.org/models/resnet50-19c8e357.pth')
    print(weights)
    # 读取本地url
    weights2 = checkpoint_loader.load_checkpoint(r'C:\Users\Lizhiqing\.cache\torch\hub\checkpoints\alexnet-owt-4df8aa71.pth')
    print(weights2)
    # 通过特定格式来读取对应的模型权重文件
    weights3 = checkpoint_loader.load_checkpoint(r'torchvision://resnet50')
    print(weights3)

def load_checkpoint_learn():
    """
    load_checkpoint是对CheckpointLoader的一次二次封装，允许直接通过此函数来读取url和本地路径的权重文件。随后将权重加载到模型中。
    除此以外，还支持通过特定格式来读取对应的模型权重文件。
    """
    from mmdet import models
    from mmdet.registry import MODELS

    resnet_cfg = dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=None)

    model = MODELS.build(resnet_cfg)
    load_checkpoint(model, 'torchvision://resnet50', strict=False)

def weights_to_cpu_learn():
    """
    weights_to_cpu:复制一份模型的权重，并将其存放到cpu中
    """
    from mmdet import models
    from mmdet.registry import MODELS

    model_cfg = dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'))

    model = MODELS.build(model_cfg).cuda()
    # 查看模型在cpu还是gpu中，通过其第一个参数来判断
    device = next(model.parameters()).device
    print(device)
    # 读取模型权重
    model_weights = model.state_dict()
    # 打印权重服务
    print(model_weights['conv1.weight'].device)
    # 打印转换后的权重服务
    converted_weights = weights_to_cpu(model_weights)
    print(converted_weights['conv1.weight'].device)

def get_state_dict_learn():
    """
    获取对应模型的权重，和torch.nn.Module.state_dict()差不多。不过增加了部分功能，可指定输出特定层的权重(感觉意义不大）。
    注意checkpoint和weights的概念有所区别，前者表示模型训练过程中的参数，后者表示矩阵参数。
    不如说checkpoint时weights的在模型训练过程中特定时期的参数表示。
    """
    from mmdet import models
    from mmdet.registry import MODELS

    model_cfg = dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'))

    model = MODELS.build(model_cfg)
    model_weights = get_state_dict(model)
    print(model_weights)

def save_checkpoint_learn():
    """
    save_checkpoint将模型权重参数保存到特定文件中。
    """
    from mmdet import models
    from mmdet.registry import MODELS

    model_cfg = dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'))

    model = MODELS.build(model_cfg)
    model_weights = get_state_dict(model)
    save_checkpoint(model_weights, './test_save.pkl')

def EpochBasedTrainLoop_learn():
    """
    EpochBasedTrainLoop用于以轮为单位的模型训练流程。
    实际上，runner的train函数核心就是通过train_cfg参数构建的train_loop来进行模型训练的。
    runner本身被作为参数传递给了loop，以便其在运行过程中访问runner中的optimzer等属性
    train_loop将整个模型训练过程解耦合出来，在其run之前为call hook before run，在其之后为call hook after run
    所有hook均实际在loop中得到调用。
    """

    from mmdet import models
    from mmdet.registry import MODELS
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import torchvision
    from mmengine.evaluator import BaseMetric
    import torch.nn.functional as F
    from mmengine.model import BaseModel

    # 评价指标的运行过程中，先进行process处理单个结果，随后进行compute_metrics计算输出指标。
    # 此处的Accuracy时简单的分类精度表示，表示recall。
    class Accuracy(BaseMetric):
        def process(self, data_batch, data_samples):
            score, gt = data_samples
            # 将一个批次的中间结果保存至 `self.results`列表中
            self.results.append({
                'batch_size': len(gt),
                'correct': (score.argmax(dim=1) == gt).sum().cpu(),
            })

        def compute_metrics(self, results):
            total_correct = sum(item['correct'] for item in results)
            total_size = sum(item['batch_size'] for item in results)
            # 返回保存有评测指标结果的字典，其中键为指标名称
            return dict(accuracy=100 * total_correct / total_size, testoutinfo='test')

    # 要使得自己构建的模型能够在runner中跑起来，其必须继承自mmengine.model中的BaseModel
    # 实际上，runner的train是通过构建EpochBasedTrainLoop来进行模型训练的，该loop运行将调用模型的train_step方法
    # 同样，val和test中也会调用模型的val_step和test_step方法。
    class MMResNet50(BaseModel):
        def __init__(self):
            super().__init__()
            self.resnet = torchvision.models.resnet50()

        # 模型在train、val、test过程中都会调用_run_forward这个函数
        # 至于为什么要在forward的基础上再套一层这个呢？
        # 答：为了对函数实现进一步地抽象，方便接受dict形式的data、同时也能够接收tuple和list形式的data。
        # 必须实现的两大模式loss和predict，另外实现的可以是tensor模式
        def forward(self, imgs, labels, mode):
            x = self.resnet(imgs)
            if mode == 'loss':
                return {'loss': F.cross_entropy(x, labels)}
            elif mode == 'predict':
                return x, labels

    norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
    train_dataloader = DataLoader(batch_size=32,
                                  shuffle=True,
                                  dataset=torchvision.datasets.CIFAR10(
                                      'data/cifar10',
                                      train=True,
                                      download=True,
                                      transform=transforms.Compose([
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(**norm_cfg)
                                      ])))

    val_dataloader = DataLoader(batch_size=32,
                                shuffle=False,
                                dataset=torchvision.datasets.CIFAR10(
                                    'data/cifar10',
                                    train=False,
                                    download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(**norm_cfg)
                                    ])))

    model = MMResNet50()

    runner = Runner(
        model=model,
        work_dir='./workdir',
        train_dataloader=train_dataloader,
        optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.001, momentum=0.9)),
        train_cfg = dict(by_epoch=True, max_epochs=5, val_interval=1),
        val_dataloader = val_dataloader,
        val_cfg = dict(),
        val_evaluator = dict(type=Accuracy)
    )

    # 若出现重复的参数，如max_epochs则实际传入的参数为准
    epoch_based_train_loop = EpochBasedTrainLoop(
        runner=runner,
        dataloader=train_dataloader,
        max_epochs=1,
        val_begin=1,
        val_interval=1
    )

    # 构建optim_wrapper,梯度优化器封装实例。
    # 梯度优化器进一步封装的好处是可以少些几行代码，并更方便地实现混合精度训练和实现梯度累加。
    # 详情见https://mmengine.readthedocs.io/zh_CN/latest/tutorials/optim_wrapper.html?highlight=optimwrapper
    runner.optim_wrapper = runner.build_optim_wrapper(runner.optim_wrapper)
    runner.scale_lr(runner.optim_wrapper, runner.auto_scale_lr)
    # param_schedulers用于对优化器的学习率和动量的等固定参数进行动态调整。
    # runner.param_schedulers = runner.build_param_scheduler(runner.param_schedulers)

    # 初始化模型权重
    runner._init_model_weights()
    # 调用相关钩子，来读取模型权重和重启训练
    runner.load_or_resume()

    # 初始化优化器封装的内部参数（应该和梯度累加的计数器有关）
    runner.optim_wrapper.initialize_count_status(
        runner.model,
        epoch_based_train_loop.iter,  # type: ignore
        epoch_based_train_loop.max_iters)  # type: ignore

    # 跑一个iter,（由于一个before train iter中会调用一次Time相关Hook，但其写在run epoch中
    # 因此，这里无法调用
    # idx = 0
    # data_batch = next(iter(train_dataloader))
    # epoch_based_train_loop.run_iter(idx, data_batch)
    # print('one iter train has been done')

    # 跑一个epoch, 由于file_backend没有初始化，默认的run epoch会调用默认的钩子CheckpointHook
    # 因此，这里将失败
    # epoch_based_train_loop.run_epoch()
    # print('one epoch train has been done')

    # 按照设定的epochs运行
    epoch_based_train_loop.run()
    print('whole train flow has been done')


def IterBasedTrainLoop_learn():
    """
    IterBasedTrainLoop用于以iter为单位的模型训练流程，其train流程以batch为单位。
    该Loop同样被Runner的train调用。
    """

    from mmdet import models
    from mmdet.registry import MODELS
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import torchvision
    from mmengine.evaluator import BaseMetric
    import torch.nn.functional as F
    from mmengine.model import BaseModel

    # 评价指标的运行过程中，先进行process处理单个结果，随后进行compute_metrics计算输出指标。
    # 此处的Accuracy时简单的分类精度表示，表示recall。
    class Accuracy(BaseMetric):
        def process(self, data_batch, data_samples):
            score, gt = data_samples
            # 将一个批次的中间结果保存至 `self.results`列表中
            self.results.append({
                'batch_size': len(gt),
                'correct': (score.argmax(dim=1) == gt).sum().cpu(),
            })

        def compute_metrics(self, results):
            total_correct = sum(item['correct'] for item in results)
            total_size = sum(item['batch_size'] for item in results)
            # 返回保存有评测指标结果的字典，其中键为指标名称
            return dict(accuracy=100 * total_correct / total_size, testoutinfo='test')

    # 要使得自己构建的模型能够在runner中跑起来，其必须继承自mmengine.model中的BaseModel
    # 实际上，runner的train是通过构建EpochBasedTrainLoop来进行模型训练的，该loop运行将调用模型的train_step方法
    # 同样，val和test中也会调用模型的val_step和test_step方法。
    class MMResNet50(BaseModel):
        def __init__(self):
            super().__init__()
            self.resnet = torchvision.models.resnet50()

        # 模型在train、val、test过程中都会调用_run_forward这个函数
        # 至于为什么要在forward的基础上再套一层这个呢？
        # 答：为了对函数实现进一步地抽象，方便接受dict形式的data、同时也能够接收tuple和list形式的data。
        # 必须实现的两大模式loss和predict，另外实现的可以是tensor模式
        def forward(self, imgs, labels, mode):
            x = self.resnet(imgs)
            if mode == 'loss':
                return {'loss': F.cross_entropy(x, labels)}
            elif mode == 'predict':
                return x, labels

    norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
    train_dataloader = DataLoader(batch_size=32,
                                  shuffle=True,
                                  dataset=torchvision.datasets.CIFAR10(
                                      'data/cifar10',
                                      train=True,
                                      download=True,
                                      transform=transforms.Compose([
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(**norm_cfg)
                                      ])))

    val_dataloader = DataLoader(batch_size=32,
                                shuffle=False,
                                dataset=torchvision.datasets.CIFAR10(
                                    'data/cifar10',
                                    train=False,
                                    download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(**norm_cfg)
                                    ])))

    model = MMResNet50()

    runner = Runner(
        model=model,
        work_dir='./workdir',
        train_dataloader=train_dataloader,
        optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.001, momentum=0.9)),
        train_cfg = dict(by_epoch=False, max_iters=100, val_interval=100),
        val_dataloader = val_dataloader,
        val_cfg = dict(),
        val_evaluator = dict(type=Accuracy)
    )

    # 若出现重复的参数，如max_epochs则实际传入的参数为准
    iter_based_train_loop = IterBasedTrainLoop(
        runner=runner,
        dataloader=train_dataloader,
        max_iters=100,
        val_begin=100,
        val_interval=1
    )

    # 构建optim_wrapper,梯度优化器封装实例。
    # 梯度优化器进一步封装的好处是可以少些几行代码，并更方便地实现混合精度训练和实现梯度累加。
    # 详情见https://mmengine.readthedocs.io/zh_CN/latest/tutorials/optim_wrapper.html?highlight=optimwrapper
    runner.optim_wrapper = runner.build_optim_wrapper(runner.optim_wrapper)
    # 根据batch大小自动调整学习率
    runner.scale_lr(runner.optim_wrapper, runner.auto_scale_lr)
    # param_schedulers用于对优化器的学习率和动量的等固定参数进行动态调整。
    # runner.param_schedulers = runner.build_param_scheduler(runner.param_schedulers)

    # 初始化模型权重
    runner._init_model_weights()
    # 调用相关钩子，来读取模型权重和重启训练
    runner.load_or_resume()

    # 初始化优化器封装的内部参数（应该和梯度累加的计数器有关）
    runner.optim_wrapper.initialize_count_status(
        runner.model,
        iter_based_train_loop.iter,
        iter_based_train_loop.max_iters)

    # 按照设定的epochs运行
    iter_based_train_loop.run()
    print('whole train flow has been done')

def ValLoop_learn():
    """
    ValLoop用于模型验证流程，其流程大致与IterBasedTrainLoop相同，只不过其前向推理过程中被torch.nograd（）装饰了。
    """

    from mmdet import models
    from mmdet.registry import MODELS
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import torchvision
    from mmengine.evaluator import BaseMetric
    import torch.nn.functional as F
    from mmengine.model import BaseModel

    # 评价指标的运行过程中，先进行process处理单个结果，随后进行compute_metrics计算输出指标。
    # 此处的Accuracy时简单的分类精度表示，表示recall。
    class Accuracy(BaseMetric):
        def process(self, data_batch, data_samples):
            score, gt = data_samples
            # 将一个批次的中间结果保存至 `self.results`列表中
            self.results.append({
                'batch_size': len(gt),
                'correct': (score.argmax(dim=1) == gt).sum().cpu(),
            })

        def compute_metrics(self, results):
            total_correct = sum(item['correct'] for item in results)
            total_size = sum(item['batch_size'] for item in results)
            # 返回保存有评测指标结果的字典，其中键为指标名称
            return dict(accuracy=100 * total_correct / total_size, testoutinfo='test')

    # 要使得自己构建的模型能够在runner中跑起来，其必须继承自mmengine.model中的BaseModel
    # 实际上，runner的train是通过构建EpochBasedTrainLoop来进行模型训练的，该loop运行将调用模型的train_step方法
    # 同样，val和test中也会调用模型的val_step和test_step方法。
    class MMResNet50(BaseModel):
        def __init__(self):
            super().__init__()
            self.resnet = torchvision.models.resnet50()

        # 模型在train、val、test过程中都会调用_run_forward这个函数
        # 至于为什么要在forward的基础上再套一层这个呢？
        # 答：为了对函数实现进一步地抽象，方便接受dict形式的data、同时也能够接收tuple和list形式的data。
        # 必须实现的两大模式loss和predict，另外实现的可以是tensor模式
        def forward(self, imgs, labels, mode):
            x = self.resnet(imgs)
            if mode == 'loss':
                return {'loss': F.cross_entropy(x, labels)}
            elif mode == 'predict':
                return x, labels

    norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
    train_dataloader = DataLoader(batch_size=32,
                                  shuffle=True,
                                  dataset=torchvision.datasets.CIFAR10(
                                      'data/cifar10',
                                      train=True,
                                      download=True,
                                      transform=transforms.Compose([
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(**norm_cfg)
                                      ])))

    val_dataloader = DataLoader(batch_size=32,
                                shuffle=False,
                                dataset=torchvision.datasets.CIFAR10(
                                    'data/cifar10',
                                    train=False,
                                    download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(**norm_cfg)
                                    ])))

    model = MMResNet50()

    runner = Runner(
        model=model,
        work_dir='./workdir',
        train_dataloader=train_dataloader,
        optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.001, momentum=0.9)),
        train_cfg=dict(by_epoch=False, max_iters=100, val_interval=100),
        val_dataloader=val_dataloader,
        val_cfg=dict(),
        val_evaluator=dict(type=Accuracy)
    )

    # 若出现重复的参数，以实际传入的参数为准
    val_loop = ValLoop(
        runner=runner,
        dataloader=val_dataloader,
        evaluator=runner.val_evaluator,
        fp16=False
    )

    # 加载权重
    runner.load_or_resume()

    # 按照设定的epochs运行
    val_loop.run()
    print('whole val flow has been done')

def TestLoop_learn():
    """
    TestLoop同ValLoop流程一致，此处不再赘述
    """
    """
    ValLoop用于模型验证流程，其流程大致与IterBasedTrainLoop相同，只不过其前向推理过程中被torch.nograd（）装饰了。
    """

    from mmdet import models
    from mmdet.registry import MODELS
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import torchvision
    from mmengine.evaluator import BaseMetric
    import torch.nn.functional as F
    from mmengine.model import BaseModel

    # 评价指标的运行过程中，先进行process处理单个结果，随后进行compute_metrics计算输出指标。
    # 此处的Accuracy时简单的分类精度表示，表示recall。
    class Accuracy(BaseMetric):
        def process(self, data_batch, data_samples):
            score, gt = data_samples
            # 将一个批次的中间结果保存至 `self.results`列表中
            self.results.append({
                'batch_size': len(gt),
                'correct': (score.argmax(dim=1) == gt).sum().cpu(),
            })

        def compute_metrics(self, results):
            total_correct = sum(item['correct'] for item in results)
            total_size = sum(item['batch_size'] for item in results)
            # 返回保存有评测指标结果的字典，其中键为指标名称
            return dict(accuracy=100 * total_correct / total_size, testoutinfo='test')

    # 要使得自己构建的模型能够在runner中跑起来，其必须继承自mmengine.model中的BaseModel
    # 实际上，runner的train是通过构建EpochBasedTrainLoop来进行模型训练的，该loop运行将调用模型的train_step方法
    # 同样，val和test中也会调用模型的val_step和test_step方法。
    class MMResNet50(BaseModel):
        def __init__(self):
            super().__init__()
            self.resnet = torchvision.models.resnet50()

        # 模型在train、val、test过程中都会调用_run_forward这个函数
        # 至于为什么要在forward的基础上再套一层这个呢？
        # 答：为了对函数实现进一步地抽象，方便接受dict形式的data、同时也能够接收tuple和list形式的data。
        # 必须实现的两大模式loss和predict，另外实现的可以是tensor模式
        def forward(self, imgs, labels, mode):
            x = self.resnet(imgs)
            if mode == 'loss':
                return {'loss': F.cross_entropy(x, labels)}
            elif mode == 'predict':
                return x, labels

    norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
    train_dataloader = DataLoader(batch_size=32,
                                  shuffle=True,
                                  dataset=torchvision.datasets.CIFAR10(
                                      'data/cifar10',
                                      train=True,
                                      download=True,
                                      transform=transforms.Compose([
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(**norm_cfg)
                                      ])))

    val_dataloader = DataLoader(batch_size=32,
                                shuffle=False,
                                dataset=torchvision.datasets.CIFAR10(
                                    'data/cifar10',
                                    train=False,
                                    download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(**norm_cfg)
                                    ])))

    model = MMResNet50()

    runner = Runner(
        model=model,
        work_dir='./workdir',
        train_dataloader=train_dataloader,
        optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.001, momentum=0.9)),
        train_cfg=dict(by_epoch=False, max_iters=100, val_interval=100),
        val_dataloader=val_dataloader,
        val_cfg=dict(),
        val_evaluator=dict(type=Accuracy),
        test_dataloader=val_dataloader,
        test_cfg=dict(),
        test_evaluator=dict(type=Accuracy)
    )

    # 若出现重复的参数，以实际传入的参数为准
    test_loop = TestLoop(
        runner=runner,
        dataloader=val_dataloader,
        evaluator=runner.val_evaluator,
        fp16=False
    )

    # 加载权重
    runner.load_or_resume()

    # 按照设定的epochs运行
    test_loop.run()
    print('whole test flow has been done')

def Runner_learn():
    """
    Runner为最为重要的部分，其包含了所有深度学习模型的三大核心过程，train、val、test。
    train中调用EpochBasedTrainLoop或IterBasedTrainLoop,
    val中调用ValLoop，
    test中调用TestLoop。
    其另外的核心功能就是在其初始化过程中会自动将缺省的参数补全。
    支持两种实例化方法，分别为init实例化方法和from_cfg实例化方法。
    """

    from mmdet import models
    from mmdet.registry import MODELS
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import torchvision
    from mmengine.evaluator import BaseMetric
    import torch.nn.functional as F
    from mmengine.model import BaseModel

    # 评价指标的运行过程中，先进行process处理单个结果，随后进行compute_metrics计算输出指标。
    # 此处的Accuracy时简单的分类精度表示，表示recall。
    class Accuracy(BaseMetric):
        def process(self, data_batch, data_samples):
            score, gt = data_samples
            # 将一个批次的中间结果保存至 `self.results`列表中
            self.results.append({
                'batch_size': len(gt),
                'correct': (score.argmax(dim=1) == gt).sum().cpu(),
            })

        def compute_metrics(self, results):
            total_correct = sum(item['correct'] for item in results)
            total_size = sum(item['batch_size'] for item in results)
            # 返回保存有评测指标结果的字典，其中键为指标名称
            return dict(accuracy=100 * total_correct / total_size, testoutinfo='test')

    # 要使得自己构建的模型能够在runner中跑起来，其必须继承自mmengine.model中的BaseModel
    # 实际上，runner的train是通过构建EpochBasedTrainLoop来进行模型训练的，该loop运行将调用模型的train_step方法
    # 同样，val和test中也会调用模型的val_step和test_step方法。
    class MMResNet50(BaseModel):
        def __init__(self):
            super().__init__()
            self.resnet = torchvision.models.resnet50()

        # 模型在train、val、test过程中都会调用_run_forward这个函数
        # 至于为什么要在forward的基础上再套一层这个呢？
        # 答：为了对函数实现进一步地抽象，方便接受dict形式的data、同时也能够接收tuple和list形式的data。
        # 必须实现的两大模式loss和predict，另外实现的可以是tensor模式
        def forward(self, imgs, labels, mode):
            x = self.resnet(imgs)
            if mode == 'loss':
                return {'loss': F.cross_entropy(x, labels)}
            elif mode == 'predict':
                return x, labels

    norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
    train_dataloader = DataLoader(batch_size=32,
                                  shuffle=True,
                                  dataset=torchvision.datasets.CIFAR10(
                                      'data/cifar10',
                                      train=True,
                                      download=True,
                                      transform=transforms.Compose([
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(**norm_cfg)
                                      ])))

    val_dataloader = DataLoader(batch_size=32,
                                shuffle=False,
                                dataset=torchvision.datasets.CIFAR10(
                                    'data/cifar10',
                                    train=False,
                                    download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(**norm_cfg)
                                    ])))

    model = MMResNet50()

    # 方法1：通过init方法初始化runner对象
    runner = Runner(
        model=model,
        work_dir='./workdir',
        train_dataloader=train_dataloader,
        optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.001, momentum=0.9)),
        train_cfg=dict(by_epoch=False, max_iters=100, val_interval=100),
        val_dataloader=val_dataloader,
        val_cfg=dict(),
        val_evaluator=dict(type=Accuracy),
        test_dataloader=val_dataloader,
        test_cfg=dict(),
        test_evaluator=dict(type=Accuracy)
    )

    runner.run()

    # 方法2:通过cfg方法初始化runner对象
    runner2 = Runner.from_cfg(runner.cfg)
    runner2.run()

def get_priority_learn():
    """
    根据优先级名称，给出对应的优先级数字；
    主要用于runner中调用hook时，对其优先级判断。
    数字越大，优先级越低，具体优先级如下：
        +--------------+------------+
    | Level        | Value      |
    +==============+============+
    | HIGHEST      | 0          |
    +--------------+------------+
    | VERY_HIGH    | 10         |
    +--------------+------------+
    | HIGH         | 30         |
    +--------------+------------+
    | ABOVE_NORMAL | 40         |
    +--------------+------------+
    | NORMAL       | 50         |
    +--------------+------------+
    | BELOW_NORMAL | 60         |
    +--------------+------------+
    | LOW          | 70         |
    +--------------+------------+
    | VERY_LOW     | 90         |
    +--------------+------------+
    | LOWEST       | 100        |
    +--------------+------------+
    """
    # 输入数字，获取优先级
    priority1 = get_priority(10)
    print(priority1)

    # 输入字符串，获取优先级
    priority2 = get_priority('ABOVE_NORMAL')
    print(priority2)

    # 输入Priority对象（注意，构建priority只能输入它预先设定好的值）
    priority_instance = Priority(10)
    priority3 = get_priority(priority_instance)
    print(priority3)

def Priority_learn():
    """
    Priority为枚举类别Enum的子类实现，可通过字典形式来访问其预设的枚举值。
    """

    from enum import Enum

    class MyPriority(Enum):
        attr1 = 'test'

    print(MyPriority['attr1'])
    print(MyPriority['attr1'].value)
    # 不可增加或更改其属性
    MyPriority.attr2 = 'test2'
    # MyPriority['attr1'] = 90
    print(MyPriority['attr2'])

def find_latest_checkpoint_learn():
    """
    寻找指定目录下的last_checpoint文件，读取并返回。（感觉没必要重写这~.~).
    代码逻辑如下，非常简单
    save_file = osp.join(path, 'last_checkpoint')
    last_saved: Optional[str]
    if os.path.exists(save_file):
        with open(save_file) as f:
            last_saved = f.read().strip()
    else:
        print_log('Did not find last_checkpoint to be resumed.')
        last_saved = None
    return last_saved
    """
    pass

def autocast_learn():
    """
    torch.autocast和torch.cuda.amp.autocast的封装。使用此函数需要pytorch版本>=1.5.0
    """

    # 混合精度训练
    with autocast():
        pass

    # 在cpu开启不了混合精度，只支持nvidia显卡?
    with autocast(device_type='cpu'):
        pass

def LogProcessor_learn():
    """
    用来对runner.message_hub.log_scalars中的日志信息进行处理。
    此实例对象将吧runner.message_hub.log_scalars从tag转换到log_str。
    默认配置下，日志处理器会统计最近一次更新的学习率、基于迭代次数平滑的损失和迭代时间。具体如下：
    04/15 12:34:24 - mmengine - INFO - Iter [10/12]  ,
    eta: 0:00:00, time: 0.003, data_time: 0.002, loss: 0.13。
    可以通过配置 custom_cfg 列表来选择日志的统计方式。custom_cfg 中的每一个元素需要包括以下信息：

    data_src：日志的数据源，用户通过指定 data_src 来选择需要被重新统计的日志，一份数据源可以有多种统计方式。
    默认的日志源包括模型输出的损失字典的 key、学习率（lr）和迭代时间（time/data_time），
    一切经消息枢纽的 update_scalar/update_scalars 更新的日志均为可以配置的数据源
    （需要去掉 train/、val/ 前缀）。（必填项）
    method_name：日志的统计方法，即历史缓冲区中的基本统计方法以及用户注册的自定义统计方法（必填项）
    log_name：日志被重新统计后的名字，如果不定义 log_name，新日志会覆盖旧日志（选填项）
    其他参数：统计方法会用到的参数，其中 window_size 为特殊字段，可以为普通的整型、
    字符串 epoch 和字符串 global。LogProcessor 会实时解析这些参数，
    以返回基于 iteration、epoch 和全局平滑的统计结果（选填项）
    """

    # 一般方法，以10个epoch对日志进行一次处理
    log_processor = dict(
        window_size=10,
        by_epoch=True
    )

    # 用户自定义新字段，将覆盖旧设置（对原loss字段的100个iter计算均值后输出到loss中）
    log_processor = dict(
        window_size=10,
        by_epoch=True,
        custom_cfg=[
            dict(data_src='loss',
                 method_name='mean',
                 window_size=100)])

    # 用户自定义新字段，不覆盖就设置（对原loss字段的100个iter计算均值后输出到loss_large_window中）
    log_processor = dict(
        window_size=10,
        by_epoch=True,
        custom_cfg=[
            dict(data_src='loss',
                 log_name='loss_large_window',
                 method_name='mean',
                 window_size=100)])

    # 通过不同方法定义两个字段
    log_processor = dict(
        window_size=10,
        by_epoch=True,
        custom_cfg=[
            dict(data_src='loss',
                 log_name='loss_large_window',
                 method_name='mean',
                 window_size=100),
            dict(data_src='loss',
                 method_name='mean',
                 window_size=100)
        ])

    # 用户自定义过程中通过一个字段覆盖两次，将报错
    log_processor = dict(
        window_size=10,
        by_epoch=True,
        custom_cfg=[
            dict(data_src='loss',
                 method_name='mean',
                 window_size=100),
            dict(data_src='loss',
                 method_name='max',
                 window_size=100)
        ])

def set_random_seed_learn():
    """
    set_random_seed用于固定训练过程中的相关随机数种子，包括设置对应cudnn和分布式的相关设置。
    注意：即使全部固定了，若模型中的训练过程中存在cuda的特殊操作，仍然可能在梯度计算过程中发生数值溢出，从而造成无法复现的问题。
    """
    set_random_seed(seed=1, deterministic=True, diff_rank_seed=False)

def FlexibleRunner_learn():
    """
    FlexibleRunner相比于Runner是一个更为灵活的Runner执行器。
    相比于Runner，FlexibleRunner首先会将初始化model，且只有在对应的训练流程中初始化相关模块，而不必在init中对所有相关组件全部初始化。
    """

    from mmdet import models
    from mmdet.registry import MODELS
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import torchvision
    from mmengine.evaluator import BaseMetric
    import torch.nn.functional as F
    from mmengine.model import BaseModel

    # 评价指标的运行过程中，先进行process处理单个结果，随后进行compute_metrics计算输出指标。
    # 此处的Accuracy时简单的分类精度表示，表示recall。
    class Accuracy(BaseMetric):
        def process(self, data_batch, data_samples):
            score, gt = data_samples
            # 将一个批次的中间结果保存至 `self.results`列表中
            self.results.append({
                'batch_size': len(gt),
                'correct': (score.argmax(dim=1) == gt).sum().cpu(),
            })

        def compute_metrics(self, results):
            total_correct = sum(item['correct'] for item in results)
            total_size = sum(item['batch_size'] for item in results)
            # 返回保存有评测指标结果的字典，其中键为指标名称
            return dict(accuracy=100 * total_correct / total_size, testoutinfo='test')

    # 要使得自己构建的模型能够在runner中跑起来，其必须继承自mmengine.model中的BaseModel
    # 实际上，runner的train是通过构建EpochBasedTrainLoop来进行模型训练的，该loop运行将调用模型的train_step方法
    # 同样，val和test中也会调用模型的val_step和test_step方法。
    class MMResNet50(BaseModel):
        def __init__(self):
            super().__init__()
            self.resnet = torchvision.models.resnet50()

        # 模型在train、val、test过程中都会调用_run_forward这个函数
        # 至于为什么要在forward的基础上再套一层这个呢？
        # 答：为了对函数实现进一步地抽象，方便接受dict形式的data、同时也能够接收tuple和list形式的data。
        # 必须实现的两大模式loss和predict，另外实现的可以是tensor模式
        def forward(self, imgs, labels, mode):
            x = self.resnet(imgs)
            if mode == 'loss':
                return {'loss': F.cross_entropy(x, labels)}
            elif mode == 'predict':
                return x, labels

    norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
    train_dataloader = DataLoader(batch_size=32,
                                  shuffle=True,
                                  dataset=torchvision.datasets.CIFAR10(
                                      'data/cifar10',
                                      train=True,
                                      download=True,
                                      transform=transforms.Compose([
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(**norm_cfg)
                                      ])))

    val_dataloader = DataLoader(batch_size=32,
                                shuffle=False,
                                dataset=torchvision.datasets.CIFAR10(
                                    'data/cifar10',
                                    train=False,
                                    download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(**norm_cfg)
                                    ])))

    model = MMResNet50()

    runner = FlexibleRunner(
        model=model,
        work_dir='./workdir',
        train_dataloader=train_dataloader,
        optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.001, momentum=0.9)),
        train_cfg=dict(by_epoch=False, max_iters=100, val_interval=100),
        val_dataloader=val_dataloader,
        val_cfg=dict(),
        val_evaluator=dict(type=Accuracy),
        test_dataloader=val_dataloader,
        test_cfg=dict(),
        test_evaluator=dict(type=Accuracy)
    )

    runner.train()


if __name__ == '__main__':
    FlexibleRunner_learn()