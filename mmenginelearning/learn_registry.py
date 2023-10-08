"""
本部分内容用于学习mmengine中的registry目录
作为openmmlab中最基础的部分，mmengine框架下包含多个目录和两个文件（__init__.py、version.py）
其中，__init__.py用于将各个目录组成的包集成；version.py用于记录表示当前mmengine的版本号
registry目录共包含以下几个类和方法
'Registry', 'RUNNERS', 'RUNNER_CONSTRUCTORS', 'HOOKS', 'DATASETS',
'DATA_SAMPLERS', 'TRANSFORMS', 'MODELS', 'WEIGHT_INITIALIZERS',
'OPTIMIZERS', 'OPTIM_WRAPPER_CONSTRUCTORS', 'TASK_UTILS',
'PARAM_SCHEDULERS', 'METRICS', 'MODEL_WRAPPERS', 'OPTIM_WRAPPERS', 'LOOPS',
'VISBACKENDS', 'VISUALIZERS', 'LOG_PROCESSORS', 'EVALUATOR', 'INFERENCERS',
'DefaultScope', 'traverse_registry_tree', 'count_registered_modules',
'build_model_from_cfg', 'build_runner_from_cfg', 'build_from_cfg',
'build_scheduler_from_cfg', 'init_default_scope', 'FUNCTIONS', 'STRATEGIES'
"""

from mmengine.registry import *


def Registry_learn():
    """
    Registry类的功能为将输入的字符串映射为对象或函数。
    一个注册器对象可以从Rigistry中获取，一个注册函数也可以从Registry中获取
    """
    # 简单地定义注册器
    MyModels = Registry('mymodels')

    # 将某个类注册到注册器中
    @MyModels.register_module()
    class MyResNet:
        pass

    # 从注册器中实例化一个类（注意type名称要与类名一致）
    mymodel = MyModels.build(dict(type='MyResNet'))
    print(mymodel)  # <__main__.Registry_learn.<locals>.MyResNet object at 0x0000018DB64B9940>

    # 定义一个层级结构的注册器, 此处为设置mymodels的下游任务结构模型
    MyDetModels = Registry('mydetmodels', parent=MyModels, scope='mydet')

    @MyDetModels.register_module()
    class MyDet:
        pass

    # 构建一个下游实例模型
    mydetmodel = MyDetModels.build(dict(type='MyDet'))
    print(mydetmodel)  # <__main__.Registry_learn.<locals>.MyDet object at 0x0000025D7D93E640>
    # 从mydet作用域下实例化一个模型
    mydetmodel = MyDetModels.build(dict(type='mydet.MyDet'))
    print(mydetmodel)  # <__main__.Registry_learn.<locals>.MyDet object at 0x00000188A88C51F0>
    # 从下游注册器中获取上游实例模型
    mymodel2 = MyDetModels.build(dict(type='MyResNet'))
    print(mymodel2)  # <__main__.Registry_learn.<locals>.MyResNet object at 0x000001B7A6165220>


def RUNNERS_learn():
    """
    Runner是registry目录下的root.py中定义的一个注册器实例化对象
    通过此注册器可以顺利拿到mmengine中定义的一系列runner对象来进行模型的训练和测试
    """
    # 由于此注册器的实例化过程中，定义了build_func，此时通过build实例化模型传入的便是build_runner_from_cfg(默认的是build_from_cfg)
    # RUNNERS = Registry('runner', build_func=build_runner_from_cfg)
    # 此时通过下面的方式定义一个Runner和实例化Runner对象将会报错
    # 自定义一个runner，导入注册器
    # @RUNNERS.register_module()
    # class MyRunner:
    #     pass
    # runner = RUNNERS.build(dict(type='MyRunner'))
    # print(runner)

    # 正确实例化方式如下
    MYRUNNERS = Registry('myrunners', build_func=build_runner_from_cfg)

    @MYRUNNERS.register_module()
    class MyRunner():
        @classmethod
        def from_cfg(cls, cfg):
            print(cfg)

    runner = MYRUNNERS.build(dict(runner_type='MyRunner', peseduo_param=1))
    print(runner)


def RUNNER_CONSTRUCTORS_learn():
    """
    RUNNER_CONSTRUCTORS也是一个注册器，相比于RUNNERS注册器，其设置的build_func并不是build_runner_from_cfg
    不是必须传入Runner基类中的类方法中的对应参数，其参数仅需要自己建立的初始化方法（注意python的构造函数是__new__())
    因此，RUNNER_CONSTURCTORS与RUNNERS的区别就是，RUNNERS通过from_cfg来实例化Runner，而RUNNER CONSTRUCTOR通过
    初始化方法来实例化Runner
    """
    MYRUNNERS = Registry('my runner constructor', build_func=build_from_cfg)

    @MYRUNNERS.register_module()
    class MYRunner():
        def __init__(self, peseduo_param):
            print(peseduo_param)

    myrunner = MYRUNNERS.build(dict(type='MYRunner', peseduo_param=1))
    print(myrunner)  # <__main__.RUNNER_CONSTRUCTORS_learn.<locals>.MYRunner object at 0x000002034B80A130>


def HOOKS_learn():
    """
    HOOKS为mmengine中定义的另一系列的注册器，其主要作用于Runner执行器进行模型训练过程中的各个生命周期。
    """
    MyHOOKS = Registry('my hooks')

    @MyHOOKS.register_module()
    class MyHook():
        def __init__(self):
            pass

    myhook = MyHOOKS.build(dict(type="MyHook"))  # <__main__.HOOKS_learn.<locals>.MyHook object at 0x00000195AEEF9F40>
    print(myhook)


def DATASETS_learn():
    """
    DATASETS同样为一普通注册器，用于管理模型训练过程中的各个数据集类
    """
    MyDATASETS = Registry('my dataset')

    @MyDATASETS.register_module()
    class MyDataset():
        pass

    mydataset = MyDATASETS.build(
        dict(type="MyDataset"))  # <__main__.DATASETS_learn.<locals>.MyDataset object at 0x0000021BDDC8E0A0>
    print(mydataset)


def DATA_SAMPLERS_learn():
    """
    DATASAMPLES同样为一普通注册器，用于管理模型训练过程中的dataloader的随机采样过程
    """
    MyDATASAMPLES = Registry('my datasampler')

    @MyDATASAMPLES.register_module()
    class MyDataSampler():
        pass

    mydatasampler = MyDATASAMPLES.build(dict(
        type="MyDataSampler"))  # <__main__.DATA_SAMPLERS_learn.<locals>.MyDataSampler object at 0x00000158FFBFE250>
    print(mydatasampler)


def TRANSFORMERS_learn():
    """
    TRANSFROMERS同样为一普通注册器，用于处理Dataset到Dataloader之间的数据流。
    """
    MyTRANSFORMERS = Registry('my transformer')

    @MyTRANSFORMERS.register_module()
    class MyTransformer():
        pass

    mytransformer = MyTRANSFORMERS.build(dict(type='MyTransformer'))
    print(mytransformer)  # <__main__.TRANSFORMERS_learn.<locals>.MyTransformer object at 0x0000022EFA2F3310>


def WEIGHT_INITIALIZERS_learn():
    """
    WEIGHT_INITIALIZERS同样为一普通注册器，用于进行模型的权重初始化。
    """
    MyWEIGHT_INITIALIZERS = Registry('my weight_initializer')

    @MyWEIGHT_INITIALIZERS.register_module()
    class MyWeight_initializer():
        pass

    myweight_initializer = MyWEIGHT_INITIALIZERS.build(dict(type='MyWeight_initializer'))
    print(myweight_initializer)


def OPTIMIZERS_learn():
    """
    OPTIMIZERS同样为一普通注册器，用于进行模型梯度优化器的构建。
    """
    MyOPTIMIZERS = Registry('my optimizer')

    @MyOPTIMIZERS.register_module()
    class MyOptimizer():
        pass

    myoptimizer = MyOPTIMIZERS.build(dict(type='MyOptimizer'))
    print(myoptimizer)


def OPTIM_WRAPPER_CONSTRUCTORS_learn():
    """
    OPTIM_WRAPPER_CONSTRUCTORS同样为一普通注册器，用于进行模型梯度优化器的封装。
    constructor可以自由地定义模型的相关参数来构建对象。
    """
    MyOPTIM_WRAPPER_CONSTRUCTORS = Registry('my optim_wrapper_constructor')

    @MyOPTIM_WRAPPER_CONSTRUCTORS.register_module()
    class MyOptim_wrapper_constructor():
        pass

    myoptim_wrapper_constructor = MyOPTIM_WRAPPER_CONSTRUCTORS.build(dict(type='MyOptim_wrapper_constructor'))
    print(myoptim_wrapper_constructor)


def TASK_UTILS_learn():
    """
    TASK_UTILS任务强相关的一些组件，如 AnchorGenerator, BboxCoder
    """
    MyTASK_UTILS = Registry('my task_util')

    @MyTASK_UTILS.register_module()
    class MyTask_util():
        pass

    mytask_util = MyTASK_UTILS.build(dict(type='MyTask_util'))
    print(mytask_util)


def PARAM_SCHEDULERS_learn():
    """
    PARAM_SCHEDULERS各种参数调度器，如 MultiStepLR
    """
    MyPARAM_SCHEDULERS = Registry('my param_scheduler')

    @MyPARAM_SCHEDULERS.register_module()
    class MyParam_scheduler():
        pass

    myparam_scheduler = MyPARAM_SCHEDULERS.build(dict(type='MyParam_scheduler'))
    print(myparam_scheduler)


def METRICS_learn():
    """
    METRICS用于计算模型精度的评估指标，如 Accuracy
    """
    MyMETRICS = Registry('my metric')

    @MyMETRICS.register_module()
    class MyMetric():
        pass

    mymetric = MyMETRICS.build(dict(type='MyMetric'))
    print(mymetric)


def MODEL_WRAPPERS_learn():
    """
    MODEL_WRAPPERS: 模型的包装器，如 MMDistributedDataParallel，用于对分布式数据并行
    """
    MyMODEL_WRAPPERS = Registry('my model_wrapper')

    @MyMODEL_WRAPPERS.register_module()
    class MyModel_wrapper():
        pass

    mymodel_wrapper = MyMODEL_WRAPPERS.build(dict(type='MyModel_wrapper'))
    print(mymodel_wrapper)


def OPTIM_WRAPPERS_learn():
    """
    OPTIM_WRAPPER: 对 Optimizer 相关操作的封装，如 OptimWrapper，AmpOptimWrapper
    """
    MyOPTIM_WRAPPERS = Registry('my optim_wrapper')

    @MyOPTIM_WRAPPERS.register_module()
    class MyOptim_wrapper():
        pass

    myoptim_wrapper = MyOPTIM_WRAPPERS.build(dict(type='MyOptim_wrapper'))
    print(myoptim_wrapper)


def LOOPS_learn():
    """
    OPTIM_WRAPPER: 对 Optimizer 相关操作的封装，如 OptimWrapper，AmpOptimWrapper
    """
    MyLOOPS = Registry('my loop')

    @MyLOOPS.register_module()
    class MyLoop():
        pass

    myloop = MyLOOPS.build(dict(type='MyLoop'))
    print(myloop)


def VISBACKENDS_learn():
    """
    VISBACKENDS: 存储训练日志的后端，如 LocalVisBackend, TensorboardVisBackend
    """
    MyVISBACKENDS = Registry('my visbackend')

    @MyVISBACKENDS.register_module()
    class MyVisbackend():
        pass

    myvisbackend = MyVISBACKENDS.build(dict(type='MyVisbackend'))
    print(myvisbackend)


def VISUALIZERS_learn():
    """
    VISUALIZERS: 管理绘制模块，如 DetVisualizer 可在图片上绘制预测框
    """
    MyVISUALIZERS = Registry('my visualizer')

    @MyVISUALIZERS.register_module()
    class MyVisualizer():
        pass

    myvisualizer = MyVISUALIZERS.build(dict(type='MyVisualizer'))
    print(myvisualizer)


def LOG_PROCESSORS_learn():
    """
    LOG_PROCESSORS: 控制日志的统计窗口和统计方法，默认使用 LogProcessor，如有特殊需求可自定义 LogProcessor
    """
    MyLOG_PROCESSORS = Registry('my log_processor')

    @MyLOG_PROCESSORS.register_module()
    class MyLog_processor():
        pass

    mylog_processor = MyLOG_PROCESSORS.build(dict(type='MyLog_processor'))
    print(mylog_processor)


def EVALUATOR_learn():
    """
    EVALUATOR: 用于计算模型精度的一个或多个评估指标
    """
    MyEVALUATOR = Registry('my evaluato')

    @MyEVALUATOR.register_module()
    class MyEvaluato():
        pass

    myevaluato = MyEVALUATOR.build(dict(type='MyEvaluato'))
    print(myevaluato)


def INFERENCERS_learn():
    """
    INFERENCERS用于管理模型推理类方法
    """
    MyINFERENCERS = Registry('my inferencer')

    @MyINFERENCERS.register_module()
    class MyInferencer():
        pass

    myinferencer = MyINFERENCERS.build(dict(type='MyInferencer'))
    print(myinferencer)

def FUNCTIONS_learn():
    """
    FUNCTIONS用于管理相关函数
    """
    MyFUNCTIONS = Registry('my function')

    @MyFUNCTIONS.register_module()
    class MyFunction():
        pass

    myfunction = MyFUNCTIONS.build(dict(type='MyFunction'))
    print(myfunction)


def STRATEGIES_learn():
    """
    STRATEGIES用于管理Runner执行器训练过程的一系列策略
    """
    MySTRATEGIES = Registry('my strategie')

    @MySTRATEGIES.register_module()
    class MyStrategie():
        pass

    mystrategie = MySTRATEGIES.build(dict(type='MyStrategie'))
    print(mystrategie)

def DefaultScope_learn():
    """
    DefaultScope用于全局管理作用域，只要在一次程序运行过程中创建过对象，
    之后，便可以在程序的任何其他地方获取该实例对象
    详细说明可见https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/manager_mixin.html
    """
    # 首次通过get_instance，作用为将task中的scope name存放到对应的全局管理器中
    DefaultScope.get_instance(name='task', scope_name='myscope_name')
    # 首次通过get_instance，作用为将task2中的scope name存放到对应的全局管理器中
    DefaultScope.get_instance(name='task2', scope_name='myscope_name2')
    # 获取各自任务中的scope name（可在程序的任何地方获取得到，只要此全局管理器实例化完成）
    task1 = DefaultScope.get_instance('task')
    task1_name = DefaultScope.get_instance('task').scope_name
    task2_name = DefaultScope.get_instance('task2').scope_name
    print(task1_name, task2_name)   # myscope_name myscope_name2
    print(task1) # <mmengine.registry.default_scope.DefaultScope object at 0x00000296EDAB78E0>


def traverse_registry_tree_learn():
    """
    从给定输入的注册器中获取其中包含的所有模块信息，感觉比较有用。
    """
    # 统计对应注册器模块之前，必须将其包导入
    import mmengine.model
    # 打印MODELS中的所有模块
    traverse_registry_tree(MODELS, verbose=True)


def count_registered_modules_learn():
    """
    count_registered_modules用于统计mmengine包下各类注册器中的所有模块，并打印成json格式输出
    """
    # save_path为统计的json存放目录
    count_results = count_registered_modules(save_path=None, verbose=True)
    for k, v in count_results['registries'].items():
        print(f'{k} count result:')
        for attr_name, attr in v[0].items():
            print(attr_name + ':\t' + str(attr))
        print('\n')
    """
    DATASETS count result:
    num_modules:	3
    scope:	mmengine
    mmengine/dataset:	['ConcatDataset', 'RepeatDataset', 'ClassBalancedDataset']
    
    
    DATA_SAMPLERS count result:
    num_modules:	2
    scope:	mmengine
    mmengine/dataset:	['DefaultSampler', 'InfiniteSampler']
    
    
    EVALUATOR count result:
    num_modules:	1
    scope:	mmengine
    mmengine/evaluator:	['Evaluator']
    
    
    FUNCTIONS count result:
    num_modules:	2
    scope:	mmengine
    mmengine/dataset:	['pseudo_collate', 'default_collate']
    
    
    HOOKS count result:
    num_modules:	14
    scope:	mmengine
    mmengine/hooks:	['CheckpointHook', 'EarlyStoppingHook', 'EMAHook', 'EmptyCacheHook', 'IterTimerHook', 'LoggerHook', 'NaiveVisualizationHook', 'ParamSchedulerHook', 'ProfilerHook', 'NPUProfilerHook', 'RuntimeInfoHook', 'DistSamplerSeedHook', 'SyncBuffersHook', 'PrepareTTAHook']
    
    
    INFERENCERS count result:
    num_modules:	0
    scope:	mmengine
    
    
    LOG_PROCESSORS count result:
    num_modules:	1
    scope:	mmengine
    mmengine/runner:	['LogProcessor']
    
    
    LOOPS count result:
    num_modules:	4
    scope:	mmengine
    mmengine/runner:	['EpochBasedTrainLoop', 'IterBasedTrainLoop', 'ValLoop', 'TestLoop']
    
    
    METRICS count result:
    num_modules:	1
    scope:	mmengine
    mmengine/evaluator:	['DumpResults']
    
    
    MODELS count result:
    num_modules:	6
    scope:	mmengine
    mmengine/model:	['StochasticWeightAverage', 'ExponentialMovingAverage', 'MomentumAnnealingEMA', 'BaseTTAModel']
    mmengine/model/base_model:	['BaseDataPreprocessor', 'ImgDataPreprocessor']
    
    
    MODEL_WRAPPERS count result:
    num_modules:	4
    scope:	mmengine
    torch/nn/parallel:	['DistributedDataParallel', 'DataParallel']
    mmengine/model/wrappers:	['MMDistributedDataParallel', 'MMSeparateDistributedDataParallel']
    
    
    OPTIMIZERS count result:
    num_modules:	13
    scope:	mmengine
    torch/optim:	['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
    mmengine/optim/optimizer:	['ZeroRedundancyOptimizer']
    
    
    OPTIM_WRAPPERS count result:
    num_modules:	3
    scope:	mmengine
    mmengine/optim/optimizer:	['OptimWrapper', 'AmpOptimWrapper', 'ApexOptimWrapper']
    
    
    OPTIM_WRAPPER_CONSTRUCTORS count result:
    num_modules:	1
    scope:	mmengine
    mmengine/optim/optimizer:	['DefaultOptimWrapperConstructor']
    
    
    PARAM_SCHEDULERS count result:
    num_modules:	29
    scope:	mmengine
    mmengine/optim/scheduler:	['StepParamScheduler', 'MultiStepParamScheduler', 'ConstantParamScheduler', 'ExponentialParamScheduler', 'CosineAnnealingParamScheduler', 'LinearParamScheduler', 'PolyParamScheduler', 'OneCycleParamScheduler', 'CosineRestartParamScheduler', 'ReduceOnPlateauParamScheduler', 'ConstantLR', 'CosineAnnealingLR', 'ExponentialLR', 'LinearLR', 'MultiStepLR', 'StepLR', 'PolyLR', 'OneCycleLR', 'CosineRestartLR', 'ReduceOnPlateauLR', 'ConstantMomentum', 'CosineAnnealingMomentum', 'ExponentialMomentum', 'LinearMomentum', 'MultiStepMomentum', 'StepMomentum', 'PolyMomentum', 'CosineRestartMomentum', 'ReduceOnPlateauMomentum']
    
    
    RUNNERS count result:
    num_modules:	2
    scope:	mmengine
    mmengine/runner:	['FlexibleRunner', 'Runner']
    
    
    RUNNER_CONSTRUCTORS count result:
    num_modules:	0
    scope:	mmengine
    
    
    STRATEGIES count result:
    num_modules:	2
    scope:	mmengine
    mmengine/_strategy:	['SingleDeviceStrategy', 'DDPStrategy']
    
    
    TASK_UTILS count result:
    num_modules:	0
    scope:	mmengine
    
    
    TRANSFORMS count result:
    num_modules:	0
    scope:	mmengine
    
    
    VISBACKENDS count result:
    num_modules:	5
    scope:	mmengine
    mmengine/visualization:	['LocalVisBackend', 'WandbVisBackend', 'TensorboardVisBackend', 'MLflowVisBackend', 'ClearMLVisBackend']
    
    
    VISUALIZERS count result:
    num_modules:	1
    scope:	mmengine
    mmengine/visualization:	['Visualizer']
    
    
    WEIGHT_INITIALIZERS count result:
    num_modules:	8
    scope:	mmengine
    mmengine/model:	['Constant', 'Xavier', 'Normal', 'TruncNormal', 'Uniform', 'Kaiming', 'Caffe2Xavier', 'Pretrained']
    """


def build_model_from_cfg_learn():
    """
    build_model_from_cfg用于构建一系列模块，与build_from_cfg不同，其参数可以为一个列表
    如传入cfg为一个包含字典得列表，则会创建一个Sequential
    """
    resnet = dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'))
    from mmdet.registry import MODELS as MMDetMODELS
    import mmdet.models
    models = build_model_from_cfg([resnet, resnet], MMDetMODELS)
    # model = build_from_cfg([resnet, resnet], MMDetMODELS) # build_from_cfg不支持这个
    print(models)

def build_runner_from_cfg_learn():
    """
    build_runner_from_cfg被用于RUNNERS注册器的build函数
    """
    MYRUNNERS = Registry('myrunners', build_func=build_runner_from_cfg)

    @MYRUNNERS.register_module()
    class MyRunner():
        @classmethod
        def from_cfg(cls, cfg):
            print(cfg)

    runner = MYRUNNERS.build(dict(runner_type='MyRunner', peseduo_param=1))
    print(runner)


def build_from_cfg_learn():
    """
    build_from_cfg被用来根据参数构建相关模型，不过还是推荐用MODELS.build方法来构建。
    其实，此函数在注册器中被调用到，和Registry.build意思差不多。
    """
    resnet = dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'))
    from mmdet.registry import MODELS as MMDetMODELS
    import mmdet.models
    model = build_from_cfg(resnet, MMDetMODELS)
    print(model)


def build_scheduler_from_cfg_learn():
    """
    由于模型训练过程中，梯度优化器的参数已经确定了。为了进一步调节学习率等超参数，需要对optimizer进行动态修改。
    """
    import mmengine.optim
    from torch.optim import SGD
    import mmdet.models
    from mmdet.registry import MODELS as MMDetMODELS

    resnet = dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'))

    scheduler_cfg = dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500)

    # 首先构建模型和对应的梯度优化器
    model = build_model_from_cfg(resnet, MMDetMODELS)
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 通过此函数构建学习率调度器必须在cfg中包含optimizer参数，从而传递给注册器找到的类对象
    scheduler_cfg['optimizer'] = optimizer
    paramschduler = build_scheduler_from_cfg(scheduler_cfg, PARAM_SCHEDULERS)
    print(paramschduler)


def init_default_scope_learn():
    """
    初始化相关任务的作用域，例如mmengine，mmdet，mmseg等等(注意，其参数可以是任意的的）
    """
    init_default_scope('mytask')
    scope_name = DefaultScope.get_instance('mytask').scope_name
    print(scope_name)


def code_generator():
    names = ['WEIGHT_INITIALIZERS',
             'OPTIMIZERS', 'OPTIM_WRAPPER_CONSTRUCTORS', 'TASK_UTILS',
             'PARAM_SCHEDULERS', 'METRICS', 'MODEL_WRAPPERS', 'OPTIM_WRAPPERS', 'LOOPS',
             'VISBACKENDS', 'VISUALIZERS', 'LOG_PROCESSORS', 'EVALUATOR', 'INFERENCERS',
             'DefaultScope', 'traverse_registry_tree', 'count_registered_modules',
             'build_model_from_cfg', 'build_runner_from_cfg', 'build_from_cfg',
             'build_scheduler_from_cfg', 'init_default_scope', 'FUNCTIONS', 'STRATEGIES']
    code_strs = []
    for name in names:
        lines = [f'def {name}_learn():', f'\tMy{name} = Registry(\'my {name[:-1].lower()}\')',
                 f'\t@My{name}.register_module()', f'\tclass My{name[0] + name[1:-1].lower()}():'
                                                   '\n\t\tpass',
                 f'\tmy{name.lower()[:-1]} = My{name}.build(dict(type=\'My{name[0] + name[1:-1].lower()}\'))',
                 f'\tprint(my{name.lower()[:-1]})']
        print('\n'.join(lines))
        print('\n')


if __name__ == '__main__':
    build_scheduler_from_cfg_learn()
