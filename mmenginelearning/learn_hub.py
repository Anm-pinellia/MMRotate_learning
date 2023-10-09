"""
本部分内容用于学习mmengine中的hub目录.
hub目录比较简单，这个子包只有两个函数，分别为：
'get_config', 'get_model'
"""

from mmengine.hub import *

def get_config_learn():
    """
    get_config可以从外部相关配置中实例化一个cfg配置。
    该函数实际调用读取的是对应安装的库中的.mim文件夹下的配置。
    在每个对应模型的文件夹下，有着对应的metafile.yml文件，该文件记录了不同配置的模型检测评价精度等信息。
    对应到.mim文件夹下的configs目录中寻找。
    """

    from mmengine.utils import get_installed_path, install_package
    from mmengine.config.utils import (_get_cfg_metainfo,
                                       _get_external_cfg_base_path,
                                       _get_package_and_cfg_path)


    # ::之前表示package名称，之后表示对应模型的路径
    cfg = get_config('mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py', pretrained=True)
    print(cfg)

    # 打印目标package中对应的配置文件的metainfo
    all_experiments_metainfo=_get_cfg_metainfo(
        package_path=r'E:\Program\Anaconda\envs\mmdet-dev\Lib\site-packages\mmdet',
        cfg_path=r'faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py')
    print(all_experiments_metainfo)

def get_model_learn():
    """
    get_model根据给定的外部配置文件实例化一个对象模型，并决定是否加载模型权重。
    其余关键字参数用于MODELS.build中调用。
    此函数中将会调用get_config来读取对应的模型配置文件。
    """
    model = get_model('mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py', pretrained=False)
    print(model)

if __name__ == '__main__':
    get_model_learn()