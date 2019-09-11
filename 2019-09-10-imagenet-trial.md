## 利用现有的网络对imagenet进行一次训练

代码示例：

https://github.com/rwightman/pytorch-image-models

其中，以effcientnet_b0, 跑imagenet

## 实验过程

Use the `--model` arg to specify model for train, validation, inference scripts. Match the all lowercase creation fn for the model you'd like.

我们用 `Generic EfficientNet (from my standalone GenMobileNet) - A generic model that implements many of the efficient models that utilize similar DepthwiseSeparable and InvertedResidual blocks` 里的`EfficientNet B0 `

### Environment

workon pyenv1在虚拟环境下`pyenv1`下运行:

`All development and testing has been done in Conda Python 3 environments on Linux x86-64 systems, specifically Python 3.6.x and 3.7.x.` 需要手动升级python至python3.7并将python指向python3.7。注意在虚拟环境中程序是在/.virtualenvs/pyenv1进行的:

```shell
(torch-env) (pyenv1) wangguangrun@AMAX:~$ which python
/home/wangguangrun/.virtualenvs/pyenv1/bin/python
(torch-env) (pyenv1) wangguangrun@AMAX:~$ which pip
/home/wangguangrun/.virtualenvs/pyenv1/bin/pip
```

`PyTorch versions 1.0 and 1.1 have been tested with this code.`

以上环境在pyenv1中虚拟环境`torch-env`安装

```
conda create -n torch-env
conda activate torch-env
conda install -c pytorch pytorch torchvision cudatoolkit=10.0
```

然后要pip install timm来调用模型

安装NVIDIA APEX：

```
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
```

Train

Exp. To `train` an `SE-ResNet34` on ImageNet, locally distributed, `4 GPUs`, one process per GPU w/ cosine schedule, random-erasing prob of 50% and per-pixel random value:

```
./distributed_train.sh 4 /data/imagenet --model seresnet34 --sched cosine --epochs 150 --warmup-epochs 5 --lr 0.4 --reprob 0.5 --remode pixel --batch-size 256 -j 4
```

先看下代码了解路径怎么传：

```python
    train_dir = os.path.join(args.data, 'train')
    if not os.path.exists(train_dir):
        logging.error('Training folder does not exist at: {}'.format(train_dir))
        exit(1)
    dataset_train = Dataset(train_dir)
```

其中args.data就是传入路径，这句话就是说文件在传入路径/train。那么如何处理文件需要看timm.Dateset()函数：此处folder就是train_dir

```python
def find_images_and_targets(folder, types=IMG_EXTENSIONS, class_to_idx=None, leaf_name_only=True, sort=True):
    if class_to_idx is None:
        class_to_idx = dict()
        build_class_idx = True
    else:
        build_class_idx = False
    labels = []
    filenames = []
    # 遍历文件夹
    for root, subdirs, files in os.walk(folder, topdown=False):
        # 判断是否为相对地址并改为相对地址
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
        if build_class_idx and not subdirs:
            class_to_idx[label] = None
        # 对于该文件夹下的图片文件
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types:
                filenames.append(os.path.join(root, f))
                labels.append(label)
    if build_class_idx:
        classes = sorted(class_to_idx.keys(), key=natural_key)
        for idx, c in enumerate(classes):
            class_to_idx[c] = idx
    images_and_targets = zip(filenames, [class_to_idx[l] for l in labels])
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    if build_class_idx:
        return images_and_targets, classes, class_to_idx
    else:
        return images_and_targets
```

