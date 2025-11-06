# Big Vision

这个代码库旨在用于在
[Cloud TPU VMs](https://cloud.google.com/blog/products/compute/introducing-cloud-tpu-vms)或 GPU 机器 上训练大规模视觉模型。它基于 [Jax](https://github.com/jax-ml/jax)/[Flax](https://github.com/google/flax)
库构建,并使用 [tf.data](https://www.tensorflow.org/guide/data) 和
[TensorFlow Datasets](https://www.tensorflow.org/datasets) 来实现可扩展且可复现的数据输入管线。

开源该代码库的主要目的有两个：
1.发布在此代码库中开发的研究项目代码（见下方列表）；

2.为在 GPU 机器和 Google Cloud TPU 上运行大规模视觉实验提供强大的起点，
并确保这些实验能够无缝扩展，从单个 TPU 核心到最多 2048 个 TPU 核心的分布式环境，实现开箱即用（out-of-the-box） 的高性能训练。

`big_vision` 的目标是支持 Google 内部的研究项目. 我们一般不会响应功能请求或接受外部贡献，除非这些修改已经事先获得批准（请先在 issue 中提出申请）。
如果你需要一个专注于模型迁移（transfer）且维护良好的代码库，可以参考 [vision_transformer](https://github.com/google-research/vision_transformer).

请注意， `big_vision` 是一个高度动态的代码库。虽然我们会尽力确保核心代码始终可用、功能完整，但无法保证位于 `.../proj/...` 子文件夹中的项目代码能得到及时更新。
不过，我们提供了一张 [table](#project-specific-commits) 其中列出了各个项目最近一次确认可正常运行的提交版本。

以下研究项目最初都是在 `big_vision`代码库中进行的：


### Architecture research

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929), by
  Alexey Dosovitskiy*, Lucas Beyer*, Alexander Kolesnikov*, Dirk Weissenborn*,
  Xiaohua Zhai*, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer,
  Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby*
- [Scaling Vision Transformers](https://arxiv.org/abs/2106.04560), by
  Xiaohua Zhai*, Alexander Kolesnikov*, Neil Houlsby, and Lucas Beyer*\
  相关资源(配置文件): [config](big_vision/configs/proj/scaling_laws/train_vit_g.py).
- [How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers](https://arxiv.org/abs/2106.10270), by
  Andreas Steiner*, Alexander Kolesnikov*, Xiaohua Zhai*, Ross Wightman,
  Jakob Uszkoreit, and Lucas Beyer*
- [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601), by
  Ilya Tolstikhin*, Neil Houlsby*, Alexander Kolesnikov*, Lucas Beyer*,
  Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner,
  Daniel Keysers, Jakob Uszkoreit, Mario Lucic, Alexey Dosovitskiy\
  相关资源(配置文件): [config](big_vision/configs/mlp_mixer_i1k.py).
- [Better plain ViT baselines for ImageNet-1k](https://arxiv.org/abs/2205.01580), by
  Lucas Beyer, Xiaohua Zhai, Alexander Kolesnikov\
  相关资源(配置文件): [config](big_vision/configs/vit_s16_i1k.py)
- [UViM: A Unified Modeling Approach for Vision with Learned Guiding Codes](https://arxiv.org/abs/2205.10337), by
  Alexander Kolesnikov^*, André Susano Pinto^*, Lucas Beyer*, Xiaohua Zhai*, Jeremiah Harmsen*, Neil Houlsby*\
  相关资源(配置文件): [readme](big_vision/configs/proj/uvim/README.md), [configs](big_vision/configs/proj/uvim), [colabs](big_vision/configs/proj/uvim).
- [FlexiViT: One Model for All Patch Sizes](https://arxiv.org/abs/2212.08013), by
  Lucas Beyer*, Pavel Izmailov*, Alexander Kolesnikov*, Mathilde Caron*, Simon
  Kornblith*, Xiaohua Zhai*, Matthias Minderer*, Michael Tschannen*, Ibrahim
  Alabdulmohsin*, Filip Pavetic*\
  相关资源(配置文件): [readme](big_vision/configs/proj/flexivit/README.md), [configs](big_vision/configs/proj/flexivit).
- [Dual PatchNorm](https://arxiv.org/abs/2302.01327), by Manoj Kumar, Mostafa Dehghani, Neil Houlsby.
- [Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design](https://arxiv.org/abs/2305.13035), by
  Ibrahim Alabdulmohsin*, Xiaohua Zhai*, Alexander Kolesnikov, Lucas Beyer*.
- (partial) [Scaling Vision Transformers to 22 Billion Parameters](https://arxiv.org/abs/2302.05442), by
  Mostafa Dehghani*, Josip Djolonga*, Basil Mustafa*, Piotr Padlewski*, Jonathan Heek*, *wow many middle authors*, Neil Houlsby*.
- (partial) [Finite Scalar Quantization: VQ-VAE Made Simple](https://arxiv.org/abs/2309.15505), by
  Fabian Mentzer, David Minnen, Eirikur Agustsson, Michael Tschannen.
- [GIVT: Generative Infinite-Vocabulary Transformers](https://arxiv.org/abs/2312.02116), by
  Michael Tschannen, Cian Eastwood, Fabian Mentzer.\
   相关资源(配置文件): [readme](big_vision/configs/proj/givt/README.md), [config](big_vision/configs/proj/givt/givt_imagenet2012.py), [colab](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/givt/givt_demo_colab.ipynb).
- [Unified Auto-Encoding with Masked Diffusion](https://arxiv.org/abs/2406.17688), by
  Philippe Hansen-Estruch, Sriram Vishwanath, Amy Zhang, Manan Tomar.
- [Jet: A Modern Transformer-Based Normalizing Flow](https://arxiv.org/abs/2412.15129), by
  Alexander Kolesnikov*, André Susano Pinto*, Michael Tschannen*, [configs](big_vision/configs/proj/jet)
- [JetFormer: An autoregressive generative model of raw images and text](https://arxiv.org/abs/2411.19722), by
  Michael Tschannen*, André Susano Pinto*, Alexander Kolesnikov*. [configs](big_vision/configs/proj/jetformer).


### Multimodal research

- [LiT: Zero-Shot Transfer with Locked-image Text Tuning](https://arxiv.org/abs/2111.07991), by
  Xiaohua Zhai*, Xiao Wang*, Basil Mustafa*, Andreas Steiner*, Daniel Keysers,
  Alexander Kolesnikov, and Lucas Beyer*\
  Resources: [trainer](big_vision/trainers/proj/image_text/contrastive.py), [config](big_vision/configs/proj/image_text/lit_coco.py), [colab](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/lit.ipynb).
- [CLIPPO: Image-and-Language Understanding from Pixels Only](https://arxiv.org/abs/2212.08045), by
  Michael Tschannen, Basil Mustafa, Neil Houlsby\
  Resources: [readme](big_vision/configs/proj/clippo/README.md), [config](big_vision/configs/proj/clippo/train_clippo.py), [colab](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/clippo/clippo_colab.ipynb).
- [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343), by
  Xiaohua Zhai*, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer*\
  Resources: [colab and models](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/SigLIP_demo.ipynb), code TODO.
- [A Study of Autoregressive Decoders for Multi-Tasking in Computer Vision](https://arxiv.org/abs/2303.17376), by
  Lucas Beyer*, Bo Wan*, Gagan Madan*, Filip Pavetic*, Andreas Steiner*, Alexander Kolesnikov, André Susano Pinto, Emanuele Bugliarello, Xiao Wang, Qihang Yu, Liang-Chieh Chen, Xiaohua Zhai*.
- [Image Captioners Are Scalable Vision Learners Too](https://arxiv.org/abs/2306.07915), by
  Michael Tschannen*, Manoj Kumar*, Andreas Steiner*, Xiaohua Zhai, Neil Houlsby, Lucas Beyer*.\
  Resources: [readme](big_vision/configs/proj/cappa/README.md), [config](big_vision/configs/proj/cappa/pretrain.py), [model](big_vision/models/proj/cappa/cappa.py).
- [Three Towers: Flexible Contrastive Learning with Pretrained Image Models](https://arxiv.org/abs/2305.16999), by Jannik Kossen, Mark Collier, Basil Mustafa, Xiao Wang, Xiaohua Zhai, Lucas Beyer, Andreas Steiner, Jesse Berent, Rodolphe Jenatton, Efi Kokiopoulou.
- (partial) [PaLI: A Jointly-Scaled Multilingual Language-Image Model](https://arxiv.org/abs/2209.06794), by Xi Chen, Xiao Wang, Soravit Changpinyo, *wow so many middle authors*, Anelia Angelova, Xiaohua Zhai, Neil Houlsby, Radu Soricut.
- (partial) [PaLI-3 Vision Language Models: Smaller, Faster, Stronger](https://arxiv.org/abs/2310.09199), by Xi Chen, Xiao Wang, Lucas Beyer, Alexander Kolesnikov, Jialin Wu, Paul Voigtlaender, Basil Mustafa, Sebastian Goodman, Ibrahim Alabdulmohsin, Piotr Padlewski, Daniel Salz, Xi Xiong, Daniel Vlasic, Filip Pavetic, Keran Rong, Tianli Yu, Daniel Keysers, Xiaohua Zhai, Radu Soricut.
- [LocCa](https://arxiv.org/abs/2403.19596), by
  Bo Wan, Michael Tschannen, Yongqin Xian, Filip Pavetic, Ibrahim Alabdulmohsin, Xiao Wang, André Susano Pinto, Andreas Steiner, Lucas Beyer, Xiaohua Zhai.
- [PaliGemma](https://arxiv.org/abs/2407.07726),
  [PaliGemma 2](https://arxiv.org/abs/2412.03555), by *wow many authors*.\
- Resources: [readme](big_vision/configs/proj/paligemma/README.md),
    [model](big_vision/models/proj/paligemma/paligemma.py),
    [transfer configs](big_vision/configs/proj/paligemma/transfers),
    [datasets](big_vision/datasets),
    [CountBenchQA](big_vision/datasets/countbenchqa/data/countbench_paired_questions.json).
- [SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features](https://arxiv.org/abs/2502.14786), by *wow many authors*.\
  Resources: [readme (with checkpoints)](big_vision/configs/proj/image_text/README_siglip2.md), [colab](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/SigLIP2_demo.ipynb).

### Training

- [Knowledge distillation: A good teacher is patient and consistent](https://arxiv.org/abs/2106.05237), by
  Lucas Beyer*, Xiaohua Zhai*, Amélie Royer*, Larisa Markeeva*, Rohan Anil,
  and Alexander Kolesnikov*\
  Resources: [README](big_vision/configs/proj/distill/README.md), [trainer](big_vision/trainers/proj/distill/distill.py), [colab](https://colab.research.google.com/drive/1nMykzUzsfQ_uAxfj3k35DYsATnG_knPl?usp=sharing).
- [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412), by
  Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur
- [Surrogate Gap Minimization Improves Sharpness-Aware Training](https://arxiv.org/abs/2203.08065), by Juntang Zhuang, Boqing Gong, Liangzhe Yuan, Yin Cui, Hartwig Adam, Nicha Dvornek, Sekhar Tatikonda, James Duncan and Ting Liu \
  Resources: [trainer](big_vision/trainers/proj/gsam/gsam.py), [config](big_vision/configs/proj/gsam/vit_i1k_gsam_no_aug.py) [reproduced results](https://github.com/google-research/big_vision/pull/8#pullrequestreview-1078557411)
- [Tuning computer vision models with task rewards](https://arxiv.org/abs/2302.08242), by
  André Susano Pinto*, Alexander Kolesnikov*, Yuge Shi, Lucas Beyer, Xiaohua Zhai.
- (partial) [VeLO: Training Versatile Learned Optimizers by Scaling Up](https://arxiv.org/abs/2211.09760) by
  Luke Metz, James Harrison, C. Daniel Freeman, Amil Merchant, Lucas Beyer, James Bradbury, Naman Agrawal, Ben Poole, Igor Mordatch, Adam Roberts, Jascha Sohl-Dickstein.

### Misc

**杂项研究**

- [Are we done with ImageNet?](https://arxiv.org/abs/2006.07159), by
  Lucas Beyer*, Olivier J. Hénaff*, Alexander Kolesnikov*, Xiaohua Zhai*, Aäron van den Oord*.
- [No Filter: Cultural and Socioeconomic Diversity in Contrastive Vision-Language Models](https://arxiv.org/abs/2405.13777), by
  Angéline Pouget, Lucas Beyer, Emanuele Bugliarello, Xiao Wang, Andreas Peter Steiner, Xiaohua Zhai, Ibrahim Alabdulmohsin.

# 代码库的高层组织与核心原则（简述）

主要的入口点是一个 训练器模块（trainer module），它通常负责所有与训练流程相关的基础工作，包括：

创建模型与优化器；

加载数据；

保存与恢复检查点（checkpoint）；

在循环中执行模型的训练与评估。

我们在项目根目录中提供了一个标准训练器文件 `train.py`， 通常，  `big_vision` 中的各个独立项目都会基于它进行 分支（fork）和定制。

所有的 模型（models）、评估器（evaluators） 和 预处理操作（preprocessing operations）
都位于各自对应的子目录中，并且通常可以在不同项目之间复用。
我们鼓励这些目录下的模块使用统一的 API 接口以提高可复用性，
但这并不是强制要求的 —— 因为某些项目可能需要引入自定义 API。

我们拥有一个功能强大的配置系统（configuration system）, 所有配置文件都位于
`configs/` 目录中. 自定义的训练器或模块可以直接扩展或修改这些配置选项，
以实现不同项目的灵活适配。

项目特定的代码位于 `.../proj/...` 命名空间中。由于各个项目之间存在差异，无法保证项目特定代码始终与核心 `big_vision` 库保持同步。
因此，我们在下方提供了每个项目的 [last known commit](#project-specific-commits)
用户可以参考该版本以确保项目代码能正常工作。

训练任务具有 容错与恢复功能：
如果训练被中断，只要用户提供正确的 `--workdir` 路径，
系统就能无缝地从上次保存的检查点（checkpoint）恢复训练。

每个配置文件（configuration file）顶部都会包含一段注释，其中有：

一个可直接运行的 `COMMAND` 命令示例；

以及该配置对应的 预期运行时间 和 实验结果提示。

通常情况下：

在 GPU 机器上运行时，可以使用命令：
`python -m COMMAND`


在 TPU（包括多主机）上运行时，使用命令：
```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all
  --command "bash big_vision/run_tpu.sh COMMAND"
```

有关如何在 GPU 机器或 Google Cloud TPU 上运行 `big_vision` 代码的更多细节，
请参考下方的运行说明。

默认情况下，系统会生成 检查点（checkpoints） 和 日志文件（logfiles）。
日志文件以 JSON 对象列表 的形式保存，
我们还提供了一个简单的Colab示例 [example colab to read
and display the logs and checkpoints](https://colab.research.google.com/drive/1R_lvV542WUp8Q2y8sbyooZOGCplkn7KI?usp=sharing)演示如何读取和可视化这些日志与检查点。

# 当前与未来内容

当前版本内容

首个版本主要包含以下核心功能：
✅ 在 Cloud TPU VM 上进行大规模分类模型的预训练、迁移学习与评估，
即支持完整的端到端训练、微调与验证流程。

后续新增的主要功能与项目

我们随后在 big_vision 框架中新增了以下关键特性与研究项目：

🧩 图文对比模型的训练与评估（如 LiT 与 CLIP）；

🔁 稳定一致的知识蒸馏（distillation）方法；

⚙️ ViT 的可扩展性研究（Scaling ViT）；

🧠 MLP-Mixer 架构支持；

🧩 UViM（Unified Vision Modeling）统一视觉建模框架。

即将发布的功能与项目（计划中）

我们计划在未来版本中陆续开放以下功能（顺序不分先后）：

📦 将 ImageNet-21k 数据集集成至 TFDS（TensorFlow Datasets）；

🔍 支持加载我们论文中使用的其他公共模型（如 NFNet、MoCov3、DINO）；

🧮 高效内存的 Polyak 平均实现（Memory-efficient Polyak averaging）；

⚡ 高级 JAX 计算与内存分析工具：
当前我们在内部使用专用分析工具，未来计划支持公开版本的性能分析接口。

未来展望

我们将持续在此仓库中发布基于 `big_vision` 框架的最新研究代码，
以便研究人员复现和扩展我们的最新论文成果。

### 非公开内容（Non-content）

以下内容仅存在于 `big_vision` 内部版本中，
目前没有公开发布计划：

🧪 质量与速度的常规回归测试：
这些测试依赖于 Google 的内部基础设施，因此无法在开源版本中提供。

📊 高级实验日志记录、监控与可视化系统：
同样依赖内部工具链。
不过我们对改进这一部分保持开放态度，
如果未来有独立且自包含的实现方案，可能会考虑在后续版本中加入。

🧬 尚未公开的研究项目：
仍处于内部开发或论文投稿阶段，因此暂不开放。


# GPU Setup

We first discuss how to setup and run `big_vision` on a (local) GPU machine,
and then discuss the setup for Cloud TPUs. Note that data preparation step for
(local) GPU setup can be largely reused for the Cloud TPU setup. While the
instructions skip this for brevity, we highly recommend using a
[virtual environment](https://docs.python.org/3/library/venv.html) when
installing python dependencies.

## Setting up python packages

The first step is to checkout `big_vision` and install relevant python
dependencies:

```
git clone https://github.com/google-research/big_vision
cd big_vision/
pip3 install --upgrade pip
pip3 install -r big_vision/requirements.txt
```

The latest version of `jax` library can be fetched as

```
pip3 install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

You may need a different `jax` package, depending on CUDA and cuDNN libraries
installed on your machine. Please consult
[official jax documentation](https://github.com/jax-ml/jax#pip-installation-gpu-cuda)
for more information.

## Preparing tfds data

For unified and reproducible access to standard datasets we opted to use the
`tensorflow_datasets` (`tfds`) library. It requires each dataset to be
downloaded, preprocessed and then to be stored on a hard drive (or, if you use
"Google Cloud", preferably stored in a "GCP bucket".).

Many datasets can be downloaded and preprocessed automatically when used
for the first time. Nevertheless, we intentionally disable this feature and
recommend doing dataset preparation step separately, ahead of the first run. It
will make debugging easier if problems arise and some datasets, like
`imagenet2012`, require manually downloaded data.

Most of the datasets, e.g. `cifar100`, `oxford_iiit_pet` or `imagenet_v2`
can be fully automatically downloaded and prepared by running

```
cd big_vision/
python3 -m big_vision.tools.download_tfds_datasets cifar100 oxford_iiit_pet imagenet_v2
```

A full list of datasets is available at [this link](https://www.tensorflow.org/datasets/catalog/overview#all_datasets).

Some datasets, like `imagenet2012` or `imagenet2012_real`, require the data to
be downloaded manually and placed into `$TFDS_DATA_DIR/downloads/manual/`,
which defaults to `~/tensorflow_datasets/downloads/manual/`. For example, for
`imagenet2012` and `imagenet2012_real` one needs to place the official
`ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar` files in that directory
and then run
`python3 -m big_vision.tools.download_tfds_datasets imagenet2012 imagenet2012_real`
(which may take ~1 hour).

If you use `Google Cloud` and, TPUs in particular, you can then upload
the preprocessed data (stored in `$TFDS_DATA_DIR`) to
"Google Cloud Bucket" and use the bucket on any of your (TPU) virtual
machines to access the data.

## Running on a GPU machine

Finally, after installing all python dependencies and preparing `tfds` data,
the user can run the job using config of their choice, e.g. to train `ViT-S/16`
model on ImageNet data, one should run the following command:

```
python3 -m big_vision.train --config big_vision/configs/vit_s16_i1k.py --workdir workdirs/`date '+%m-%d_%H%M'`
```

or to train MLP-Mixer-B/16, run (note the `gpu8` config param that reduces the default batch size and epoch count):

```
python3 -m big_vision.train --config big_vision/configs/mlp_mixer_i1k.py:gpu8 --workdir workdirs/`date '+%m-%d_%H%M'`
```

# Cloud TPU VM setup

## Create TPU VMs

To create a single machine with 8 TPU cores, follow the following Cloud TPU JAX
document:
https://cloud.google.com/tpu/docs/run-calculation-jax

To support large-scale vision research, more cores with multiple hosts are
recommended. Below we provide instructions on how to do it.

First, create some useful variables, which we be reused:

```
export NAME=<a name of the TPU deployment, e.g. my-tpu-machine>
export ZONE=<GCP geographical zone, e.g. europe-west4-a>
export GS_BUCKET_NAME=<Name of the storage bucket, e.g. my_bucket>
```

The following command line will create TPU VMs with 32 cores,
4 hosts.

```
gcloud compute tpus tpu-vm create $NAME --zone $ZONE --accelerator-type v3-32 --version tpu-ubuntu2204-base
```

## Install `big_vision` on TPU VMs

Fetch the `big_vision` repository, copy it to all TPU VM hosts, and install
dependencies.

```
git clone https://github.com/google-research/big_vision
gcloud compute tpus tpu-vm scp --recurse big_vision/big_vision $NAME: --zone=$ZONE --worker=all
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command "bash big_vision/run_tpu.sh"
```

## Download and prepare TFDS datasets

We recommend preparing `tfds` data locally as described above and then uploading
the data to `Google Cloud` bucket. However, if you prefer, the datasets which
do not require manual downloads can be prepared automatically using a TPU
machine as described below. Note that TPU machines have only 100 GB of disk
space, and multihost TPU slices do not allow for external disks to be attached
in a write mode, so the instructions below may not work for preparing large
datasets. As yet another alternative, we provide instructions
[on how to prepare `tfds` data on CPU-only GCP machine](#preparing-tfds-data-on-a-standalone-gcp-cpu-machine).

Specifically, the seven TFDS datasets used during evaluations will be generated
under `~/tensorflow_datasets` on TPU machine with this command:

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=0 --command "TFDS_DATA_DIR=~/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.tools.download_tfds_datasets cifar10 cifar100 oxford_iiit_pet oxford_flowers102 cars196 dtd uc_merced"
```

You can then copy the datasets to GS bucket, to make them accessible to all TPU workers.

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=0 --command "rm -r ~/tensorflow_datasets/downloads && gsutil cp -r ~/tensorflow_datasets gs://$GS_BUCKET_NAME"
```

If you want to integrate other public or custom datasets, i.e. imagenet2012,
please follow [the official guideline](https://www.tensorflow.org/datasets/catalog/overview).

## Pre-trained models

For the full list of pre-trained models check out the `load` function defined in
the same module as the model code. And for example config on how to use these
models, see `configs/transfer.py`.

## Run the transfer script on TPU VMs

The following command line fine-tunes a pre-trained `vit-i21k-augreg-b/32` model
on `cifar10` dataset.

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command "TFDS_DATA_DIR=gs://$GS_BUCKET_NAME/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train --config big_vision/configs/transfer.py:model=vit-i21k-augreg-b/32,dataset=cifar10,crop=resmall_crop --workdir gs://$GS_BUCKET_NAME/big_vision/workdir/`date '+%m-%d_%H%M'` --config.lr=0.03"
```

## Run the train script on TPU VMs

To train your own big_vision models on a large dataset,
e.g. `imagenet2012` ([prepare the TFDS dataset](https://www.tensorflow.org/datasets/catalog/imagenet2012)),
run the following command line.

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command "TFDS_DATA_DIR=gs://$GS_BUCKET_NAME/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train --config big_vision/configs/bit_i1k.py  --workdir gs://$GS_BUCKET_NAME/big_vision/workdir/`date '+%m-%d_%H%M'`"
```

## FSDP training.

`big_vision` supports flexible parameter and model sharding strategies.
Currently, we support a popular FSDP sharding via a simple config change, see [this config example](big_vision/configs/transfer.py).
For example, to run FSDP finetuning of a pretrained ViT-L model, run the following command (possible adjusting batch size depending on your hardware):

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command "TFDS_DATA_DIR=gs://$GS_BUCKET_NAME/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train --config big_vision/configs/transfer.py:model=vit-i21k-augreg-l/16,dataset=oxford_iiit_pet,crop=resmall_crop,fsdp=True,batch_size=256 --workdir gs://$GS_BUCKET_NAME/big_vision/workdir/`date '+%m-%d_%H%M'` --config.lr=0.03"
```

## Image-text training with SigLIP.

A minimal example that uses public `coco` captions data:

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command "TFDS_DATA_DIR=gs://$GS_BUCKET_NAME/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.trainers.proj.image_text.siglip --config big_vision/configs/proj/image_text/siglip_lit_coco.py --workdir gs://$GS_BUCKET_NAME/big_vision/`date '+%Y-%m-%d_%H%M'`"
```



## Sometimes useful gcloud commands

- Destroy the TPU machines: `gcloud compute tpus tpu-vm delete $NAME --zone $ZONE`
- Remove all big_vision-related folders on all hosts: `gcloud compute tpus tpu-vm ssh $NAME --zone $ZONE --worker=all --command 'rm -rf ~/big_vision ~/bv_venv'`

## Preparing `tfds` data on a standalone GCP CPU machine.

First create a new machine and a disk (feel free to adjust exact machine type and disk settings/capacity):

```
export NAME_CPU_HOST=<A name of a CPU-only machine>
export NAME_DISK=<A name of a disk>
gcloud compute instances create $NAME_CPU_HOST --machine-type c3-standard-22 --zone $ZONE --image-family ubuntu-2204-lts --image-project ubuntu-os-cloud
gcloud compute disks create $NAME_DISK --size 1000GB --zone $ZONE --type pd-balanced
```

Now attach the disk to the newly create machine:

```
gcloud compute instances attach-disk $NAME_CPU_HOST --disk $NAME_DISK --zone $ZONE
```

Next, `ssh` to the machine `gcloud compute ssh $NAME_CPU_HOST --zone=$ZONE` and
[follow instructions to format and mount the disk](https://cloud.google.com/compute/docs/disks/format-mount-disk-linux).
Let's assume it was mounted to `/mnt/disks/tfds`.

Almost there, now clone and set up `big_vision`:

```
gcloud compute ssh $NAME_CPU_HOST --zone=$ZONE --command "git clone https://github.com/google-research/big_vision.git && cd big_vision && sh big_vision/run_tpu.sh"
```

Finally, prepare the dataset (e.g. `coco_captions`) using the utility script and
copy the result to you google cloud bucket:

```
gcloud compute ssh $NAME_CPU_HOST --zone=$ZONE --command "cd big_vision && TFDS_DATA_DIR=/mnt/disks/tfds/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.tools.download_tfds_datasets coco_captions"
gcloud compute ssh $NAME_CPU_HOST --zone=$ZONE --command "rm -rf /mnt/disks/tfds/tensorflow_datasets/downloads && gsutil cp -r /mnt/disks/tfds/tensorflow_datasets gs://$GS_BUCKET_NAME"
```


# ViT baseline

We provide a well-tuned ViT-S/16 baseline in the config file named
`vit_s16_i1k.py`. It achieves 76.5% accuracy on ImageNet validation split in
90 epochs of training, being a strong and simple starting point for research
on the ViT models.

Please see our [arXiv note](https://arxiv.org/abs/2205.01580) for more details
and if this baseline happens to by useful for your research, consider citing

```
@article{vit_baseline,
  url = {https://arxiv.org/abs/2205.01580},
  author = {Beyer, Lucas and Zhai, Xiaohua and Kolesnikov, Alexander},
  title = {Better plain ViT baselines for ImageNet-1k},
  journal={arXiv preprint arXiv:2205.01580},
  year = {2022},
}
```

# Project specific commits

The last known commit where the specific project code is expected to work. The
core code and configs are expected to work at head.

| Project    | Commit                                                                                        |
|------------|-----------------------------------------------------------------------------------------------|
| UViM       | https://github.com/google-research/big_vision/commit/21bd6ebe253f070f584d8b777ad76f4abce51bef |
| image_text | https://github.com/google-research/big_vision/commit/8921d5141504390a8a4f7b2dacb3b3c042237290 |
| distill    | https://github.com/google-research/big_vision/commit/2f3f493af048dbfd97555ff6060f31a0e686f17f |
| GSAM       | WIP                                                                                           |
| CLIPPO     | https://github.com/google-research/big_vision/commit/fd2d3bd2efc9d89ea959f16cd2f58ae8a495cd44 |
| CapPa      | https://github.com/google-research/big_vision/commit/7ace659452dee4b68547575352c022a2eef587a5 |
| GIVT       | https://github.com/google-research/big_vision/commit/0cb70881dd33b3343b769347dc19793c4994b8cb |

# Citing the codebase

If you found this codebase useful for your research, please consider using
the following BibTEX to cite it:

```
@misc{big_vision,
  author = {Beyer, Lucas and Zhai, Xiaohua and Kolesnikov, Alexander},
  title = {Big Vision},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/google-research/big_vision}}
}
```

# Disclaimer

This is not an official Google Product.

# License

Unless explicitly noted otherwise, everything in the big_vision codebase
(including models and colabs) is released under the Apache2 license.
See the LICENSE file for the full license text.
