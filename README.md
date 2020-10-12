[<img src="figure/DramaQA_logo2.png" width="300"/>](https://github.com/liveseongho/DramaQA/)

[![Last commit](https://img.shields.io/github/last-commit/liveseongho/DramaQA)](https://github.com/liveseongho/DramaQA/commits/main)
[![DramaQA Homepage](https://img.shields.io/badge/homepage-dramaqa.snu.ac.kr-blue)](https://dramaqa.snu.ac.kr)
[![DramaQA Dataset](https://img.shields.io/badge/dataset-download-blue)](https://dramaqa.snu.ac.kr/Download)
[![GitHub stars](https://img.shields.io/github/stars/liveseongho/DramaQA)](https://github.com/liveseongho/DramaQA/stargazers)
[![Github forks](https://img.shields.io/github/forks/liveseongho/DramaQA)](https://github.com/liveseongho/DramaQA/network/members)
[![GitHub issues](https://img.shields.io/github/issues/liveseongho/DramaQA)](https://github.com/liveseongho/DramaQA/issues)
[![LICENSE](https://img.shields.io/github/license/liveseongho/DramaQA)](https://github.com/liveseongho/DramaQA/blob/main/LICENSE)

#### DramaQA dataset is currently in progress, and this repository will also be updated continuously.

DramaQA dataset is a large-scale video QA task based on a Korean popular TV show, `Another Miss Oh`. This dataset contains four levels of QA on difficulty and character-centered video annotations. We are expecting this dataset could be a starting point to evaluate human level video story understanding. Please refer more detailed information on [DramaQA homepage](https://dramaqa.snu.ac.kr).

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [DramaQA starter code](#DramaQA)
	* [Requirements](#requirements)
	* [Directory Structure](#directory-structure)
	* [Usage](#usage)
		* [Resuming from checkpoints](#resuming-from-checkpoints)
    * [Using Multiple GPU](#using-multiple-gpu)
	* [Customization](#customization)
		* [Custom CLI options](#custom-cli-options)
		* [Additional logging](#additional-logging)
		* [Checkpoints](#checkpoints)
    * [Tensorboard Visualization](#tensorboard-visualization)
	* [TODOs](#todos)
	* [License](#license)
	* [Contact information](#contact-information)
	* [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Requirements
* Python >= 3.6 (3.6 recommended)
* PyTorch >= 1.4.0 (1.4.0 recommended)
* tensorboard >= 1.14 (see [Tensorboard Visualization](#tensorboard-visualization))

## Directory Structure
  ```
  DramaQA/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  │
  ├── config.json - holds configuration for training
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── data_loaders.py
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │
  └── utils/ - small utility functions
      ├── util.py
      └── ...


  data/AnotherMissOh/ - default directory for storing dataset
  ├── AnotherMissOh_images/
  ├── AnotherMissOh_QA/
  │
  ├── AnotherMissOh_visual.json
  └── AnotherMissOh_script.json

  results/
  ├── models/ - trained models are saved here
  └── log/ - default logdir for tensorboard and logging output
  ```

## Usage
* Clone this repo `git clone https://github.com/liveseongho/DramaQA`.
* Download DramaQA dataset [here](https://dramaqa.snu.ac.kr/Download) and make directory structure like [this](#directory-structure).
* Try `python train.py -c config.json` to run code. You need to install [requirements](#requirements).


### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint
  ```

### Using Multiple GPU
You can enable multi-GPU training by setting `n_gpu` argument of the config file to larger number.
If configured to use smaller number of gpu than available, first n devices will be used by default.
Specify indices of available GPUs by cuda environmental variable.
  ```
  python train.py --device 2,3 -c config.json
  ```
  This is equivalent to
  ```
  CUDA_VISIBLE_DEVICES=2,3 python train.py -c config.py
  ```

## Customization

### Custom CLI options

This template uses the configurations stored in the json file by default, but by registering custom options as follows
you can change some of them using CLI flags.

  ```python
  # simple class-like object having 3 attributes, `flags`, `type`, `target`.
  CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
  options = [
      CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
      CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
      # options added here can be modified by command line flags.
  ]
  ```
`target` argument should be sequence of keys, which are used to access that option in the config dict. In this example, `target`
for the learning rate option is `('optimizer', 'args', 'lr')` because `config['optimizer']['args']['lr']` points to the learning rate.
`python train.py -c config.json --bs 256` runs training with options given in `config.json` except for the `batch size`
which is increased to 256 by command line options.



### Additional logging
If you have additional information to be logged, in `_train_epoch()` of your trainer class, merge them with `log` as shown below before returning:

  ```python
  additional_log = {"gradient_norm": g, "sensitivity": s}
  log.update(additional_log)
  return log
  ```

### Testing
You can test trained model by running `test.py` passing path to the trained checkpoint by `--resume` argument.

### Checkpoints
You can specify the name of the training session in config files:
  ```json
  "name": "MNIST_LeNet",
  ```

The checkpoints will be saved in `save_dir/name/timestamp/checkpoint_epoch_n`, with timestamp in mmdd_HHMMSS format.

A copy of config file will be saved in the same folder.

**Note**: checkpoints contain:
  ```python
  {
    'arch': arch,
    'epoch': epoch,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.mnt_best,
    'config': self.config
  }
  ```

### Tensorboard Visualization
This template supports Tensorboard visualization by using either  `torch.utils.tensorboard` or [TensorboardX](https://github.com/lanpa/tensorboardX).

1. **Install**

    If you are using pytorch 1.4.0 or higher, install tensorboard by 'pip install tensorboard>=1.14.0'.

    Otherwise, you should install tensorboardx. Follow installation guide in [TensorboardX](https://github.com/lanpa/tensorboardX).

2. **Run training**

    Make sure that `tensorboard` option in the config file is turned on.

    ```
     "tensorboard" : true
    ```

3. **Open Tensorboard server**

    Type `tensorboard --logdir saved/log/` at the project root, then server will open at `http://localhost:6006`

By default, values of loss and metrics specified in config file, input images, and histogram of model parameters will be logged.
If you need more visualizations, use `add_scalar('tag', data)`, `add_image('tag', image)`, etc in the `trainer._train_epoch` method.
`add_something()` methods in this template are basically wrappers for those of `tensorboardX.SummaryWriter` and `torch.utils.tensorboard.SummaryWriter` modules.

**Note**: You don't have to specify current steps, since `WriterTensorboard` class defined at `logger/visualization.py` will track current steps.


## TODOs

- [ ] Load specific datasets for model inputs
- [x] Add BERT tokenizer

## License
This project is licensed under the MIT License. See [LICENSE](https://github.com/liveseongho/DramaQA/blob/main/LICENSE) for more details.

## Contact information
For help or issues using DramaQA starter code, please submit a [GitHub issue](https://github.com/liveseongho/DramaQA/issues/new).

 Please feel free to contact official e-mail (dramaqa.challenge@gmail.com) if you have any questions about DramaQA challenge and dataset download. For personal communication related to DramaQA, please contact Seongho Choi (shchoi@bi.snu.ac.kr).

## Acknowledgements
This work was partly supported by the Institute for Information & Communications Technology Promotion (2015-0-00310-SW.StarLab, 2017-0-01772-VTT, 2018-0-00622-RMI, 2019-0-01367-BabyMind) and Korea Institute for Advancement Technology (P0006720-GENKO) grant funded by the Korea government.