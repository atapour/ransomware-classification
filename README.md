# _A King’s Ransom for Encryption_: Ransomware Classification using AugmentedOne-Shot Learning and Bayesian Approximation

Requires an NVIDIA GPU, Python 3, [CUDA CuDNN](https://developer.nvidia.com/cudnn), [PyTorch 1.2](http://pytorch.org), and [OpenCV](http://www.opencv.org).
<br>
Other libraries such as [visdom](https://github.com/facebookresearch/visdom) and [colorama](https://pypi.org/project/colorama/) are also optionally used in the code.

![General Pipeline](https://github.com/atapour/ransomware-classification/blob/master/imgs/architecture.png)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Custom Network Architecture

## Method:

_"Newly emerging variants of ransomware pose an ever-growing threat to computer systems governing every aspect of modern life through the handling and analysis of big data. While various recent security-based approaches have focused on detecting and classifying ransomware at the network or system level, easy-to-use post-infection ransomware classification for the lay user has not been attempted before. In this paper, we investigate the possibility of classifying the ransomware a system is infected with simply based on a screenshot of the splash screen or the ransom note captured using a consumer camera commonly found in any modern mobile device. To train and evaluate our system, we create a sample dataset of the splash screens of 50 well-known ransomware variants. In our dataset, only a single training image is available per ransomware. Instead of creating a large training dataset of ransomware screenshots, we simulate screenshot capture conditions via carefully designed data augmentation techniques, enabling simple and efficient one-shot learning. Moreover, using model uncertainty obtained via Bayesian approximation, we ensure special input cases such as unrelated non-ransomware images and previously-unseen ransomware variants are correctly identified for special handling and not mis-classified. Extensive experimental evaluation demonstrates the efficacy of our work, with accuracy levels of up to 93.6% for ransomware classification."_

[[Atapour-Abarghouei, Bonner and McGough, 2019](https://arxiv.org/pdf/1908.06750.pdf)]

---




---
## Instructions to train the model:

* First and foremost, this repository needs to be cloned:

```
$ git clone https://github.com/atapour/ransomware-classification.git
$ cd ransomware-classification
```

* The second step would be to download the dataset used for the training and evaluation of the model. 
* The script entitled "download_data.sh" will download the data and automatically checks the downloaded file integrity using MD5 checksum. In order to download the dataset, run the following commands:

```
$ chmod +x ./download_data.sh
$ ./download_data.sh
```
---

![](https://github.com/atapour/ransomware-classification/blob/master/imgs/data.png)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Example of the data used to train and evaluate the approach

* The training code utilizes [visdom](https://github.com/facebookresearch/visdom) to display training results and plots, in order to do which simply run `visdom` and then navigate to the URL http://localhost:8097.

* To train the model, run the following command:

```
$ python train.py <experiment_name> --data_root=./dataset/train --aug rotate contrast brightness occlusion regularblur defocusblur motionblur perspective gray colorjitter noise --input_size=256 --arch=inception
```

* All the arguments for the training are passed from the file `train_arguments.py`. Refer to that file for further information.

---
## Instructions to test the model:

* In order to easily test the model, we provide two sets of pre-trained weights `pretrained_weights/densenet201.pth` and `pretrained_weights/AmirNet_DO.pth`, with the former used to produce high accuracy classification results on the positive test set, and the latter used to evaluate model uncertainty estimates.

* To test the approach based on a [densenet201](https://arxiv.org/abs/1608.06993) architecture, pre-trained on [ImageNet](http://www.image-net.org/) and the full augmentation protocol, run the following command:

```
$ python test.py --pos_root=./dataset/positive --test_checkpoint_path=./pretrained_weights/densenet201.pth --input_size=256 --pretrained --arch=densenet201

```
---

* To test the uncertainty estimation capabilities of the approach using our custom architecture based on [Bayesian approximation](https://arxiv.org/pdf/1506.02142.pdf) employing Monte Carlo drop-out, run the following command:

```
$ python uncertainty.py --pos_root=./dataset/positive --neg_root=./dataset/negative --test_checkpoint_path=./pretrained_weights/AmirNet_DO.pth --input_size=128 --arch=AmirNet_DO

```

---

This work is created as part of the project published in the following. The released weights of the models have been re-trained.
## Reference:

[A King's Ransom for Encryption: Ransomware Classification using AugmentedOne-Shot Learning and Bayesian Approximation](https://arxiv.org/pdf/1908.06750.pdf)
(A. Atapour-Abarghouei, S. Bonner and A.S. McGough), under review in IEEE Int. Conf. Big Data, 2019. [[pdf](https://arxiv.org/pdf/1908.06750.pdf)]

```

@article{atapour2019kings,
  title={A Kings Ransom for Encryption: Ransomware Classification using Augmented One-Shot Learning and Bayesian Approximation},
  author={Atapour-Abarghouei, Amir and Bonner, Stephen and McGough, Andrew Stephen},
  journal={arXiv preprint arXiv:1908.06750},
  year={2019}
}

```
---
