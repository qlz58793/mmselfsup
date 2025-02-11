# Rotation Prediction

> [Unsupervised Representation Learning by Predicting Image Rotation](https://arxiv.org/abs/1803.07728)

<!-- [ALGORITHM] -->

## Abstract

Over the last years, deep convolutional neural networks (ConvNets) have transformed the field of computer vision thanks to their unparalleled capacity to learn high level semantic image features. However, in order to successfully learn those features, they usually require massive amounts of manually labeled data, which is both expensive and impractical to scale. Therefore, unsupervised semantic feature learning, i.e., learning without requiring manual annotation effort, is of crucial importance in order to successfully harvest the vast amount of visual data that are available today. In our work we propose to learn image features by training ConvNets to recognize the 2d rotation that is applied to the image that it gets as input. We demonstrate both qualitatively and quantitatively that this apparently simple task actually provides a very powerful supervisory signal for semantic feature learning. We exhaustively evaluate our method in various unsupervised feature learning benchmarks and we exhibit in all of them state-of-the-art performance. Specifically, our results on those benchmarks demonstrate dramatic improvements w.r.t. prior state-of-the-art approaches in unsupervised representation learning and thus significantly close the gap with supervised feature learning.

<div align="center">
<img  src="https://user-images.githubusercontent.com/36138628/149723477-8f63e237-362e-4962-b405-9bab0f579808.png" width="700" />
</div>

## Models and Benchmarks

In this page, we provide benchmarks as much as possible to evaluate our pre-trained models. If not mentioned, all models are pre-trained on ImageNet-1k dataset.

### Classification

The classification benchmarks includes 4 downstream task datasets, **VOC**, **ImageNet**,  **iNaturalist2018** and **Places205**. If not specified, the results are Top-1 (%).

#### VOC SVM / Low-shot SVM

The **Best Layer** indicates that the best results are obtained from which layers feature map. For example, if the **Best Layer** is **feature3**, its best result is obtained from the second stage of ResNet (1 for stem layer, 2-5 for 4 stage layers).

Besides, k=1 to 96 indicates the hyper-parameter of Low-shot SVM.

| Self-Supervised Config                                                                                                                                      | Best Layer | SVM   | k=1   | k=2   | k=4   | k=8   | k=16  | k=32  | k=64  | k=96  |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| [resnet50_8xb16-steplr-70e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py) | feature4   | 67.70 | 20.60 | 24.35 | 31.41 | 39.17 | 46.56 | 53.37 | 59.14 | 62.42 |

#### ImageNet Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [resnet50_mhead_linear-8xb32-steplr-90e_in1k](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/classification/imagenet/resnet50_mhead_linear-8xb32-steplr-90e_in1k.py) for details of config.

| Self-Supervised Config                                                                                                                                      | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- |
| [resnet50_8xb16-steplr-70e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py) | 12.15    | 31.99    | 44.57    | 54.20    | 45.94    |

<table class="docutils">
<thead>
  <tr>
	    <th rowspan="2">Algorithm</th>
	    <th rowspan="2">Backbone</th>
	    <th rowspan="2">Epoch</th>
      <th rowspan="2">Batch Size</th>
      <th colspan="2" align="center">Results (Top-1 %)</th>
      <th colspan="3" align="center">Links</th>
	</tr>
	<tr>
      <th>Linear Eval</th>
      <th>Fine-tuning</th>
      <th>Pretrain</th>
      <th>Linear Eval</th>
      <th>Fine-tuning</th>
	</tr>
  </thead>
  <tbody>
  <tr>
	    <td>Rotation-Pred</td>
	    <td>ResNet50</td>
	    <td>70</td>
      <td>128</td>
      <td>47.0</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k/rotation-pred_resnet50_8xb16-steplr-70e_in1k_20220825-a8bf5f69.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k/rotation-pred_resnet50_8xb16-steplr-70e_in1k_20220805_113136.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb32-steplr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k/resnet50_linear-8xb32-steplr-100e_in1k/resnet50_linear-8xb32-steplr-100e_in1k_20220825-7c6edcb3.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k/resnet50_linear-8xb32-steplr-100e_in1k/resnet50_linear-8xb32-steplr-100e_in1k_20220808_143921.json'>log</a></td>
      <td>/</td>
	</tr>
  </tbody>
</table>

#### Places205 Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [resnet50_mhead_8xb32-steplr-28e_places205.py](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/classification/places205/resnet50_mhead_8xb32-steplr-28e_places205.py) for details of config.

| Self-Supervised Config                                                                                                                                      | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- |
| [resnet50_8xb16-steplr-70e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py) | 18.94    | 34.72    | 44.53    | 46.30    | 44.12    |

#### ImageNet Nearest-Neighbor Classification

The results are obtained from the features after GlobalAveragePooling. Here, k=10 to 200 indicates different number of nearest neighbors.

| Self-Supervised Config                                                                                                                                      | k=10 | k=20 | k=100 | k=200 |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | ---- | ----- | ----- |
| [resnet50_8xb16-steplr-70e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py) | 11.0 | 11.9 | 12.6  | 12.4  |

### Detection

The detection benchmarks includes 2 downstream task datasets, **Pascal VOC 2007 + 2012** and **COCO2017**. This benchmark follows the evluation protocols set up by MoCo.

#### Pascal VOC 2007 + 2012

Please refer to [config](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/mmdetection/voc0712/faster-rcnn_r50-c4_ms-24k_voc0712.py) for details.

| Self-Supervised Config                                                                                                                                      | AP50  |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| [resnet50_8xb16-steplr-70e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py) | 79.67 |

#### COCO2017

Please refer to [config](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/mmdetection/coco/mask-rcnn_r50_fpn_ms-1x_coco.py) for details.

| Self-Supervised Config                                                                                                                                      | mAP(Box) | AP50(Box) | AP75(Box) | mAP(Mask) | AP50(Mask) | AP75(Mask) |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | --------- | --------- | --------- | ---------- | ---------- |
| [resnet50_8xb16-steplr-70e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py) | 37.9     | 56.5      | 41.5      | 34.2      | 53.9       | 36.7       |

### Segmentation

The segmentation benchmarks includes 2 downstream task datasets, **Cityscapes** and **Pascal VOC 2012 + Aug**. It follows the evluation protocols set up by MMSegmentation.

#### Pascal VOC 2012 + Aug

Please refer to [config](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_4xb4-20k_voc12aug-512x512.py) for details.

| Self-Supervised Config                                                                                                                                      | mIOU  |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| [resnet50_8xb16-steplr-70e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py) | 64.31 |

## Citation

```bibtex
@inproceedings{komodakis2018unsupervised,
  title={Unsupervised representation learning by predicting image rotations},
  author={Komodakis, Nikos and Gidaris, Spyros},
  booktitle={ICLR},
  year={2018}
}
```
