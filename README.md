# Continuous Flood Monitoring via Efficient Flood Video Segmentation

This repository contains the code and dataset for the paper "Continuous Flood Monitoring via Efficient Flood Video Segmentation".

## Abstract

Continuous flood monitoring serves a critical function in disaster response by rapidly pinpointing evacuation requirements and potential risks. AI technology has revolutionized flood detection, especially through automated semantic segmentation, significantly boosting efficiency. However, the analysis using satellite images or fixed-point images, poses substantial challenges in maintaining consistent and real-time detection. This has sparked a notable surge of interest in flood video segmentation, particularly leveraging data obtained from Unmanned Aerial Vehicles (UAVs). However, the feasibility of analyzing each frame within videos is compromised due to the significant time required for inference. In this paper, we introduce a frame interpolation approach that selects partial frames for inference and utilizes interpolation to generate results for the intermediate frames. To optimize its efficiency, we integrate semi-supervised learning as a solution to address the challenge posed by the scarcity of labeled UAV video datasets featuring instances of flooding. From the results, our approach demonstrates that it can accelerate the process by up to 5.4 times while achieving an increase in accuracy by 4.93%. After incorporating several methods to decrease the inference time, a segmentation frame rate of up to 76.85 frames per second was achieved on a high performance computing cluster.

## Dataset

For more information on the dataset, click [here](dataset/flow/README.md).

## Usage

First, install the dependencies from `Pipfile.lock`:

```bash
mkdir .venv
pipenv sync
```

To train a model with linear segmentation-based frame interpolation using semi-supervised learning, the following command can be used:

```bash
sbatch ./train_flow.sh flow_gan --data.train_w 433 --model.no_warp True --model.feature_based False
```

## Code Sources

This implementation contains code from the following open source repositories:

- [U2PL](https://github.com/Haochen-Wang409/U2PL)
- [semseg](https://github.com/hszhao/semseg)
- [semisup-semseg](https://github.com/sud0301/semisup-semseg)
- [segmenter](https://github.com/rstrudel/segmenter)

The directory `u2pl` is a copy of the `u2pl` module from [U2PL](https://github.com/Haochen-Wang409/U2PL).
Similarly, the directory `segm` is a copy of the `segm` module from [segmenter](https://github.com/rstrudel/segmenter).
Small changes to the code in those modules were done to correctly integrate them into the project. Therefore, they are included in the source code of this project and not as a standard python dependency.