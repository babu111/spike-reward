# Modifying SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks

## Environment Setup
Pull the docker image fromm [SpikeGPT Container](https://github.com/eddiem3/SpikeGPT-container).

## Pre-training on Enwik8

1. Download the [enwik8 dataset](https://data.deepai.org/enwik8.zip).
2. Run `train.py`

## Reward modeling on hh-rlhf

1. Download the [hh-rlhf dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf).
2. Run `train_reward.py`


## Citation

```
@article{zhu2023spikegpt,
        title = {SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks},
        author = {Zhu, Rui-Jie and Zhao, Qihang and Li, Guoqi and Eshraghian, Jason K.},
        journal = {arXiv preprint arXiv:2302.13939},
        year    = {2023}
}
```
