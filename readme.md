# Modifying SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks

<p align="center" float="center">
  <img src="https://github.com/ridgerchu/SpikeGPT/blob/master/static/spikegpt.png"/>
</p>

## Training on Enwik8

1. Download the [enwik8 dataset](https://data.deepai.org/enwik8.zip).
2. Run `train.py`

## Inference with Prompt

You can choose to run inference with either your own customized model or with our pre-trained model. Our pre-trained model is available [here](https://huggingface.co/ridger/SpikeGPT-OpenWebText-216M). This model trained 5B tokens on OpenWebText. 
1. download our pre-trained model, and put it in the root directory of this repo.
2. Modify the  'context' variable in `run.py` to your custom prompt
3. Run `run.py`

## Fine-Tune with NLU tasks
1. run the file in 'NLU' folders
2. change the path in line 17 to the model path


## Citation

```
@article{zhu2023spikegpt,
        title = {SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks},
        author = {Zhu, Rui-Jie and Zhao, Qihang and Li, Guoqi and Eshraghian, Jason K.},
        journal = {arXiv preprint arXiv:2302.13939},
        year    = {2023}
}
```
