

![image](DFB.png)


# DFB

This is the official implementation of the paper: DFB: A Data-Free, Low-Budget, and High-Efficacy Clean-Label Backdoor Attack. The POOD dataset can be downloaded from the following link: https://pan.baidu.com/s/1g65ja7q2xIhLqi7K6YLB9g?pwd=r8da
Extraction code: r8da


## Requirements
####
- Python >= 3.6
- PyTorch >= 1.10.1
- TorchVisison >= 0.11.2
- OpenCV >= 4.5.3


# Get Started Quickly
Please run the training process for the decoder
```python
python decode_train.py
```

Please run the training process for the encoder

```python
python encode_train.py
```

Conducting an Assessment on the Impact of an Attack with a Fixed Trigger
```python
python test_exp.py
```

Conducting an Assessment on the Impact of an Attack with the encoder
```python
python test_exp_encode.py
```


