# STDP-GCN
Automatic Sleep Staging Using Adaptive Graph Convolutional Networks With STDP Mechanism
![model_architecture](stdpgcn.svg)
## How to run
Data can be downloaded by *get_ISRUC_S3_win.sh*
### Preprocess
Thanks for Jiaziyu's data preprocess files which is a great contribution to this work.
```
python isruc_s3_preprocess.py
```
### Train Model
```
python train.py
```
