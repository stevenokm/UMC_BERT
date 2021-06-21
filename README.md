# UMC BERT Repo
BERT models' repo for UMC NLP researches

## How to use?
* You should git clone this repository first.
* And then cd into the folder, type as follows according to your cuda version and torch version.
```
sudo docker build --no-cache -f Dockerfile-py36-torch17-cuda102 -t pytorch_env/pytorch:1.7_py36_cu102 .
```
* For the pytorch headless script, you can use as the hereunder part.
```
sudo docker run --runtime=nvidia --ipc host --rm -it -v "$PWD":/home/workspace -w /home/workspace pytorch_env/pytorch:1.7_py36_cu102 python3 Albert_SOP_SUBMIT.py
# for Docker >= 19.03
sudo docker run --gpus all --ipc host --rm -it -v "$PWD":/home/workspace -w /home/workspace pytorch_env/pytorch:1.7_py36_cu102 python3 Albert_SOP_SUBMIT.py
```
* The pytorch jupyter notebook file `Albert_SOP_SUBMIT.ipynb` could be viewed under any jupyter notebook environment, but run under docker environment needs port forwording configurations on `dockers run`

## Pre-training data preparation
* The pre-training data should follow following folder structure
```
folder/
├── file_split_00
├── file_split_01
├── file_split_02
...
└── file_split_last
```
* each file should follw following file format, and the inputs are readed line by line
``` html
<doc id=SOME_IDs>
some plain texts line 1
some plain texts line 2
some plain texts line 3
....
some plain texts line last
</doc>
```
* Note: the OSCAR corpus is used for test-run, users can replace any corpus follow the format above
