FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update -y
RUN apt-get upgrade -y

# install python
RUN apt install software-properties-common -y

# python3 is installed, we want also the python command (for 3rd party tools that might assume python) which python3 
RUN apt install python-is-python3

# install pip
RUN apt-get -y install python3-pip
RUN pip3 install numpy==1.23.5 pandas==1.5.3 torch==1.13.1 scipy==1.10.1 python-dateutil==2.8.1 pytz==2021.1 certifi==2020.12.5 pyyaml==5.4.1 networkx==2.5.1 scikit-learn==0.24.2 scikit-learn==0.24.2 keras  six==1.15.0 theano==1.0.3 psutil==5.8.0 pympler==0.9 tensorflow  tables scikit-optimize==0.8.1 python-telegram-bot==13.5 tqdm==4.64.1 dill==0.3.6
#Installation 
WORKDIR "/root"