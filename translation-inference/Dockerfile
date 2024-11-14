#FROM nvidia/cuda:11.6.2-base-ubuntu20.04
FROM nvidia/cuda:11.6.2-devel-ubuntu20.04
MAINTAINER Michal Dalal

# define timezone to avoid build stuck
ENV TZ=Asia/Jerusalem
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir /CODE
ADD . /CODE
RUN mkdir /DATA


RUN apt-get update && apt-get install -y python3.8 python3-pip ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#WORKDIR /CODE

#Install torch
RUN pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0

#Install apex
#RUN pip install packaging
#RUN git clone https://github.com/NVIDIA/apex.git
#WORKDIR apex
#RUN pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
#WORKDIR ../

#Install everything else
#RUN pip install pandas numpy==1.19.5 cython
#RUN pip install transformers fasttext nltk huggingface

# Install Megatron
RUN git clone --depth=1 --branch v2.4 https://github.com/NVIDIA/Megatron-LM.git
WORKDIR Megatron-LM
RUN pip install -e .
WORKDIR ../

# Install fairscale
RUN git clone https://github.com/facebookresearch/fairscale.git
WORKDIR fairscale
# needed when loading MoE checkpoint w/num_experts < num_gpus
RUN git checkout origin/experts_lt_gpus_moe_reload_fix
RUN pip3 install -e .
WORKDIR ../

# for fairseq translation
#RUN pip install fastBPE sacremoses subword_nmt

#Install fairseq nllb branch
RUN git clone https://github.com/facebookresearch/fairseq.git
WORKDIR fairseq
RUN git checkout nllb
RUN pip install -e .
RUN python3 setup.py build_ext --inplace
WORKDIR ../

# Install stopes
RUN git clone https://github.com/facebookresearch/stopes.git
WORKDIR stopes
RUN pip install -e '.[dev]'
WORKDIR ../

# Install pre-commit hooks
RUN pip install pre-commit && pre-commit install
