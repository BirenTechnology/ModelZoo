FROM ubuntu:20.04
SHELL ["/bin/bash", "-c"]
 
WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive
 
# Set timezone
ENV TZ=Asia/Shanghai
RUN ln -sf /usr/share/zoneinfo/$TZ /etc/localtime && \
     echo $TZ > /etc/timezone
 
# Install core packages
RUN apt-get update \
       && apt-get install -y --no-install-recommends \
       dpkg-dev \
              sudo \
              ccache \
              python3-pip \
              python3.8-dev \
              wget \
              g++ \
              vim \
              git \
              curl \
              libcairo2-dev \
              pkg-config \
              libjpeg-dev \
              zlib1g-dev \
              libsdl2-2.0-0 \
              libsdl2-dev \
              liborc-0.4-dev \
              ocl-icd-opencl-dev \
              libopencv-dev \
              libboost-regex-dev \
              libgoogle-glog-dev \
              libsndio7.0 \
              libxv1 \
              libxfixes3 \
              net-tools \
              infiniband-diags \
              perftest  \
              libibverbs-dev && \
       apt-get clean && \
       rm -rf /var/lib/apt/lists/*
 
# Install pypi packages
RUN python3 -m pip install --upgrade pip --no-cache-dir && \
       python3 -m pip install --no-cache-dir onnxruntime \
              json_tricks \
              requests \
              regex \
              ipython \
              Cython \
              poetry \
              transformers==4.40.1 \
              loguru  \
              torch==2.0.1 \
              accelerate==0.33.0 \
              sentencepiece==0.2.0

WORKDIR /workspace