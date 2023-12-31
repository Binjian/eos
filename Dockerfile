## set base image (host S)
ARG BASE_IMAGE=mambaorg/micromamba:1.2-jammy-cuda-11.8.0
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
RUN echo "BASE_IMAGE=${BASE_IMAGE}"

LABEL maintainer="binjian.xin@newrizon.com"
ARG builder=binjian.xin
ARG version=1.0.0
LABEL builder=$builder
LABEL base_image = $BASE_IMAGE

# Micromamba environment
ARG MAMBA_USER=mamba
ARG MAMBA_USER_ID=1000
ARG MAMBA_USER_GID=1000
ENV MAMBA_USER=$MAMBA_USER
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

#RUN echo set debconf to Noninteractive",
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.0 driver>=450"
ENV TZ=Asia/Shanghai
ENV PIP_DEFAULT_TIMEOUT=600

USER root
# set local time zone to Shanghai
RUN apt-get update -y --no-install-recommends &&\
    apt-get install -y software-properties-common &&\
    add-apt-repository "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted multiverse" &&\
    add-apt-repository -y ppa:deadsnakes/ppa &&\
    apt-get install apt-utils &&\
    apt-get install debconf &&\
    apt-get install -y --no-install-recommends python3.10 &&\
    apt-get install -y --no-install-recommends python3-pip &&\
    apt-get install -y --no-install-recommends git &&\
    apt-get install -y --no-install-recommends curl &&\
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone &&\
    apt-get install -y --no-install-recommends tzdata &&\
    rm -rf /var/lib/apt/lists/* &&\
    apt-get -y autoremove &&\
    apt-get clean &&\
    pip install --upgrade pip



#RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple &&\
#    pip config set global.timeout 150 &&\


#SHELL ["/bin/bash", "--login", "-c"]

#RUN micromamba self-update --yes -c conda-forge

#COPY --chown=$MAMBA_USER:$MAMBA_USER ueos-env.yaml /tmp/ueos-env.yaml
#RUN micromamba install -n base -f /tmp/ueos-env.yaml -c conda-forge &&\
#    micromamba clean --all --yes
USER $MAMBA_USER

ARG MAMBA_DOCKERFILE_ACTIVATE=1

COPY --chown=$MAMBA_USER:$MAMBA_USER ueos-env-lock.txt /tmp/ueos-env-lock.txt
RUN micromamba install -n base -f /tmp/ueos-env-lock.txt -c conda-forge -c nvidia &&\
    micromamba clean --all --yes

RUN pip install rocketmq pyqt5 cutelog poetry2conda

RUN micromamba --version &&\
    python --version &&\
    pip --version &&\
    micromamba list

COPY . /app
# set the working directory in the container
WORKDIR /app

RUN ["git", "config", "--global", "--add", "safe.directory", "/app"]

RUN pip install -e .
#VOLUME /app/data

RUN python -c 'import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices("GPU")); print(tf.test.is_built_with_cuda()); print(tf.__file__)'
#RUN ["micromamba", "run", "python", "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices("GPU")); print(tf.test.is_built_with_cuda()); print(tf.__file__)"]
#ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "eos/realtime_train_infer_ddpg.py", "-v", "VB7", "-d", "longfei", "-m", "can_intra", "-w", "rocket_intra", "-u", "mobile", "-o", "mongo_local", "-i"]
# ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "eos/realtime_train_infer_rdpg.py", "-v HMZABAAH7MF011058", "-d longfei"]


#ENTRYPOINT ["poetry", "run", "python", "eos/realtime_train_infer_rdpg.py", "-v HMZABAAH7MF011058", "-d longfei"]
#ENTRYPOINT ["/bin/sh", "-c", "micromamba activate ueos | python eos/realtime_train_infer_rdpg.py -v HMZABAAH7MF011058 -d longfei"]

#RUN mkdir -p /app/data/udp-pcap &&\
#    cp ./eos/config/*.csv /app/data/ &&\
#    mkdir -p /app/data/testremote/py_logs &&\
#    mkdir -p /app/data/testremote/tables &&\
#    mkdir -p /app/data/testremote/tf_logs-vb/ddpg &&\
#    mkdir -p /app/data/testremote/tf_logs-vb/rdpg &&\
#    mkdir -p /app/data/testremote/tf_ckpts-vb/l045a_ddpg_actor &&\
#    mkdir -p /app/data/testremote/tf_ckpts-vb/l045a_ddpg_critic &&\
#    mkdir -p /app/data/testremote/vb_checkpoints/rdpg_actor &&\
#    mkdir -p /app/data/testremote/vb_checkpoints/rdpg_critic &&\
#    cp ./eos/config/*.csv /app/data/testremote

#
#RUN python3 -c 'import urllib3; http = urllib3.PoolManager(maxsize=100)'
# copy the code to the working directory
# PORT
# mongodb:
##        Url="127.0.0.1",  # url for the database server
##        Port=27017,  # port for the database server

##  RemoteCANHost="10.0.64.78:5000",  # IP address of the remote CAN host
## TripControlHost="10.0.64.78:9876",  # IP address of the trip control host

## cutelog:
#        skh = SocketHandler("127.0.0.1", 19996)
#EXPOSE 5000


# RUN #poetry shell


#ENTRYPOINT ["/bin/bash"]
#ENTRYPOINT ["bash", "poetry", "shell"]
# ENTRYPOINT ["poetry", "shell"]
#ENTRYPOINT ["poetry", "run", "python", "eos/realtime_train_infer_ddpg.py", "--cloud -r -t -p cloudtest"]
# install depnedencies
# RUN pip install -r requirements.txt

# RUN pip install --editable .
# docker container run -it --gpus all eos:latest /bin/sh -c
# 'poetry run python eos/realtime_train_infer_ddpg.py --cloud -t -p testremote -r'
# docker run -rm --gpus 'all, "capabilities=compute,utility"' \
#   nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi


# docker container run -it --gpus all eos-gpu:latest /bin/sh -c
# 'poetry run python eos/realtime_train_infer_ddpg.py --cloud -t -p testremote -r'
# PORT
# mongodb:
##        Url="127.0.0.1",  # url for the database server
##        Port=27017,  # port for the database server

##  RemoteCANHost="10.0.64.78:5000",  # IP address of the remote CAN host
## TripControlHost="10.0.64.78:9876",  # IP address of the trip control host

## cutelog:
#        skh = SocketHandler("127.0.0.1", 19996)
# docker container run -it --gpus all --network host -p 27017:27017 -p 19996:19996 eos-gpu:latest /bin/sh -c
# 'poetry run python eos/realtime_train_infer_ddpg.py --cloud -t -p testremote -r'
# docker container run -it --rm --gpus all --network host  \
#    -p 27017:27017 -p 19996:19996 eos-gpu:latest /bin/sh -c  \
#    'poetry run python eos/realtime_train_infer_ddpg.py --cloud -t -p testremote -r'

# docker container run -it --rm --gpus all --network host  \
#    -p 27017:27017 -p 19996:19996 -p 6006:6006 eos:latest /bin/sh -c  \
#    'poetry run python eos/realtime_train_infer_ddpg.py -v "HMZABAAH7MF011058" -d "longfei"'
# docker build --network=host --build-arg BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 -t eos .
# entrypoint
# ENTRYPOINT ["poetry", "run", "python", "eos/realtime_train_infer_ddpg.py", "--cloud -r -t -p cloudtest"]

# docker container run -it --gpus all --network host --mount source=eosdata,target=/app/data eos:local
# docker run -it --gpus all --network host --mount source=eosdata,target=/app/data --entrypoint "/bin/sh" eos:baiduyun -c 'poetry shell | poetry run python eos/realtime_train_infer_rdpg.py -v "HMZABAAH7MF011058" -d "longfei" -m "10.0.64.78:5000" -u "10.0.64.78:9876"'
# volume
# VOLUME /app/data

# network
# EXPOSE 27017


#docker image build --network=host --build-arg BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 -t eos .
#docker build --network=host --build-tag BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 -t eos .
#docker build --network=host --build-arg BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 -t eos https://gitlab.newrizon.work/its/ai/eos.git#DOCKER
#docker build --network=host --build-arg BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 -t eos https://gitlab.newrizon.work/its/ai/eos.git#DOCKER:docker
#docker build --network=host --build-arg BASE_IMAGE=registry.cn-shanghai.aliyuncs.com/tengfeiwu/nvidia-cuda:11.8.0 -t eos:baiduyun https://gitlab.newrizon.work/its/ai/eos.git#DOCKER
#docker build --network=host -t ueos:cloud https://gitlab.newrizon.work/its/ai/eos.git#DOCKER_TEST | tee docker-build-cloud.txt

# build ueos:
# 1. local micromamba env build
# 1.1 poetry update/lock/build eos,
# 1.1 activate micromamba env,
# 2. conda list --explicit --md5 >ueos-env-lock.txt
# 3. docker build --network=host -t ueos:lock -f Dockerfile.umamba .
# run container from image ueos:
# 1. docker container run -it --gpus all --network host ueos:cloud /bin/bash
# 2. docker run -it --gpus all --network host --mount source=eosdata,target=/app/data  --entrypoint "/bin/bash" ueos:cloud
# 3. docker run -it --gpus all --network host --mount source=eosdata,target=/app/data  --entrypoint "/bin/sh" ueos:cloud -c 'python eos/realtime_train_infer_rdpg.py -v "HMZABAAH7MF011058" -d "longfei" -m "10.0.64.78:5000" -u "10.0.64.78:9876" -o "ivy"'
#
#docker build --network=host -t ueos:lock -f Dockerfile .
#docker build --network=host -t ueos:cloud https://gitlab.newrizon.work/its/ai/eos.git#DOCKER_TEST | tee docker-build-cloud.log
