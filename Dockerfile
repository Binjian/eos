# set base image (host S)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

LABEL maintainer="binjian.xin@newrizon.com"

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.0 driver>=450"
ENV TZ=Asia/Shanghai

# ENV HTTP_PROXY=http://127.0.0.1:20171
# ENV HTTPS_PROXY=http://127.0.0.1:20171
# RUN add-apt-repository "deb http://archive.ubuntu.com/ubuntu jammy universe"
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
    apt-get install -y --no-install-recommends tzdata

# set local time zone to Shanghai
#RUN apt-get install dialog apt-utils -y
#RUN echo set debconf to Noninteractive",
#RUN echo 'debconf debconf/frontend select Noninteractive'
#RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
#RUN apt-get install -y python3.10
#RUN apt-get install -y git
#RUN apt-get install -y curl
#
#RUN which python
#RUN python3 -m ensurepip --upgrade
RUN apt-get install -y python3-pip &&\
    pip install --upgrade pip &&\
    pip install 'poetry==1.2'
##RUN mkdir -p /etc/poetry
##RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/etc/poetry python3 -
##ENV PATH="$PATH:/etc/poetry/bin"
#
COPY . /app
# set the working directory in the container
WORKDIR /app
#
#RUN python3 -c 'import urllib3; http = urllib3.PoolManager(maxsize=100)'
# copy the code to the working directory
RUN poetry install
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
#    'poetry run python eos/realtime_train_infer_ddpg.py --cloud -t -p testremote -r