# set base image (host S)
FROM ubuntu:jammy-20221003

LABEL maintainer="binjian.xin@newrizon.com"

# ENV HTTP_PROXY=http://127.0.0.1:20171
# ENV HTTPS_PROXY=http://127.0.0.1:20171

RUN apt-get update -y
RUN apt-get install dialog apt-utils -y
RUN apt-get install software-properties-common -y
RUN add-apt-repository -y ppa:deadsnakes/ppa
#RUN echo set debconf to Noninteractive",
#RUN echo 'debconf debconf/frontend select Noninteractive'
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get install -y python3.10
RUN apt-get install -y git
RUN apt-get install -y curl
#
#RUN which python
#RUN python3 -m ensurepip --upgrade
RUN apt-get install -y python3-pip
RUN pip install --upgrade pip
RUN pip install 'poetry==1.2'
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

ENTRYPOINT ["bash"]
#ENTRYPOINT ["bash", "poetry", "shell"]
#ENTRYPOINT ["poetry", "shell"]
#ENTRYPOINT ["poetry", "run", "python", "eos/realtime_train_infer_ddpg.py", "--cloud -r -t -p cloudtest"]
# install depnedencies
# RUN pip install -r requirements.txt

# RUN pip install --editable .