FROM ubuntu:16.04

#  $ docker build . -t continuumio/anaconda3:latest -t continuumio/anaconda3:5.3.0
#  $ docker run --rm -it continuumio/anaconda3:latest /bin/bash
#  $ docker push continuumio/anaconda3:latest
#  $ docker push continuumio/anaconda3:5.3.0

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y sudo wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN useradd -rm -d /home/tta -s /bin/bash -g root -G sudo -u 1001 tta
RUN echo 'tta:password' | chpasswd
USER tta
WORKDIR /home/tta

RUN mkdir /home/tta/.conda
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /home/tta/anaconda3 && \
    rm ~/anaconda.sh && \
    echo ". /home/tta/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN conda install -y cudatoolkit=10.1 && \
	python -m pip install tensorflow-gpu



