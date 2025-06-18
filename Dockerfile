FROM nvidia/cuda:11.7.1-devel-ubuntu20.04
ENV MINICONDA_VERSION="py312_24.5.0-0"
ENV CONDA_ENV_NAME="mamba_cuda_env"
ENV DEBIAN_FRONTEND noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8


RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    vim \
    gawk perl python3

#instal packages advised by Gianluca Della Vedova <gianluca@dellavedova.org>
RUN apt-get install -y procps uuid-runtime

WORKDIR /tmp
SHELL ["/bin/bash", "-c","-l"]

RUN wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh 
RUN chmod ugo+x miniconda.sh 

RUN ./miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/profile

COPY mamba_cuda_env.yml /tmp/mamba_cuda_env.yml
RUN conda config --add channels defaults && conda config --add channels bioconda && conda config --add channels conda-forge
RUN conda env create -f /tmp/mamba_cuda_env.yml

ENV PATH "/opt/conda/bin:/opt/conda/envs/${CONDA_ENV_NAME}/bin:${PATH}"
ENV CONDA_DEFAULT_ENV $CONDA_ENV_NAME
ENV CONDA_PREFIX /opt/conda/envs/$CONDA_ENV_NAME

# for interactive occam-run which uses -l and has the user home mounted by default
RUN echo "export PATH=/opt/conda/bin:/opt/conda/envs/${CONDA_ENV_NAME}/bin:${PATH}" >> /etc/profile

CMD ["/bin/bash"]
