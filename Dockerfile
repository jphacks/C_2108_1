FROM ubuntu:18.04

ENV PYTHONIOENCODING=utf-8 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8
    #LANG=ja_JP.utf8

WORKDIR /app

RUN apt-get update \
    && apt-get install -y \
        build-essential \
        cmake \
        git \
        gfortran \
        libffi-dev \
        python3 \
        python3-dev \
        python3-pip \
        wget


RUN mkdir tmp \
    && cd ./tmp \
    && wget -c https://github.com/ku-nlp/jumanpp/releases/download/v2.0.0-rc3/jumanpp-2.0.0-rc3.tar.xz \
    && tar xf ./jumanpp-2.0.0-rc3.tar.xz \ 
    && mkdir bld \ 
    && cd bld \
    && cmake ../jumanpp-2.0.0-rc3 \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
    \
    && make install \
    && cd ../../ \ 
    && rm -r ./tmp
    #&& apt-get clean \
    #&& rm -rf /var/lib/apt/lists/*

COPY . ./
RUN pip3 install --upgrade pip setuptools \
    && pip3 install --requirement ./requirements.txt --find-links https://download.pytorch.org/whl/torch_stable.html \
    && cd ./fairseq \
    && pip3 install --editable .

EXPOSE 8000
CMD ["python3", "main.py"]
#CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]