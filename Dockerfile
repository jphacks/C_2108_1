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
    \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --upgrade pip setuptools \
    && pip install --requirement ./requirements.txt --find-links https://download.pytorch.org/whl/torch_stable.html

COPY . ./
EXPOSE 8000
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]