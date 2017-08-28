ARG BASE_IMAGE=python:2.7.13-jessie

FROM $BASE_IMAGE

ARG DEVELOPER_UID=1000

RUN pip install numpy==1.9.3 scipy==0.16.0 Cython==0.23 nose==1.3.7

RUN apt-get update && apt-get install -y python-matplotlib python-sphinx

RUN useradd --create-home -r -u $DEVELOPER_UID developer
USER developer

VOLUME /project

WORKDIR /project
