FROM python:2.7.13-jessie

RUN pip install numpy==1.8.2 scipy==0.13.3 Cython==0.23 nose==1.3.7

RUN useradd --create-home -r -u 1000 developer
USER developer

VOLUME /project

WORKDIR /project
