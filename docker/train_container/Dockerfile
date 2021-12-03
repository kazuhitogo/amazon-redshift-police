FROM python:3.9-slim-buster

RUN apt-get -y update && \
    apt-get -y upgrade
                       
WORKDIR /workspace
COPY requirements.txt /workspace

RUN pip3 install -r requirements.txt
ENV PYTHONUNBUFFERED=TRUE