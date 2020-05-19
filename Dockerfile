FROM python:3.7-buster

MAINTAINER Taylor Cavazos <taycav2@gmail.com>

WORKDIR /app
ADD . /app

ENV DEBIAN_FRONTEND=noninteractive
ENV OUTDATED_IGNORE=1
RUN apt-get update -qq -y && apt-get upgrade -qq -y
RUN apt-get --assume-yes install git r-base libgsl0-dev gcc

RUN pip install -r requirements.txt

RUN git clone https://github.com/slowkoni/rfmix.git &&\
	cd rfmix &&\
	autoreconf --force --install &&\ 
	./configure &&\
	make &&\
	cd .. &&\
	mv rfmix/simulate simulation/simulate-admixed &&\
	rm -rf rfmix

ENTRYPOINT ["python", "./run_simulation.py"] 
