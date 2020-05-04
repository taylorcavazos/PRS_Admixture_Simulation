FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends build-essential r-base r-cran-randomforest python3.7 python3-pip python3-setuptools python3-dev pkg-config

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip3 install -r requirements.txt

RUN "git clone https://github.com/slowkoni/rfmix.git"
RUN "cd rfmix"
RUN "autoreconf --force --install"
RUN "./configure"
RUN "make"
RUN "cd .."
RUN "mv rfmix/simulate simulation/simulate-admixed"
RUN "rm -rf rfmix"

COPY . /app
