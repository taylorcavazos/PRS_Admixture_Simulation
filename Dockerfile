FROM python:3.7-buster

MAINTAINER Taylor Cavazos <taycav2@gmail.com>

WORKDIR /app
ADD . /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq -y && apt-get upgrade -qq -y
RUN apt-get --assume-yes install git r-base libgsl0-dev

RUN pip install -r requirements.txt

#RUN git clone https://github.com/slowkoni/rfmix.git
#RUN cd rfmix
#RUN autoreconf --force --install 
#RUN ./configure
#RUN make
#RUN cd ..
#RUN mv rfmix/simulate simulation/simulate-admixed
#RUN rm -rf rfmix 
