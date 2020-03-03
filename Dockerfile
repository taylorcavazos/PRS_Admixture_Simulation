
FROM python:3
RUN pip install -r requirements.txt

RUN "git clone https://github.com/slowkoni/rfmix.git"
RUN "cd rfmix"
RUN "autoreconf --force --install"
RUN "./configure"
RUN "make"
RUN "cd .."
RUN "mv rfmix/simulate simulation/simulate-admixed"
RUN "rm -rf rfmix"
