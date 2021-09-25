FROM python:3.9

WORKDIR /

COPY . ./

RUN apt-get update -y && \ 
  apt-get install apt-utils pandoc -y && \
  apt-get install libopenmpi-dev cmake libz-dev build-essential libssl-dev libffi-dev python-dev -y

RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt
#RUN pip install -e .[docs,tests]

RUN pip install -e .[tensorforce,baselines,fbm]
#RUN pip install tensortrade[tf,tensorforce,baselines,ccxt,fbm]@git+https://github.com/lakhwindersingh/tensortrade.git

RUN pip install -r ./examples/requirements.txt