FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-devel

RUN pip install pip --upgrade
COPY ./requirements.txt  ./requirements.txt
RUN pip install -r ./requirements.txt
RUN rm ./requirements.txt