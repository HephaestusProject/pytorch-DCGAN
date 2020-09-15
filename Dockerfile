FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
ADD requirements.txt /workspace
RUN pip install -r requirements.txt 
COPY . /workspace