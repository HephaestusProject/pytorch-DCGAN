FROM python:3.8.5
ADD serving/requirements.txt /workspace
RUN pip install -r requirements.txt 
COPY . /workspace