FROM python:3

ARG PIPEDPYT_VERSION="3-1"

RUN mkdir /safe_data /safe_outputs /scratch

WORKDIR /usr/app
RUN mkdir src/ models/

COPY ./src/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ./src/train-torch.py src/
COPY ./models/resnet50.pth models/
COPY ./src/config.yml src/

CMD ["python3", "/usr/app/src/train-torch.py --config /usr/app/src/config.yml"]