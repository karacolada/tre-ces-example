FROM nvcr.io/nvidia/pytorch:24.03-py3

ARG NVPYT_VERSION="24.03-1"

RUN mkdir /safe_data /safe_outputs /scratch

WORKDIR /usr/app
RUN mkdir src/ models/

COPY ./src/requirements_no_torch.txt requirements.txt
RUN pip install -r requirements.txt

COPY ./src/train-torch.py src/
COPY ./models/resnet50.pth models/
COPY ./src/config.yml src/

CMD ["python3", "usr/app/src/train-torch.py --config /usr/app/src/config.yml"]

