FROM nvcr.io/nvidia/pytorch:24.03-py3

RUN mkdir /safe_data /safe_outputs /scratch
RUN mkdir /src

COPY ./src/requirements_no_torch.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt

COPY ./src/train-torch.py /src

ENV DATA_JSON="/safe_data/kmoraw-gpu/ocean_data/ocean_data.json"
ENV DATA_IMAGES="/safe_data/kmoraw-gpu/ocean_data/ocean_images"
ENV TRAIN_OUTPUT="/safe_outputs"

CMD ["python3", "/src/train-torch.py"]

