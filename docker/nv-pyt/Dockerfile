FROM nvcr.io/nvidia/pytorch:24.03-py3

RUN mkdir /safe_data /safe_outputs /scratch

WORKDIR /scratch
COPY ./src/requirements.txt ./
RUN pip install -r requirements.txt

COPY ./src/train-torch.py ./

ENV DATA_JSON="/safe_data/ag-gpu/kmoraw-gpu/ocean_data.json"
ENV DATA_IMAGES="/safe_data/ag-gpu/kmoraw-gpu/ocean_images"
ENV TRAIN_OUTPUT="/safe_outputs"

CMD ["python3", "./train_torch.py"]

