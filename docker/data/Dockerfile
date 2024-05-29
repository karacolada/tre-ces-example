FROM alpine:3.20.0

RUN mkdir /safe_data /safe_outputs /scratch

COPY ./data/subtrain/ /scratch/ocean_images
COPY ./data/subtrain.json /scratch/ocean_data.json
COPY ./src/copy_data.sh /scratch/copy_data.sh

RUN chmod u+x /scratch/copy_data.sh

CMD sh /scratch/copy_data.sh

