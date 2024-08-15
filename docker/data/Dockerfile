FROM alpine:3.20.0

ARG DATA_VERSION="1"

RUN mkdir /safe_data /safe_outputs /scratch
RUN mkdir /data

COPY ./data/subtrain/ /data/ocean_images
COPY ./data/subtrain.json /data/ocean_data.json
COPY ./src/copy_data.sh /src/copy_data.sh

RUN chmod a+x /src/copy_data.sh

CMD ["sh", "/src/copy_data.sh"]

