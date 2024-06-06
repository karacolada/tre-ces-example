#! /bin/bash

cd ../
docker build -t ghcr.io/karacolada/tre-ces-example/piped-pyt:0.1 -f docker/piped-pyt/Dockerfile .
docker build -t ghcr.io/karacolada/tre-ces-example/nv-pyt:0.1 -f docker/nv-pyt/Dockerfile .
docker build -t ghcr.io/karacolada/tre-ces-example/pyt:0.1 -f docker/pyt/Dockerfile .
docker build -t ghcr.io/karacolada/tre-ces-example/data:0.3 -f docker/data/Dockerfile .
docker push ghcr.io/karacolada/tre-ces-example/piped-pyt:0.1
docker push ghcr.io/karacolada/tre-ces-example/nv-pyt:0.1
docker push ghcr.io/karacolada/tre-ces-example/pyt:0.1
docker push ghcr.io/karacolada/tre-ces-example/data:0.3