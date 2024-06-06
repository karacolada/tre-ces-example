# tre-ces-example
Containers for testing the TRE Container Execution Service.

Use case: Training an object detection model for the [Fathom24 Kaggle competition](https://www.kaggle.com/competitions/fathomnet2024).
- containerising data [`docker/data`](./docker/data/)
- containerising pre-built models (:construction: TODO)
- containerising training scripts with different starting points
  - NVIDIA PyTorch image [`docker/nv-pyt`](./docker/nv-pyt/)
  - PyTorch image [`docker/pyt`](./docker/pyt/)
  - Python image [`docker/piped-pyt`](./docker/piped-pyt/)
  - Tensorflow image (:construction: TODO)

:construction: GH actions will be used for CI/CD. In the meantime, build locally and push to this repository's CR instead, as described below.

## Local build and push

```bash
export CR_PAT=<your-token>
echo $CR_PAT | docker login ghcr.io -u <username> --password-stdin
docker build -t ghcr.io/karacolada/tre-ces/<container_name>:<container_tag> -f docker/<path_to>/Dockerfile . 
docker push ghcr.io/karacolada/tre-ces/<container_name>:<container_tag>
```

In the SH:

```bash
sudo ces-pull <user> <token> ghcr.io/karacolada/tre-ces-example/<container_name>:<container_tag>
# no GPU, blocking
ces-run ghcr.io/karacolada/tre-ces-example/<container_name>:<container_tag>
# with GPU, blocking
ces-gpu-run ghcr.io/karacolada/tre-ces-example/<container_name>:<container_tag>
```
