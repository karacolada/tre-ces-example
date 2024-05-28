# tre-ces
Containers for testing the TRE Container Execution Service

## Local build and push

```bash
export CR_PAT=<your-token>
echo $CR_PAT | docker login ghcr.io -u <username> --password-stdin
docker build -t ghcr.io/karacolada/tre-ces/<container_name>:<container_tag> -f docker/<path_to>/Dockerfile . 
docker push ghcr.io/karacolada/tre-ces/<container_name>:<container_tag>
```