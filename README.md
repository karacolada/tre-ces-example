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

:construction: GH actions will eventually be used for CI/CD. In the meantime, build locally and push to this repository's CR instead, as described below.

## Guidance

The National Safe Haven provides tools that allow researchers to run their analyses using containers.
Rather than asking system administrators to install software in specific VMs after acquiring approval from the research coordinator (RC),
researchers can package any software requirements in a container and run it in a safe, isolated manner.

### Background

A *container* is a piece of software in which an application is packaged with an isolated environment (a guest operating system with libraries and dependencies already installed) that includes everything the application requires to run. Containerisation is a virtualisation technique, but much more lightweight than a virtual machine. 

Containers do not have a fixed hardware mapping of CPUs, memory, and disk space like virtual machines. Instead, they share those resources like any other process on the host system would. They are usually not built for direct interaction with the user but meant to run in the background. Usually, a container will either provide a service that users can interact with, such as a database or website, and run continuously, or it will run an application once and terminate afterwards.

#### Container construction

The typical workflow for containers is as follows:

1. Definition: The container is defined using a *Dockerfile* which contains a set of instructions. The instructions specify the desired operating system, how dependencies should be installed, what files should be included, and more.
2. Build: The containers is built from the Dockerfile, meaning that every instruction in the Dockerfile is carried out. This process takes a while, as software packages are usually downloaded and installed. The result is the packaged environment and application.
3. Run: This is the only stage that needs to take place on the target host. The container starts up and runs the application in the packaged environment. It terminates after the application has finished.

If using a container registry, the workflow is slightly amended:

1. Definition
2. Build
3. Push: The built container is pushed to the container registry.
4. Pull: The container is pulled from the container registry onto the target host.
5. Run

We will go into practical details in later sections.

#### Container isolation 

Applications in a container have no knowledge of running inside a container: all they see is the environment they were packaged with. This environment looks just like a normal operating system with all the components necessary for running the application already installed. The guest operating system can be different from the host operating system. All content, code and data in a container is isolated from the host filesystem and network.

By default, the container cannot access directories in the host filesystem, i.e. code running inside the container cannot read data not packaged within, and it cannot write or modify anything outside the container. However, the container can be configured at runtime to have access to specific directories in the filesystem through *volumes* or *bind mounts*. This allows any application within the container to access the specified host directory. By default, read and write access is enabled, but can be limited to read-only. Similarly, a container can be configured at runtime to expose its network ports to the outside world. In practice, ports of the host system are mapped to ports of the container, allowing communication to and from the container for all actors with access to the host port.

In the Safe Haven, the runtime settings are pre-configured and researchers and RCs are currently not able to make changes to that configuration.
The container gets access to a small set of pre-defined host directories and a GPU if available.
They run non-interactively, meaning that researchers cannot log into the container while it is running.
They have no access to the host network, and it should also be noted that the host network has no access to the internet - even if a container were to get access to the host network, it will not be able to load data from the internet.

##### File structure

Containers have access to three directories located on the host system as detailed in the following table:

| Directory on the host system | Directory in the container | Intended use |
| --- | --- | --- |
| `/safe_data/<your_project_name>/` | `/safe_data/` | Read-only access to data and other files. |
| `~/outputs_<unique_id>` | `/safe_outputs/` | Will be created at container startup as an empty directory. Intended for any outputs: logs, data, models. |
| `~/scratch_<unique_id>` | `/scratch` | Temporary directory that is removed after container termination on the host system. Any temporary files should be placed here. |

Later sections cover how these directories should be used for different workflows in more detail.

###### About `/scratch`

Currently, temporary files can also be written into any directory in the container file system.
However, we would encourage the use of `/scratch` instead, mainly for two reasons:

1. In the future, write access to the container file system might be further restricted for security reasons. Writing only to `/scratch` at runtime is more future-proof.
2. Using `/scratch` can be more efficient, as we are able to map it to high-performing hardware.

### Making a workflow container-ready

:construction: **TODO:** rewrite as sections with more detail, and order for Dockerfile specification. Start with an explanation of a Dockerfile.

If starting from scratch, we would recommend using our :construction: template repository as a starting point. 
This should make container building very smooth.

To adapt existing workflows for containerisation, you need to answer a handful of questions:
1. What files do I need to run my application, and where does the application expect them? Collect any scripts, data, etc. that your application needs. You won't be able to quickly copy them into the container after it has been built. Less is more here - don't include things you don't need as those will unnecessarily increase build and load time. Decide what is needed at build time and what is needed at runtime.
2. Can my application run unsupervised? You won't be able to interact with the running application.
3. Which steps of my workflow do I want to reuse, and how can I configure those? Switch from command line arguments to configuration files - you can change those in `/safe_data` and have the container read them at runtime. Split your workflow into modules, for example if you want to run an analysis multiple times but data transformation needs to run only once. (DRY)
4. If I wanted to run this application on a new machine, what would I need to prepare? This will tell you what your software requirements are, and help to figure out how to install them in a container.
5. When does my machine communicate with the internet? The Safe Haven doesn't have access to the internet, and containers are no exception. To start with, you could try running your application while switching of your machine's internet connection. You might have previously downloaded models into the cache though - this is where local testing of the container will help. Anything that needs to be downloaded from the internet has to be provided and packaged with the container.

> Containers usually run a Linux-based operating system, and the Safe Haven assumes as much. Base images are often more sophisticated than a bare-metal operating system though, so familiarity with Linux is rarely required.

### Building and pushing a container

:construction: TODOs:
- [ ] what it means to build
- [ ] layers
- [ ] rebuilding when changes are made

#### Manual & local

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

#### Automated

:construction: GH actions or other CI/CD tools

### Running containers in the SH

:construction: TODOs:
- [ ] who can run a container - researchers vs RCs
- [ ] interim solution
- [ ] Open OnDemand