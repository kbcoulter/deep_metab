# Docker Installation Guide (Ubuntu/Linux/WSL)

This guide walks through installing **Docker Engine** on Ubuntu using the official Docker repository, then shows how to **build a Docker image from a Dockerfile** that was used for graphormercontainer.

## 1. Update Package List

Update your system's package index:

```bash
sudo apt update
```

## 2. Install Required Dependencies

Install packages required for Docker’s repository and secure downloads.

```bash
sudo apt install apt-transport-https ca-certificates curl software-properties-common
```

These packages allow APT to retrieve packages over HTTPS and manage repository sources.

## 3. Add Docker’s Official GPG Key

Download and add Docker’s official GPG key to verify package authenticity.

```bash
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
| sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
```

## 4. Add Docker Repository

Add the Docker repository to your system’s APT sources.

```bash
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
| sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

## 5. Update Package Index Again

Refresh the package list so APT can access Docker packages.

```bash
sudo apt update
```

## 6. Install Docker

Install Docker Engine and required components.

```bash
sudo apt install docker-ce docker-ce-cli containerd.io
```

Installed components include:

* **docker-ce** – Docker Community Edition
* **docker-ce-cli** – Docker command line interface
* **containerd.io** – Container runtime

## 7. Verify Installation

Run the test container to confirm Docker is installed correctly.

```bash
sudo docker run hello-world
```

If Docker is installed successfully, you should see a confirmation message indicating the container ran correctly.

## Optional: Run Docker Without `sudo`

Add your user to the `docker` group so you can run Docker commands without `sudo`.

```bash
sudo usermod -aG docker $USER
```

Then log out and back in (or run `newgrp docker`) to apply the change.

## Confirm Docker Version

You can check the installed version with:

```bash
docker --version
```

Example output:

```text
Docker version 24.x.x, build xxxxx
```

## 8. Build a Docker Image from a Dockerfile

Once Docker is installed, you can build an image from a provided `Dockerfile`.

### Step 1: Navigate to the Project Directory

Move into the folder that contains the `Dockerfile`.

```bash
cd /path/to/project
```

You can confirm the file is there with:

```bash
ls
```

You should see something like:

```text
Dockerfile
```

### Step 2: Build the Image

Run the following command in the same directory as the `Dockerfile`:

```bash
docker build -t my-image-name .
```

Explanation:

* `docker build` tells Docker to build an image
* `-t my-image-name` assigns a name (tag) to the image
* `.` tells Docker to use the current directory as the build context

Example:

```bash
docker build -t proj_deepmetab:v0.3 .
```

### Step 3: Verify the Image Was Built

List local Docker images:

```bash
docker images
```

You should see your newly built image in the output.

Example:

```text
REPOSITORY       TAG       IMAGE ID       CREATED         SIZE
proj_deepmetab   v0.3      abcdef123456   1 minute ago    24.9GB
```

## 9. Build Using a Dockerfile with a Different Name

If the Dockerfile has a custom name, specify it with `-f`.

```bash
docker build -f MyDockerfile -t my-image-name .
```

Example:

```bash
docker build -f Dockerfile.reconstructed -t proj_deepmetab:v0.3 .
```

## 10. Run the Built Image

After building, you can start a container from the image:

```bash
docker run -it proj_deepmetab:v0.3
```

Explanation:

* `docker run` starts a container
* `-it` opens an interactive terminal
* `proj_deepmetab:v0.3` is the image name and tag

If the Dockerfile sets:

```dockerfile
CMD ["/bin/bash"]
```

this will open a shell inside the container.

## 11. Optional: Run with GPU Support

If the Docker image is based on NVIDIA CUDA and your system supports GPUs, run:

```bash
docker run --gpus all -it proj_deepmetab:v0.3
```

This allows the container to access available NVIDIA GPUs.

## 12. Optional: Mount a Local Directory into the Container

To access files from your host machine inside the container:

```bash
docker run -it -v /path/on/host:/workspace/data proj_deepmetab:v0.3
```

Example:

```bash
docker run -it -v $(pwd):/workspace proj_deepmetab:v0.3
```

This mounts your current directory into `/workspace` inside the container.


## Resources

* Docker Documentation: [https://docs.docker.com/](https://docs.docker.com/)
* Docker Hub: [https://hub.docker.com/](https://hub.docker.com/)

