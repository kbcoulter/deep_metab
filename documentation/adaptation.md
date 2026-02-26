# ADAPTING SCRIPTS FOR NON-HPC OR NON-SLURM ENVIRONMENTS

We do not provide direct support for environments without SLURM or Apptainer(Singularity). This guide offers general instructions for adapting scripts to run outside SLURM-managed HPC systems or without Apptainer.

Container permissions and GPU access vary across systems. These instructions are general guidelines and may require adjustment for your specific environment. If you are working on an HPC system, consult your system documentation for container and GPU usage policies.

Docker and Apptainer behavior is **not identical**.
If unfamiliar with Apptainer, one resource for a quick overview can be found **[here](https://uoracs.github.io/talapas2-knowledge-base/docs/software/apptainer).**

---

## Running Outside HPC (No SLURM or Apptainer)

### 1. Ensure Docker With GPU Support Is Installed

Install Docker and, if using NVIDIA GPUs, install the [NVIDIA Container Toolkit](https://www.nvidia.com/en-us/drivers/).

If ``docker run –rm –gpus all nvidia/cuda:12.2.0-base nvidia-smi`` prints information, Docker GPU support should be configured.

### 2. Convert Apptainer Commands To Docker

```{bash}
### Apptainer:

apptainer run 
–nv 
–bind /data:/data 
workflow.sif

### Docker equivalent:

docker run 
–gpus all 
-v /data:/data 
username/image:tag
```

Notes:

- `–nv` in Apptainer becomes `–gpus all` in Docker
- `–bind` in Apptainer becomes `-v` in Docker (Volume Mount)
- `.sif` images must be replaced with a Docker image name and tag

### 3. Pull the Docker Image 

If using our Docker: `docker pull dnhem/proj_deepmetab:v0.3`

### 4. Remove SLURM

Delete lines beginning with: `#SBATCH`. These are scheduler directives and have no meaning outside SLURM

###  Example

```{bash}
#!/bin/bash

docker run –rm 
–gpus “device=${CUDA_VISIBLE_DEVICES:-all}” 
-v “$(pwd)”:/work 
-w /work 
dnhem/proj_deepmetab:v0.3 
bash setup_model/setup_HPC/HILIC_RP.sh
```

Notes:
- `–rm` removes the container after completion
- `${CUDA_VISIBLE_DEVICES:-all}` uses the environment variable if set, otherwise defaults to all GPUs
- `-w` sets the working directory inside the container


## If SLURM Is Not Available

If command `scancel` returns `command not found`, you are **not** in a SLURM environment.


### Change SLURM Specifications (or similar):

```{bash}
FROM:
--mem=64G
--cpus-per-task=8

TO:
apptainer run --cpus 8 workflow.sif
```

### Handling GPU:

```{bash}
CUDA_VISIBLE_DEVICES=0 apptainer run --nv workflow.sif
```

---

## Apptainer / Singularity Not Available  


If command `apptainer --version` returns `command not found`, you are **not** in an Apptainer/Singularity Environment.

### 1. Convert Apptainer Command to Docker

```{bash}
FROM: 
apptainer run \
    --nv \
    --bind /data:/data \
    workflow.sif

TO: 
docker run \
    --gpus all \
    -v /data:/data \
    username/image:tag
```

### 2. Handle GPU

```{bash}
FROM:
#SBATCH --gres=gpu:1
### OR SIMILAR

TO:
CUDA_VISIBLE_DEVICES=0 apptainer run --nv workflow.sif
```

### 3. System or Permission-Specific Changes  

Additional system or permission-specific modifications may be required. Please consult your HPC documentation for containerization guidance.

---

#### When Complete, your script may look something like this:

```{bash}
docker run --rm \
    -u $(id -u):$(id -g) \
    --gpus "device=$CUDA_VISIBLE_DEVICES" \
    -v $PWD:$PWD \
    -w $PWD \
    dnhem/proj_deepmetab:v0.3 \
    python setup_model/setup_HPC/HILIC_RP.sh
```

> **NOTE:** When converting scripts, we recommend following these steps as a general guide. These changes have not been tested and environment-specific issues may arise.

## Important Notes
- These changes are general guidelines and may require modification depending on system configuration.
- GPU access requires proper driver installation and Toolkit configuration.
- Always test with a small workload before running full pipelines.


## Contact or Contribute

If you are adapting these scripts for your own use and have questions, feedback, or suggestions, please feel free to reach out. We are happy to help if we can, though support may be limited.

- kcoulter [at] uoregon [dot] edu  
- dnhem [at] uoregon [dot] edu  
- ewi [at] uoregon [dot] edu  

## Help Improve This Guide! 

Found helpful suggestions not on this document? Please consider submitting a pull request to update this manual! Contributions that make this resrouce more accurate and/or useful are especially welcome. 