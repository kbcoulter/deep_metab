
# deep_metab

## Applying Deep Learning to LC-MS Data to Improve Metabolite Annotation


---

## Background

Liquid chromatography–mass spectrometry (LC‑MS) enables metabolite identification using **molecule mass**, **mass entropy**, and **retention time (RT)**. However, RT varies significantly across LC–MS setups and structural isomers often share near-identical masses, making metabolite annotation difficult and error‑prone.

Recent deep‑learning models, like **[Graphormer‑RT](https://pubs.acs.org/doi/10.1021/acs.analchem.4c05859)** , can predict retention times dependent on specific LC–MS configurations, offering a path to improved annotation workflows [(repository)](https://github.com/HopkinsLaboratory/Graphormer-RT).

This repository provides a full pipeline for applying modern deep learning to LC–MS Hydrophilic Interaction Liquid Chromatography (HILIC) and Reverse Phase (RP) data to improve annotation.

## Table of Contents:
- [Background](#background)
- [Quick Links](#quick-links)
- [Features](#features)
- [Manuals (HowTo)](#how-to)
- [Warnings](#warning)
- [Common Issues](#common-issues)
- [References](documentation/references.md)
- [Contact](#contact)
- [Contribute](#contribute)
- [License](#license)


## Quick Links:
- **Graphormer-RT For Deep Learning:**
    - **Graphormer-RT: https://github.com/HopkinsLaboratory/Graphormer-RT**
    - **Our Submodule: [https://github.com/kbcoulter/Graphormer-RT](https://github.com/kbcoulter/Graphormer-RT/)**
- **Zenodo For Model Weights**:
    - **Graphormer-RT or "OG" Weights: https://zenodo.org/records/15021743**
    - **Our or "DM" Weights: https://zenodo.org/records/18867980**
 - **Docker: https://hub.docker.com/r/dnhem/proj_deepmetab**
 - **RepoRT: https://github.com/michaelwitting/RepoRT**

## Features

#### Below is an organized summary of the main capabilities provided by this repository:

| **Feature Category** | **Description** |
|----------------------|-----------------|
| **LC–MS Preprocessing** | Tools to clean, format, and structure LC–MS datasets for prediction |
| **Data Loaders** | Scripts to load, featurize, and register data  |
| **Workflow Setup** | Setup directory, container, etc. for RP/HILIC/Both RT prediction and annotation  |
| **Model Training** | Train RP Graphormer‑RT from scratch (Not Recomended)|
| **Model Finetuning** | Finetune Graphormer‑RT models (HILIC transfer learning and HILIC finetuning) |
| **RT Prediction** | Generate RT predictions and integrate results back into LC–MS feature tables |
| **Annotation Scoring Framework** | Score candidate molecules to resolve annotation ambiguities |
| **Quality Control** | Automatically flag potentially mis‑annotated mass feature IDs |
| **Stereoisomer Flagging** | Identify and label stereoisomer mass feature IDs |
| **LazyPredict ✨** | Run **entire** automated workflow|




## How To:

Choose the manual based on your preferences:


| Environment | Workflow Type | Documentation | Status |
|-------------|---------------|---------------|--------|
| **HPC / Cloud** | Step-by-Step | [Manual Guide](documentation/app_manual.md) | ✅ |
| **HPC / Cloud** | Automated | [LazyPredict w/ Nextflow](documentation/nextflow_pipeline.md) | ✅ |
| **Local on Windows** | Building Docker Image* | [Docker Guide Installation](documentation/docker_guide.md) | ✅ |

> *Note: Building the docker image is not required for our workflow since users can easily pull it from dockerhub as listed under [quick links](#quick-links). The Dockerfile is only provided to showcase the configurations required to construct the image.

#### Support for Non-HPC Environments  

Because non-HPC resources are limited, we do not provide support for running this tool outside SLURM-managed systems or without Apptainer (Singularity). If you wish to run this workflow in such environments (typically non-cloud or non-HPC), please refer to our **[Script Adaptation Guidelines](documentation/adaptation.md)**.


## Warnings:


> **DO NOT MOVE `.sif` FILES**  
> The following directories are bind‑mounted and depend on fixed paths and filenames. Take extreme caution editing.
```
workspace/Graphormer-RT/checkpoints_RP/
graphormer_checkpoints_RP/
workspace/Graphormer-RT/checkpoints_HILIC/
graphormer_checkpoints_HILIC/
my_data/HILIC_ft/
workspace/Graphormer-RT/my_data/HILIC_ft/
```

## Common Issues:
* Fairseq Errors: Please ensure that you are using the Docker provided. The original [Graphormer](https://github.com/microsoft/Graphormer) was built on a Snapshot Release of [fairseq](https://github.com/facebookresearch/fairseq). This version is not available via conda/mamba. If error persists, please ensure that container version installation matches the most recent available on [Docker](https://hub.docker.com/r/dnhem/proj_deepmetab).

* Python Package Issues: Please first ensure that you are using the [Docker](https://hub.docker.com/r/dnhem/proj_deepmetab) materials provided.
 
* Obviously Incorrect Predictions: This workflow does not verify that HILIC data is used with the HILIC model (or RP data with the RP model). As a result, it is easy to accidentally input HILIC data into the RP model, which produces extremely inaccurate predictions.

* Apptainer/SLURM Issues: Please run the commands `apptainer --version` and `scancel` to ensure that these tools are available. If they are not, reference this [guide](documentation/adaptation.md).


* **If your issue is not shown above, please [raise an issue](https://github.com/kbcoulter/deep_metab/issues/new).**




## References

#### **Please access our references [here](documentation/references.md)**

Special thanks to the Graphormer‑RT development team, our collaborators at PNNL, and to the BGMP instructional team for their mentorship. This work benefited from access to the University of Oregon HPC cluster, [Talapas](https://uoracs.github.io/talapas2-knowledge-base/).


## Contact

If you have questions, feedback, or ideas, feel free to reach out to any of us:

- kcoulter [at] uoregon [dot] edu  
- dnhem [at] uoregon [dot] edu  
- ewi [at] uoregon [dot] edu  

## Contribute
### We welcome (and encourage) contributions!

If you find this project helpful or interesting, please consider starring the repository ⭐

While active development has concluded, PRs are welcome! Feel free to contribute by opening a pull request with fixes, improvements, or new features.

## License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for more details.



