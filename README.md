
# deep_metab

### Applying Deep Learning to LC-MS Data to Improve Metabolite Annotation

> **⚠️ This project is under active development. Expect rapid changes.**
---

## Background

Liquid chromatography–mass spectrometry (LC‑MS) enables metabolite identification using **molecule mass**, **mass entropy**, and **retention time (RT)**. However, RT varies significantly across LC–MS setups and structural isomers often share near-identical masses, making metabolite annotation difficult and error‑prone.

Recent deep‑learning models, like **[Graphormer‑RT](https://github.com/HopkinsLaboratory/Graphormer-RT)**, can predict retention times dependent on specific LC–MS configurations, offering a path to improved annotation workflows.

This repository provides a full pipeline for applying modern deep learning to LC–MS Hydrophilic Interaction Liquid Chromatography (HILIC) and Reverse Phase (RP) data to improve annotation.

## Table of Contents:
- [Background](#background)
- [Quick Links](#quick-links)
- [Features](#features)
- [HowTo (Manuals)](#how-to)
- [Warnings](#warning)
- [Common Issues](#common-issues)
- [**References**](documentation/references.md)
- [Contact](#contact-or-contribute)
- [License](LICENSE)


## Quick Links:
 - **Docker: https://hub.docker.com/r/dnhem/proj_deepmetab**
 - **Graphormer-RT Forked Submodule: [https://github.com/kbcoulter/Graphormer-RT](https://github.com/kbcoulter/Graphormer-RT/tree/e98e344e52301fcf73541ca2165fc89c0fbd2869)**
 - **RT-Zenodo: https://zenodo.org/records/15021743**

## Features

#### Below is an organized summary of the main capabilities provided by this repository:

| **Feature Category** | **Description** |
|----------------------|-----------------|
| **LC–MS Preprocessing** | Tools to clean, format, and structure LC–MS datasets for prediction |
| **Data Loaders** | Scripts to load, featurize, and register data  |
| **Workflow Setup** | Setup directory, container, etc. for RP/HILIC/Both RT prediction and annotation  |
| **Model Training** | Train Graphormer‑RT from scratch (RP) |
| **Model Finetuning** | Finetune Graphormer‑RT models (HILIC transfer learning and HILIC finetuning) |
| **RT Prediction** | Generate RT predictions and integrate results back into LC–MS feature tables |
| **Annotation Scoring Framework** | Score candidate molecules to resolve annotation ambiguities |
| **Quality Control** | Automatically flag potentially mis‑annotated mass feature IDs |
| **Stereoisomer Flagging** | Identify and label stereoisomer mass feature IDs |
| **LazyPredict ✨** | Run **entire** automated workflow |




## How To:

Choose the instructions based on your system permissions:


| Environment | Workflow Type | Documentation | Status |
|-------------|---------------|---------------|--------|
| **HPC / No Root Access** | Step-by-Step | [Manual Guide](documentation/app_manual.md) | ✅ |
| **HPC / No Root Access** | Automated | LazyPredict | 🚧 |
| **Local Machine / Root Access Available** | Step-by-Step | Root Manual Guide | 🚧 |
| **Local Machine / Root Access Available** | Automated | Root LazyPredict | 🚧 |


#### Support for non-HPC machines is in development and is expected soon...
---

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

> **Writing over Weights:**
> Finetuning HILIC will write over ```best_checkpoint.pt``` and ```last_checkpoint.pt``` should they exist in ```graphormer_checkpoints_HILIC/```.

## Common Issues:
* Fairseq Errors: Please ensure that you are using the Docker provided. The original [Graphormer](https://github.com/microsoft/Graphormer) was built on a Snapshot Release of [fairseq](https://github.com/facebookresearch/fairseq). This version is not available via conda/mamba. If error persists, please ensure that container version installation matches the most recent available on [Docker](https://hub.docker.com/r/dnhem/proj_deepmetab).

* Python Package Issues: Please first ensure that you are using the Docker provided. Reference previous fairseq issue for guidance. 
 
* **If your issue is not shown above, please [raise an issue](https://github.com/kbcoulter/deep_metab/issues/new).**



## References

#### **Please access our references [here](documentation/references.md)**

Special thanks to the Graphormer‑RT development team and to Maxine Wren and the BGMP instructional team for their mentorship. This work benefited from access to the University of Oregon HPC cluster, [Talapas](https://uoracs.github.io/talapas2-knowledge-base/).


## Contact or Contribute

If you have questions, feedback, or ideas, feel free to reach out to any of us:

- kcoulter [at] uoregon [dot] edu  
- dnhem [at] uoregon [dot] edu  
- ewi [at] uoregon [dot] edu  

---
### We welcome (and encourage) contributions!

If you find this project helpful or interesting, please consider **starring the repository**. Your support helps motivate continued development and improvements 🙂



