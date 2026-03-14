# Deep Metab Nextflow Pipeline Walkthrough

This document explains the full `next.nf` pipeline step by step, including what each parameter does, what each process produces, and how data flows through the workflow. 
> NOTE: Please ensure you have followed the first few steps outlined in [app_manual.md](./app_manual.md) up until **Setup**.

> NOTE: Please ensure you have nextflow installed! Follow the guide [here](./nextflow_installations.md)!

## Overview

This pipeline is a DSL2 Nextflow workflow for running the Deep Metab retention-time prediction pipeline. It performs the following major steps:

1. Aggregate raw LC-MS CSV files into a single input table.
2. Convert the aggregated table into Graphormer-ready input format.
3. Generate a metadata pickle file for Graphormer evaluation.
4. Run retention-time prediction through Graphormer using Apptainer.
5. Generate a stereochemistry-stripped reference CSV.
6. Evaluate the prediction output.
7. Apply calibration and scoring to produce final reports.

The workflow is designed so that each process writes its outputs into the task work directory first, and then `publishDir` copies the user-facing outputs into the desired output directory.

## Script header

```groovy
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2
```

* The shebang allows the script to be run as a Nextflow pipeline.
* `nextflow.enable.dsl = 2` enables DSL2 syntax.

## Parameters

The pipeline uses `params.*` syntax to define default values. These can be overridden from the command line.

### Core input parameters

```groovy
params.raw_lcms_dir = "${baseDir}/my_data/example_data/RP_example"
params.chrom_type = "rp"
params.setup_id = "0001"
```

* `raw_lcms_dir`: directory containing the raw LC-MS CSV files.
* `chrom_type`: chromatography type. Expected values are `rp` or `hilic`.
* `setup_id`: chromatography setup ID used in Graphormer input generation.

### Output and file structure parameters

```groovy
params.output_dir = "${baseDir}/my_data/example_data"
params.processed_data_dir = "${baseDir}/my_data/example_data/metadata_files"
params.graphormer_sif = "${baseDir}/graphormercontainer.sif"
```

* `output_dir`: where most user-facing outputs are copied.
* `processed_data_dir`: directory containing metadata files needed for pickle generation.
* `graphormer_sif`: path to the Apptainer image used for Graphormer prediction.

### LC-MS column settings

```groovy
params.mass_feature_id_col = "Mass Feature ID"
params.retention_time_col = "Retention Time (min)"
params.smiles_col = "smiles"
```

These tell the aggregation script which columns to use from the raw LC-MS CSV files.

### Optional knowns inputs

```groovy
params.knowns_file = null
params.knowns_preds_file = null
```

These are optional. If provided, they are passed to the scoring step. If they are `null`, the scoring process omits those arguments.

## Computed values

```groovy
def chrom_upper = params.chrom_type.toUpperCase()

params.checkpoint_dir = "${baseDir}/graphormer_checkpoints_${chrom_upper}"
params.predictions_dir = "${baseDir}/predictions_${chrom_upper}"
params.stereo_file = "${baseDir}/Destereo_Files/${params.chrom_type}_destereo.csv"
```

These values are derived from `chrom_type`.

* `chrom_upper`: converts `rp` to `RP` and `hilic` to `HILIC`.
* `checkpoint_dir`: Graphormer checkpoint directory for the chosen chromatography type.
* `predictions_dir`: final publish directory for prediction outputs.
* `stereo_file`: expected stereochemistry-stripped reference file path.

## Process 1: `aggregateCSVs`

This process combines all raw LC-MS CSV files into a single aggregated CSV.

### Input

```groovy
input:
path raw_data_dir
```

This receives the raw LC-MS directory as a path channel.

### Output

```groovy
output:
path "aggregated_${params.chrom_type.toUpperCase()}.csv"
```

For RP, this becomes `aggregated_RP.csv`.

### Script behavior

It calls:

```bash
python data_prep/process_lcms_data/initiate_dataset.py
```

This script:

* scans the raw data folder,
* extracts the configured mass feature, retention time, and SMILES columns,
* writes a unified aggregated CSV.

### Published output

The resulting aggregated CSV is copied to `params.output_dir`.

## Process 2: `buildGraphormerInput`

This process converts the aggregated CSV into the format required for Graphormer evaluation.

### Input

```groovy
input:
path aggregated_csv
```

### Output

```groovy
output:
path "graphormer_input_${chrom_upper}.csv"
```

For RP, this becomes `graphormer_input_RP.csv`.

### Script behavior

It calls:

```bash
python data_prep/process_lcms_data/initiate_input_for_graphormer_eval.py
```

This script:

* reads the aggregated CSV,
* reformats the data into Graphormer-compatible input,
* attaches the chromatography setup ID.

## Process 3: `generateMetadata`

This process creates the metadata pickle required by Graphormer.

### Input

```groovy
input:
path processed_data_dir
```

This is a directory containing the metadata files.

### Output

```groovy
output:
path "${chrom_upper}_metadata.pickle"
```

For RP, this becomes `RP_metadata.pickle`.

### Script behavior

It calls:

```bash
python data_prep/process_experimental_conditions/gen_pickle_dict.py
```

This script uses:

* the metadata directory,
* the Tanaka database TSV,
* the HSMB database TSV,
* and writes a pickle file into the local task directory.

The `-o .` is important because Nextflow expects outputs to be created inside the task work directory.

## Process 4: `makePredictions`

This is the Graphormer evaluation step.

### Inputs

```groovy
input:
path graphormer_input
path metadata_pickle
```

It receives:

* the Graphormer input CSV,
* the metadata pickle.

### Output

```groovy
output:
path "${params.chrom_type.toUpperCase()}_predictions.csv"
```

For RP, this becomes `RP_predictions.csv`.

### Script behavior

This process:

1. Creates a temporary input folder called `temp_data`.
2. Copies the Graphormer input CSV and metadata pickle into it.
3. Calls a chromatography-specific wrapper script:

   * `app_evaluate_RP.sh` for RP
   * `app_evaluate_HILIC.sh` for HILIC
4. The wrapper script launches Graphormer through Apptainer.
5. The resulting prediction CSV is found in a local `predictions` directory.
6. That file is renamed to a stable final name like `RP_predictions.csv`.

### Important design notes

* The process creates outputs in the local task directory first.
* `publishDir` then copies the final prediction CSV to `params.predictions_dir`.
* This step is configured to run on Slurm using the `makePredictions` process settings in `nextflow.config`.

## Process 5: `stripStereochemistry`

This process removes stereochemistry information from the aggregated CSV to create a reference file.

### Input

```groovy
input:
path aggregated_csv
```

### Output

```groovy
output:
path "${params.chrom_type.toUpperCase()}_destereo.csv"
```

For RP, this becomes `RP_destereo.csv`.

### Script behavior

It calls:

```bash
python data_prep/process_lcms_data/rm_stereochemistry.py
```

This script strips stereochemistry using the aggregated CSV and writes the destereo output.

## Process 6: `evaluatePredictions`

This process generates a quick prediction report.

### Input

```groovy
input:
path predictions_csv
```

### Output

```groovy
output:
path "prediction_report.txt"
```

### Script behavior

It calls:

```bash
python data_prep/eval_rt_preds/analyze_predictions.py
```

This script analyzes the prediction output and writes a simple text report.

## Process 7: `applyScoring`

This process performs calibration and scoring using the raw LC-MS data, predictions, and destereo reference file.

### Inputs

```groovy
input:
path raw_lcms_dir
path predictions_csv
path stereo_file
```

### Output

```groovy
output:
path "*_Calibration_&_Scoring/**"
```

This captures the full calibration and scoring output directory.

### Optional arguments

Inside the process script, optional flags are added only if the corresponding params exist:

```groovy
def knowns_flag = params.knowns_file ? "--knowns ${params.knowns_file}" : ""
def knowns_preds_flag = params.knowns_preds_file ? "--knowns_preds ${params.knowns_preds_file}" : ""
def rt_filter_flag = (params.knowns_file && params.chrom_type == "hilic") ? "--rt_filter" : ""
```

This avoids passing `null` channels through the workflow.

### Script behavior

It calls:

```bash
python data_prep/eval_rt_preds/filter_and_scoring.py
```

This script:

* reads the raw LC-MS files,
* uses the prediction CSV,
* uses the destereo reference,
* optionally incorporates knowns files,
* writes the calibration and scoring directory.

## Workflow section

The workflow defines the dataflow between all processes.

```groovy
workflow {
    raw_lcms = Channel.fromPath(params.raw_lcms_dir, checkIfExists: true)
    aggregated = aggregateCSVs(raw_lcms)

    processed_dir = Channel.fromPath(params.processed_data_dir, checkIfExists: true)
    metadata = generateMetadata(processed_dir)

    println "Aggregated CSV: ${aggregated}"
    println "Metadata Pickle: ${metadata}"

    graphormer_input = buildGraphormerInput(aggregated)
    println "Graphormer Input CSV: ${graphormer_input}"

    predictions = makePredictions(graphormer_input, metadata)
    destereo = stripStereochemistry(aggregated)
    eval_report = evaluatePredictions(predictions)

    final_scores = applyScoring(
        raw_lcms,
        predictions,
        destereo
    )
}
```

### Dataflow summary

The workflow graph is:

* `raw_lcms` -> `aggregateCSVs` -> `aggregated`
* `aggregated` -> `buildGraphormerInput` -> `graphormer_input`
* `processed_dir` -> `generateMetadata` -> `metadata`
* `graphormer_input` + `metadata` -> `makePredictions` -> `predictions`
* `aggregated` -> `stripStereochemistry` -> `destereo`
* `predictions` -> `evaluatePredictions` -> `eval_report`
* `raw_lcms` + `predictions` + `destereo` -> `applyScoring` -> `final_scores`

## Running the pipeline

Basic run:

```bash
nextflow run test_flow.nf
```

Override parameters on the command line:

```bash
nextflow run test_flow.nf \
  --raw_lcms_dir /path/to/raw_data \
  --chrom_type rp \
  --setup_id 0001
```

## Notes on `nextflow.config`

The associated `nextflow.config` is important for resource configuration, especially for `makePredictions`.

Typical behavior is:

* most preprocessing steps run locally,
* `makePredictions` runs on Slurm with GPU resources,
* account and partition settings are defined there.

## Common debugging lessons from this pipeline

1. Process outputs should be written into the local task work directory.
2. `publishDir` should be used to expose outputs externally.
3. Apptainer paths inside the container should use absolute bound mount targets like `/predictions`.
4. Host-side save paths should stay relative or absolute to the task directory, not root-level paths like `/predictions`.
5. If a file exists in a nested output folder, the `output:` pattern must match that nested structure.
6. Optional CLI arguments are cleaner when handled inside the process script instead of passing `null` channels.

## Final expected outputs

Depending on chromatography type (for this example, we're using RP), the pipeline should produce files such as:

* `aggregated_RP.csv`
* `graphormer_input_RP.csv`
* `RP_metadata.pickle`
* `RP_predictions.csv`
* `RP_destereo.csv`
* `prediction_report.txt`
* `rp_Calibration_&_Scoring/` directory

For HILIC, these names change accordingly.
