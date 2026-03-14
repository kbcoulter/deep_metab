#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

// PARAMETER

params.raw_lcms_dir         = "${baseDir}/my_data/example_data/RP_example"
params.chrom_type           = "rp"              // "rp" or "hilic"
params.setup_id             = "0001" // change to your chromotography setup ID

params.output_dir           = "${baseDir}/my_data/example_data" // change to desire output

params.mass_feature_id_col  = "Mass Feature ID"
params.retention_time_col   = "Retention Time (min)"
params.smiles_col           = "smiles"

params.processed_data_dir = "${baseDir}/my_data/example_data/metadata_files" // change to metadata file directory

params.knowns_file          = null
params.knowns_preds_file    = null


params.graphormer_sif = "${baseDir}/graphormercontainer.sif"

// =========================
// COMPUTED VALUES
def chrom_upper = params.chrom_type.toUpperCase()

params.checkpoint_dir = "${baseDir}/graphormer_checkpoints_${chrom_upper}"
params.predictions_dir = "${baseDir}/predictions_${chrom_upper}"
params.stereo_file = "${baseDir}/Destereo_Files/${params.chrom_type}_destereo.csv"


// PROCESS 1: Aggregate LC-MS CSVs
process aggregateCSVs {

    publishDir params.output_dir, mode: 'copy', overwrite: true

    input:
    path raw_data_dir

    output:
    path "aggregated_${params.chrom_type.toUpperCase()}.csv"

    script:

    """
    python ${baseDir}/data_prep/process_lcms_data/initiate_dataset.py \\
        --dataset_folder ${raw_data_dir} \\
        --output_csv aggregated_${chrom_upper}.csv \\
        --mass_feature_id_col "${params.mass_feature_id_col}" \\
        --retention_time_col "${params.retention_time_col}" \\
        --smiles_col "${params.smiles_col}"
    """
}


// PROCESS 2: Build Graphormer Input
process buildGraphormerInput {

    publishDir params.output_dir, mode: 'copy', overwrite: true

    input:
    path aggregated_csv

    output:
    path "graphormer_input_${chrom_upper}.csv"

    script:
    """
    python ${baseDir}/data_prep/process_lcms_data/initiate_input_for_graphormer_eval.py \
        -i ${aggregated_csv} \
        -o graphormer_input_${chrom_upper}.csv \
        -s ${params.setup_id}
    """
}


// PROCESS 3: Generate Metadata Pickle
process generateMetadata {

    publishDir params.output_dir, mode: 'copy', overwrite: true

    input:
    path processed_data_dir

    output:
    path "${chrom_upper}_metadata.pickle"

    script:
    """
    python ${baseDir}/data_prep/process_experimental_conditions/gen_pickle_dict.py \
        --data_directory ${processed_data_dir} \
        --tanaka ${baseDir}/my_data/example_data/tanaka_database.tsv \
        --hsmb ${baseDir}/my_data/example_data/hsmb_database.tsv \
        -o . \
        -of ${chrom_upper}_metadata.pickle
    """
}


// PROCESS 4: Make RT Predictions
process makePredictions {

    publishDir params.predictions_dir, mode: 'copy', overwrite: true

    input:
    path graphormer_input
    path metadata_pickle

    output:
    path "${params.chrom_type.toUpperCase()}_predictions.csv"

    script:
    def script_name = params.chrom_type == "rp" ? "app_evaluate_RP.sh" : "app_evaluate_HILIC.sh"

    """
    mkdir -p temp_data
    mkdir -p predictions

    cp ${graphormer_input} temp_data/
    cp ${metadata_pickle} temp_data/

    bash ${baseDir}/make_predictions/${script_name} \\
        --host-data-dir temp_data/ \\
        --checkpoint-dir ${params.checkpoint_dir} \\
        --save-path ./predictions \\
        --container-path ${params.graphormer_sif}

    pred_file=\$(find predictions -maxdepth 1 -name "*.csv" -print -quit)

    if [[ -z "\$pred_file" ]]; then
        echo "Error: no prediction CSV found in predictions/"
        exit 1
    fi

    mv "\$pred_file" ${chrom_upper}_predictions.csv
    """
}

// PROCESS 5: Strip Stereochemistry Reference
process stripStereochemistry {

    publishDir params.output_dir, mode: 'copy', overwrite: true

    input:
    path aggregated_csv

    output:
    path "${params.chrom_type.toUpperCase()}_destereo.csv"

    script:
    """
    python ${baseDir}/data_prep/process_lcms_data/rm_stereochemistry.py \\
        -i ${aggregated_csv} \\
        -o ${chrom_upper}_destereo.csv \\
        --smiles-col-index 3 \\
        --has-header
    """
}


// PROCESS 6: Evaluate Predictions
process evaluatePredictions {

    publishDir params.output_dir, mode: 'copy', overwrite: true

    input:
    path predictions_csv

    output:
    path "prediction_report.txt"

    script:
    """
    python ${baseDir}/data_prep/eval_rt_preds/analyze_predictions.py \
        -i ${predictions_csv} \
        -o prediction_report.txt
    """
}


// PROCESS 7: Apply Scoring
process applyScoring {

    publishDir params.output_dir, mode: 'copy', overwrite: true

    input:
    path raw_lcms_dir
    path predictions_csv
    path stereo_file

    output:
    path "*_Calibration_&_Scoring/**"

    script:
    def knowns_flag = params.knowns_file ? "--knowns ${params.knowns_file}" : ""
    def knowns_preds_flag = params.knowns_preds_file ? "--knowns_preds ${params.knowns_preds_file}" : ""
    def rt_filter_flag = (params.knowns_file && params.chrom_type == "hilic") ? "--rt_filter" : ""

    """
    python ${baseDir}/data_prep/eval_rt_preds/filter_and_scoring.py \\
        -i ${raw_lcms_dir} \\
        -p ${predictions_csv} \\
        -t ${params.chrom_type} \\
        -a \\
        --save \\
        --stereo_file ${stereo_file} \\
        --flag_stereo \\
        -s \\
        ${knowns_flag} \\
        ${knowns_preds_flag} \\
        ${rt_filter_flag}
    """
}

// WORKFLOW
workflow {

    /*
     * Create input channel from the raw LC-MS directory.
     * checkIfExists helps fail early if the path is wrong.
     */
    raw_lcms = Channel.fromPath(params.raw_lcms_dir, checkIfExists: true)

    /*
     * Step 1: Aggregate all LC-MS CSVs
     */
    aggregated = aggregateCSVs(raw_lcms)

    // for the metadata
    processed_dir = Channel.fromPath(params.processed_data_dir, checkIfExists: true)

    metadata = generateMetadata(processed_dir)

    println "Aggregated CSV: ${aggregated}"
    println "Metadata Pickle: ${metadata}"

    graphormer_input = buildGraphormerInput(aggregated)

    println "Graphormer Input CSV: ${graphormer_input}"

    // If metadata.pickle already exists:
    // metadata = Channel.fromPath("${params.output_dir}/metadata.pickle", checkIfExists: true)

    predictions = makePredictions(graphormer_input, metadata)

    destereo = stripStereochemistry(aggregated)

    eval_report = evaluatePredictions(predictions)

    final_scores = applyScoring(
        raw_lcms,
        predictions,
        destereo
    )
}