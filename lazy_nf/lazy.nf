// main.nf - Deep Metab Pipeline using Nextflow DSL2

nextflow.enable.dsl = 2

// PARAMETERS - Set defaults here, override with --param value on CLI

params {
    // baseDir -> current working directory, used for relative paths
    // Data directories
    raw_lcms_dir = "${baseDir}/my_data/all_RP_raw_data"
    chrom_type = "rp"  // "rp" or "hilic"
    setup_id = "setup_001"
    
    // Output directories
    output_dir = "${baseDir}/results"
    
    // Column specifications (matching LC-MS CSV structure)
    mass_feature_id_col = "Mass Feature ID"
    retention_time_col = "Retention Time (min)"
    smiles_col = "smiles"
    
    // Prediction settings
    checkpoint_dir = "${baseDir}/graphormer_checkpoints_${params.chrom_type.toUpperCase()}"
    predictions_dir = "${baseDir}/predictions_${params.chrom_type.toUpperCase()}"
    
    // Scoring settings
    stereo_file = "${baseDir}/Destereo_Files/${params.chrom_type}_destereo.csv"
    
    // Optional knowns for RT filtering (HILIC only)
    knowns_file = null
    knowns_preds_file = null
}

// PROCESS 1: Aggregate LC-MS CSVs
process aggregateCSVs {
    publishDir params.output_dir, mode: 'copy', overwrite: true
    
    input:
    path raw_data_dir
    
    output:
    path "aggregated.csv"
    
    script:
    """
    python ${baseDir}/data_prep/process_lcms_data/initiate_dataset.py \\
        --dataset_folder ${raw_data_dir} \\
        --output_csv aggregated.csv \\
        --mass_feature_id_col "${params.mass_feature_id_col}" \\
        --retention_time_col "${params.retention_time_col}" \\
        --smiles_col "${params.smiles_col}"
    """
}

// PROCESS 2: Build Graphormer Input (add setup_id, average RT per SMILES)
process buildGraphormerInput {
    publishDir params.output_dir, mode: 'copy', overwrite: true
    
    input:
    path aggregated_csv
    
    output:
    path "graphormer_input.csv"
    
    script:
    """
    python ${baseDir}/data_prep/process_lcms_data/initiate_input_for_graphormer_eval.py \\
        -i ${aggregated_csv} \\
        -o graphormer_input.csv \\
        -s ${params.setup_id}
    """
}

// PROCESS 3: Generate Metadata Pickle
process generateMetadata {
    publishDir params.output_dir, mode: 'copy', overwrite: true
    
    input:
    path processed_data_dir
    
    output:
    path "metadata.pickle"
    
    script:
    """
    python ${baseDir}/data_prep/process_experimental_conditions/gen_pickle_dict.py \\
        --data_directory ${processed_data_dir} \\
        -o . \\
        -of metadata.pickle
    """
}

// PROCESS 4: Make RT Predictions (RP or HILIC)
process makePredictions {
    publishDir params.predictions_dir, mode: 'copy', overwrite: true
    
    input:
    path graphormer_input
    path metadata_pickle
    
    output:
    path "${params.chrom_type}_predictions.csv"
    
    script:
    def chrom_upper = params.chrom_type.toUpperCase()
    def script_name = params.chrom_type == "rp" ? "app_evaluate_RP.sh" : "app_evaluate_HILIC.sh"
    
    """
    # Create a temporary data directory
    mkdir -p temp_data
    cp ${graphormer_input} temp_data/
    cp ${metadata_pickle} temp_data/
    
    # Run the prediction script directly (Nextflow handles SLURM submission)
    bash ${baseDir}/make_predictions/${script_name} \\
        --host-data-dir temp_data/ \\
        --checkpoint-dir ${params.checkpoint_dir} \\
        --save-path ./predictions
    
    # Collect predictions
    mv predictions/${params.chrom_type}_preds.csv ${params.chrom_type}_predictions.csv
    """
}
 
// PROCESS 5: Strip Stereochemistry Reference (optional but recommended)
process stripStereochemistry {
    publishDir params.output_dir, mode: 'copy', overwrite: true
    
    input:
    path aggregated_csv
    
    output:
    path "${params.chrom_type}_destereo.csv"
    
    script:
    """
    python ${baseDir}/data_prep/process_lcms_data/rm_stereochemistry.py \\
        -i ${aggregated_csv} \\
        -o ${params.chrom_type}_destereo.csv \\
        --smiles-col-index 3 \\
        --has-header
    """
}

 
// PROCESS 6: Evaluate Predictions (Quick Check)
process evaluatePredictions {
    publishDir params.output_dir, mode: 'copy', overwrite: true
    
    input:
    path predictions_csv
    
    output:
    path "prediction_report.txt"
    
    script:
    """
    python ${baseDir}/data_prep/evaluate_rt_preds/analyze_predictions.py \\
        -i ${predictions_csv} \\
        -o prediction_report.txt
    """
}

 
// PROCESS 7: Apply Scoring Framework & RT Filtering
process applyScoring {
    publishDir params.output_dir, mode: 'copy', overwrite: true
    
    input:
    path raw_lcms_dir
    path predictions_csv
    path stereo_file
    path knowns_file
    path knowns_preds_file
    
    output:
    path "*_stats_report.txt"
    path "*_Calibration_*/**"
    
    script:
    def knowns_flag = knowns_file ? "--knowns ${knowns_file}" : ""
    def knowns_preds_flag = knowns_preds_file ? "--knowns_preds ${knowns_preds_file}" : ""
    def rt_filter_flag = (knowns_file && params.chrom_type == "hilic") ? "--rt_filter" : ""
    
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

 
// WORKFLOW: Chain processes together
 
workflow {
    
    // INPUT CHANNELS
    raw_lcms = channel.fromPath(params.raw_lcms_dir, type: 'dir')
    
    // Step 1: Aggregate all LC-MS CSVs
    aggregated = aggregateCSVs(raw_lcms)
    
    // Step 2: Build Graphormer input
    graphormer_input = buildGraphormerInput(aggregated)
    
    // Step 3: Generate metadata pickle 
    // metadata = generateMetadata(raw_lcms)
    
    // For now, assume metadata.pickle exists
    metadata = channel.fromPath("${params.output_dir}/metadata.pickle")
    
    // Step 4: Make predictions
    predictions = makePredictions(graphormer_input, metadata)
    
    // Step 5: Strip stereochemistry (creates reference file)
    destereo = stripStereochemistry(aggregated)
    
    // Step 6: Quick evaluation (optional)
    eval = evaluatePredictions(predictions)
    
    // Step 7: Apply scoring framework & RT filtering
    final_scores = applyScoring(
        raw_lcms,
        predictions,
        destereo,
        params.knowns_file ? channel.fromPath(params.knowns_file) : null,
        params.knowns_preds_file ? channel.fromPath(params.knowns_preds_file) : null
    )
}