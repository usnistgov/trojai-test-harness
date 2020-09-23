# *******************************
# ES
# *******************************

# TEST_HARNESS_DIR is the location of a copy of the data from the test server. This should contain the subfolders for each queue on the test server
TEST_HARNESS_DIR=/mnt/scratch/trojai/v100/round2
# DATA_DIR is the directory containing the dataset. I.e. the test dataset if working against the ES queue on the test server
DATA_DIR=/mnt/scratch/trojai/data/round2/round2-test-dataset
# OUTPUT_DIR this is the parent output directory all plots and csv files will be saved under
OUTPUT_DIR=/mnt/scratch/trojai/v100/round2/es/data-science

export PYTHONPATH="$PYTHONPATH:/home/mmajurski/usnistgov/trojai-test-harness/"

#echo "Building test global csv results file from test harness directories and metadata file"
#python compile_global_csv_results.py --test-harness-dirpath=${TEST_HARNESS_DIR} --server=es --metadata-filepath=${DATA_DIR}/METADATA.csv --output-dirpath=${OUTPUT_DIR}
#
#echo "Building leaderboard archive csv file"
#python build_leaderboard_archive.py --global-results-csv-filepath=${OUTPUT_DIR}/es-global-results.csv --queue=es --output-dirpath=${OUTPUT_DIR}
#
#echo "Plotting CE and ROC-AUC as a function of trojan percentage sweep"
#python plot_trojan_percentage_sweep.py --global-results-csv-filepath=${OUTPUT_DIR}/es-global-results.csv --nb-reps=10 --output-dirpath=${OUTPUT_DIR}/trojan-percentage
#
#echo "Plotting ROC Curves"
#python plot_roc_curve.py --global-results-csv-filepath=${OUTPUT_DIR}/es-global-results.csv --output-dirpath=${OUTPUT_DIR}/roc-curves
#
#echo "Plotting Per-Model CE Histograms"
#python plot_per_model_ce_histogram.py --global-results-csv-filepath=${OUTPUT_DIR}/es-global-results.csv --output-dirpath=${OUTPUT_DIR}/ce-hist
#
#echo "Plotting experimental design factors"
#python plot_dex_factors.py --global-results-csv-filepath=${OUTPUT_DIR}/es-global-results.csv --metric="cross_entropy" --output-dirpath=${OUTPUT_DIR}/dex-plots
#
#echo "Plotting every other column against the selected one"
#python plot_features.py --global-results-csv-filepath=${OUTPUT_DIR}/es-global-results.csv --metric="cross_entropy" --output-dirpath=${OUTPUT_DIR}/feature-plots
#
#echo "Plotting mean effects for every other column against the selected one"
#python plot_mean_effects.py --global-results-csv-filepath=${OUTPUT_DIR}/es-global-results.csv --metric="cross_entropy" --output-dirpath=${OUTPUT_DIR}/mean-effects-plots #--box-plot
#
#echo "Converting global results csv into factor levels"
#python compile_factor_data.py --global-results-csv-filepath=${OUTPUT_DIR}/es-global-results.csv --output-dirpath=${OUTPUT_DIR}







