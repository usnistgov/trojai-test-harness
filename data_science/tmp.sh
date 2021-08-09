
ROUND_NAME=round5

# *******************************
# ES
# *******************************

# TEST_HARNESS_DIR is the location of a copy of the data from the test server. This should contain the subfolders for each queue on the test server
TEST_HARNESS_DIR=/home/mmajurski/Downloads/peter/
# DATA_DIR is the directory containing the dataset. I.e. the test dataset if working against the ES queue on the test server
DATA_DIR=/mnt/scratch/trojai/data/round6/round6-test-dataset
# OUTPUT_DIR this is the parent output directory all plots and csv files will be saved under
OUTPUT_DIR=/home/mmajurski/Downloads/peter/data-science

export PYTHONPATH="$PYTHONPATH:/home/mmajurski/usnistgov/trojai-test-harness/"

#echo "Building test global csv results file from test harness directories and metadata file"
#python compile_global_csv_results.py --test-harness-dirpath=${TEST_HARNESS_DIR} --server=es --metadata-filepath=${DATA_DIR}/METADATA.csv --output-dirpath=${OUTPUT_DIR}

echo "Plotting mean effects for every other column against the selected one"
python plot_mean_effects.py --global-results-csv-filepath=${OUTPUT_DIR}/es-global-results-test.csv --metric="cross_entropy" --output-dirpath=${OUTPUT_DIR}/mean-effects-plots-test


python plot_mean_effects.py --global-results-csv-filepath=${OUTPUT_DIR}/es-global-results-train.csv --metric="cross_entropy" --output-dirpath=${OUTPUT_DIR}/mean-effects-plots-train




