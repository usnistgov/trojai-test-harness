# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


clear
ROUND_NAME=round6

# *******************************
# ES
# *******************************

# TEST_HARNESS_DIR is the location of a copy of the data from the test server. This should contain the subfolders for each queue on the test server
TEST_HARNESS_DIR=/mnt/scratch/trojai/v100/${ROUND_NAME}
# DATA_DIR is the directory containing the dataset. I.e. the test dataset if working against the ES queue on the test server
DATA_DIR=/mnt/scratch/trojai/data/${ROUND_NAME}/${ROUND_NAME}-test-dataset
# OUTPUT_DIR this is the parent output directory all plots and csv files will be saved under
OUTPUT_DIR=/mnt/scratch/trojai/v100/${ROUND_NAME}/es/data-science

export PYTHONPATH="$PYTHONPATH:/home/mmajurski/usnistgov/trojai-test-harness/"

echo "Building test global csv results file from test harness directories and metadata file"
python compile_global_csv_results.py --test-harness-dirpath=${TEST_HARNESS_DIR} --server=es --metadata-filepath=${DATA_DIR}/METADATA.csv --output-dirpath=${OUTPUT_DIR}

echo "Building leaderboard archive csv file"
python build_leaderboard_archive.py --global-results-csv-filepath=${OUTPUT_DIR}/es-global-results.csv --queue=es --output-dirpath=${OUTPUT_DIR}

echo "Plotting CE and ROC-AUC as a function of trojan percentage sweep"
python plot_trojan_percentage_sweep.py --global-results-csv-filepath=${OUTPUT_DIR}/es-global-results.csv --nb-reps=10 --output-dirpath=${OUTPUT_DIR}/trojan-percentage

echo "Plotting ROC Curves"
python plot_roc_curve.py --global-results-csv-filepath=${OUTPUT_DIR}/es-global-results.csv --output-dirpath=${OUTPUT_DIR}/roc-curves

echo "Plotting Per-Model CE Histograms"
python plot_per_model_ce_histogram.py --global-results-csv-filepath=${OUTPUT_DIR}/es-global-results.csv --output-dirpath=${OUTPUT_DIR}/ce-hist

echo "Plotting every other column against the selected one"
python plot_features.py --global-results-csv-filepath=${OUTPUT_DIR}/es-global-results.csv --metric="cross_entropy" --output-dirpath=${OUTPUT_DIR}/feature-plots

echo "Plotting mean effects for every other column against the selected one"
python plot_mean_effects.py --global-results-csv-filepath=${OUTPUT_DIR}/es-global-results.csv --metric="cross_entropy" --output-dirpath=${OUTPUT_DIR}/mean-effects-plots #--box-plot






# *******************************
# Holdout
# *******************************

# TEST_HARNESS_DIR is the location of a copy of the data from the test server. This should contain the subfolders for each queue on the test server
TEST_HARNESS_DIR=/mnt/scratch/trojai/v100/${ROUND_NAME}
# DATA_DIR is the directory containing the dataset. I.e. the test dataset if working against the ES queue on the test server
DATA_DIR=/mnt/scratch/trojai/data/${ROUND_NAME}/${ROUND_NAME}-holdout-dataset
# OUTPUT_DIR this is the parent output directory all plots and csv files will be saved under
OUTPUT_DIR=/mnt/scratch/trojai/v100/${ROUND_NAME}/holdout/data-science

export PYTHONPATH="$PYTHONPATH:/home/mmajurski/usnistgov/trojai-test-harness/"

echo "Building test global csv results file from test harness directories and metadata file"
python compile_global_csv_results.py --test-harness-dirpath=${TEST_HARNESS_DIR} --server=holdout --metadata-filepath=${DATA_DIR}/METADATA.csv --output-dirpath=${OUTPUT_DIR}

echo "Building leaderboard archive csv file"
python build_leaderboard_archive.py --global-results-csv-filepath=${OUTPUT_DIR}/holdout-global-results.csv --queue=holdout --output-dirpath=${OUTPUT_DIR}







