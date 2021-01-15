import os
import numpy as np
import sklearn.metrics


from actor_executor import metrics


def process_results(output_directory, results_directory, ground_truth_directory, team_name, machine):
    results_directory = os.path.join(results_directory, machine, team_name)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Process results
    if os.path.isdir(results_directory):
        targets_list = list()
        predictions_list = list()

        models = [fn for fn in os.listdir(ground_truth_directory) if fn.startswith('id-')]
        models.sort()

        for model in models:
            result_filename = '{}.txt'.format(model)

            model_name = result_filename.split('.')[0]
            result_filepath = os.path.join(results_directory, result_filename)
            ground_truth_filepath = os.path.join(ground_truth_directory, model_name, 'ground_truth.csv')
            if not os.path.exists(ground_truth_filepath):
                raise RuntimeError('Missing ground truth file: {}'.format(ground_truth_filepath))

            with open(ground_truth_filepath) as truth_file:
                file_contents = truth_file.readline().strip()
                ground_truth = float(file_contents)
                targets_list.append(ground_truth)

            if os.path.exists(result_filepath):
                with open(result_filepath) as result_file:
                    file_contents = result_file.readline().strip()
                    result = float(file_contents)
                    predictions_list.append(result)
            else:
                predictions_list.append(np.nan)

        predictions = np.array(predictions_list).reshape(-1, 1)
        targets = np.array(targets_list).reshape(-1, 1)

        if np.all(np.isnan(predictions)):
            # do nothing to record those which completly failed
            pass
        else:
            # if prediction is nan, then replace with guess (0.5)
            predictions[np.isnan(predictions)] = 0.5

        elementwise_cross_entropy = metrics.elementwise_binary_cross_entropy(predictions, targets)
        ce = float(np.mean(elementwise_cross_entropy))
        ce_95_ci = metrics.cross_entropy_confidence_interval(elementwise_cross_entropy)
        brier_score = metrics.binary_brier_score(predictions, targets)

        TP_counts, FP_counts, FN_counts, TN_counts, TPR, FPR, thresholds = metrics.confusion_matrix(targets, predictions)
        roc_auc = float(sklearn.metrics.auc(FPR, TPR))

        header = 'model name, target, prediction, cross entropy\n'
        with open(os.path.join(output_directory, '{}-{}-elementwise_ce.csv'.format(team_name, machine)), "w") as output_file:
            output_file.write(header)

            for i in range(len(models)):
                new_line = "{}, {}, {}, {}\n".format(models[i], targets_list[i], predictions_list[i], float(elementwise_cross_entropy[i]))
                output_file.write(new_line)

        if not os.path.exists(os.path.join(output_directory, 'metrics.csv')):
            header = 'machine, team name, cross entropy, cross entropy 95% CI, brier score, roc-auc\n'
            with open(os.path.join(output_directory, 'metrics.csv'), "w") as output_file:
                output_file.write(header)

        with open(os.path.join(output_directory, 'metrics.csv'), "a") as output_file:
            new_line = "{}, {}, {}, {}, {}, {}\n".format(machine, team_name, ce, ce_95_ci, brier_score, roc_auc)
            output_file.write(new_line)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--ground-truth-directory', type=str,
                        help='Path to ground truth (if not set then will use models_directory)')
    parser.add_argument('--output-directory', type=str,
                        help='The output file for outputting CSV results')
    parser.add_argument('--results-directory', type=str,
                        help='Path to the results directory')
    parser.add_argument('--team-name', type=str, default=None)
    parser.add_argument('--machine', type=str, default=None)

    # add team name and container name

    args = parser.parse_args()

    # Load parameters
    output_directory = args.output_directory
    results_directory = args.results_directory
    ground_truth_directory = args.ground_truth_directory
    team_name = args.team_name
    machine = args.machine

    # process_results(output_directory, results_directory, ground_truth_directory, team_name, machine)

    machines = ['laura', 'a100', 'v100', 'threadripper']

    containers = ['ARM-UCSD-20201114T063002', 'ICSI-20201107T083001', 'Perspecta-PurdueRutgers-20201113T063001', 'PL-GIFT-20201202T141001', 'PL-GIFT-20201207T160002', 'TrinitySRITrojAI-20201115T100001']

    for container in containers:
        for machine in machines:
            process_results(output_directory, results_directory, ground_truth_directory, container, machine)