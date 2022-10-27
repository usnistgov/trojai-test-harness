# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import copy
import pandas as pd
import itertools


def build_model_lists(metadata_df: pd.Dataframe, columns_of_interest: list) -> dict:
    model_lists = {}
    column_variations = {}

    if len(columns_of_interest) == 0 or 'all' in columns_of_interest:
        model_ids = metadata_df['model_name'].tolist()
        model_lists['all'] = model_ids
        temp_columns_of_interest = copy.deepcopy(columns_of_interest)
        temp_columns_of_interest.remove('all')
    else:
        temp_columns_of_interest = columns_of_interest

    # Gather unique names in columns of interest
    for column_name in temp_columns_of_interest:
        unique_values_in_column = metadata_df[column_name].unique()
        if len(unique_values_in_column) > 0:
            column_variations[column_name] = unique_values_in_column

    # Remove instances of nan/null
    for column_variation in column_variations.keys():
        column_variations[column_variation] = [v for v in column_variations[column_variation] if
                                               not (pd.isnull(v))]

    # Create permutations of columns of interest
    keys, values = zip(*column_variations.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    removal_list = []

    # Generate lists of models
    for i, permutation in enumerate(permutations_dicts):
        subset_df = metadata_df
        index = ''
        for key, value in permutation.items():
            if index == '':
                index = value
            else:
                index += ':' + value
            subset_df = subset_df[subset_df[key] == value]

        # Output the list of models that meet this requirement
        model_ids = subset_df['model_name'].tolist()

        if len(model_ids) == 0:
            removal_list.append(index)

        model_lists[index] = model_ids

    for index in sorted(removal_list, reverse=True):
        del model_lists[index]

    return model_lists
