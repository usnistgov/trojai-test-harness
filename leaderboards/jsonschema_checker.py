import jsonschema
import numpy as np
import random
import copy
import json
import sys

from hypothesis_jsonschema import _canonicalise
from hypothesis_jsonschema import _resolve

from spython.main import Client as client

num_issues = 0
fatal_issues = 0

def get_value(param_name, canonicalized_param, type_name, base_value, variation_amount, random_variation_sign, random_from_min_max, print_warnings):
    global num_issues
    if random_variation_sign:
        result = np.random.binomial(1, 0.5)
        if result == 1:
            sign = 1.0
        else:
            sign = -1.0
    else:
        sign = 1.0

    if 'enum' in canonicalized_param:
        # sample from all enums
        enum_param = canonicalized_param['enum']
        return random.choice(enum_param)
    elif 'const' in canonicalized_param:
        return canonicalized_param['const']
    elif 'oneOf' in canonicalized_param:
        if isinstance(base_value, dict):
            if 'name' in base_value:
                oneOf_name = base_value['name']

                for canon_param in canonicalized_param['oneOf']:
                    if isinstance(canon_param, dict):
                        if 'name' in canon_param:
                            if oneOf_name == canon_param['name']:
                                value = build_json_config(canon_param, base_value, print_warnings=print_warnings)
                                return value

        selected_param = random.choice(canonicalized_param['oneOf'])
        value = build_json_config(selected_param, base_value, print_warnings=print_warnings)

        return value
    elif 'number' in type_name:
        min_value, max_value, _, _ = _canonicalise.get_number_bounds(canonicalized_param)

        if 'suggested_minimum' in canonicalized_param:
            min_value = canonicalized_param['suggested_minimum']

        if 'suggested_maximum' in canonicalized_param:
            max_value = canonicalized_param['suggested_maximum']

        if min_value is None:
            if print_warnings:
                print('Warning: min value or suggested_minimum for param "{}" is missing.'.format(param_name))
            num_issues += 1
            min_value = sys.float_info.min
        if max_value is None:
            if print_warnings:
                print('Warning: max value or suggested_maximum for param "{}" is missing.'.format(param_name))
            num_issues += 1
            max_value = sys.float_info.max

        if min_value == max_value:
            return min_value

        if random_from_min_max:
            value = np.random.uniform(low=min_value, high=max_value)
        else:
            # apply percent variation
            value = base_value + base_value * variation_amount * sign

        if value < min_value:
            value = min_value

        if value > max_value:
            value = max_value

        return value

    elif 'boolean' in type_name:
        # sample from true/false
        return random.choice([True, False])

    elif 'null' in type_name:
        # value is null
        return None

    elif 'integer' in type_name:
        min_value, max_value = _canonicalise.get_integer_bounds(canonicalized_param)

        if 'suggested_minimum' in canonicalized_param:
            min_value = canonicalized_param['suggested_minimum']

        if 'suggested_maximum' in canonicalized_param:
            max_value = canonicalized_param['suggested_maximum']

        if min_value is None:
            if print_warnings:
                print('Warning: min value or suggested_minimum for param "{}" is missing.'.format(param_name))
            num_issues += 1
            min_value = -2147483647
        if max_value is None:
            if print_warnings:
                print('Warning: max value or suggested_maximum for param "{}" is missing.'.format(param_name))
            num_issues += 1
            max_value = 2147483647

        if min_value == max_value:
            return min_value

        if random_from_min_max:
            value = np.random.randint(low=min_value, high=max_value)
        else:
            value = int(float(base_value) + float(base_value) * variation_amount * sign)

        if value < min_value:
            value = min_value

        if value > max_value:
            value = max_value

        return value

    # elif 'string' in type and not 'enum' in canonicalized_param:
    #     print('test')

    elif 'array' in type_name:
        array_schema = canonicalized_param.get('items', {})
        min_size = canonicalized_param.get('minItems', 0)
        max_size = canonicalized_param.get('maxItems')

        array_ret = []
        if isinstance(base_value, list):
            min_size = max(0, min_size - len(base_value))
            if max_size is None:
                max_size = len(base_value)

            for i in range(max_size):
                default_value = base_value[i]

                print('Processing array {} out of {} for param "{}"'.format(i+1, max_size, param_name))
                array_config = build_json_config(array_schema, default_value, print_warnings=print_warnings)
                print('')
                array_ret.append(array_config)
            return array_ret
        else:
            print('Unknown format for param "{}" (expecting list) for base_value in array: {}'.format(param_name, base_value))
            num_issues += 1

        print('Not implemented: array for {}, param name: {}, using default: {}'.format(canonicalized_param, param_name, base_value))
        num_issues += 1

    elif 'object' in type_name:
        print('Not implemented: object for {}, param name: {}, using default: {}'.format(canonicalized_param, param_name, base_value))
        num_issues += 1
    else:
        print('Unknown option: {}, param_name: {}, using default: {}'.format(canonicalized_param, param_name, base_value))
        num_issues += 1

    return base_value

def build_json_config(json_schema, json_original_config, randomly_perturb_param=False, perturb_param_chance=0.5,
                      variation_amount=0.05, random_variation_sign=True, random_from_min_max=False, print_warnings=True):
    global num_issues
    global fatal_issues

    if '$schema' in json_schema:

        if 'technique' not in json_schema:
            if print_warnings:
                print('Missing "technique" in root schema, see "https://pages.nist.gov/trojai/docs/submission.html#parameter-loading" for an example')
            num_issues += 1
        if 'technique_description' not in json_schema:
            if print_warnings:
                print('Missing "technique_description" in root schema, see "https://pages.nist.gov/trojai/docs/submission.html#parameter-loading" for an example')
            num_issues += 1
        if 'technique_changes' not in json_schema:
            if print_warnings:
                print('Missing "technique_changes" in root schema, see "https://pages.nist.gov/trojai/docs/submission.html#parameter-loading" for an example')
            num_issues += 1
        if 'commit_id' not in json_schema:
            if print_warnings:
                print('Missing "commit_id" in root schema, see "https://pages.nist.gov/trojai/docs/submission.html#parameter-loading" for an example')
            num_issues += 1
        if 'repo_name' not in json_schema:
            if print_warnings:
                print('Missing "repo_name" in root schema, see "https://pages.nist.gov/trojai/docs/submission.html#parameter-loading" for an example')
            num_issues += 1
        if 'technique_type' not in json_schema:
            if print_warnings:
                print('Missing "technique_type" in root schema, see "https://pages.nist.gov/trojai/docs/submission.html#parameter-loading" for an example')
            num_issues += 1

    resolved_json_schema = _resolve.resolve_all_refs(copy.deepcopy(json_schema))

    param_properties = resolved_json_schema['properties']

    config = {}

    # Check strings for presence of enum
    for param_name in param_properties.keys():
        param = param_properties[param_name]

        if param_name not in json_original_config:
            continue

        if 'type' in param and param['type'] == 'string' and 'enum' not in param:
            if print_warnings:
                print('Warning: param "{}" is marked as a string and does not contain an enum. Using "{}" as the default.'.format(param_name, json_original_config[param_name]))
            num_issues += 1
            param_properties[param_name]['enum'] = [json_original_config[param_name]]

    for param_name in param_properties.keys():
        param = param_properties[param_name]
        type_name = _canonicalise.get_type(param)

        if randomly_perturb_param:
            # do not skip array or objects
            if type_name != 'array' and type_name != 'object':
                # check if we should skip
                if np.random.uniform() >= perturb_param_chance:
                    continue

        if param_name not in json_original_config:
            if print_warnings:
                print('Warning: param "{}" was not found in your configuration file. Skipping adding it to new config.'.format(param_name))
            num_issues += 1
            continue

        resolved_param = _resolve.resolve_all_refs(param)

        canonicalized_param = _canonicalise.canonicalish(resolved_param)

        base_value = json_original_config[param_name]
        config[param_name] = get_value(param_name, canonicalized_param, type_name, base_value, variation_amount, random_variation_sign, random_from_min_max, print_warnings)

    # Attempt check if we are missing anything from the original config
    for param_name in json_original_config.keys():
        if param_name not in config:
            if print_warnings:
                print('Warning: param "{}" was found in original config file, but is not in the generated one'.format(param_name))
            num_issues += 1



    return config



def collect_json_metaparams(s_path):
    client.load(s_path)
    output = client.execute(['cat', '/metaparameters.json'])

    if not isinstance(output, str):
        print('Failed to load metaparameters_schema.json, output: {}'.format(output))
        return None

    if "return_code" in output and output["return_code"] != 0:
        print('Error, unable to find metaparameters')
        return None

    return json.loads(output)

def collect_json_metaparams_schema(s_path) :
    client.load(s_path)

    output = client.execute(['cat', '/metaparameters_schema.json'])
    if not isinstance(output, str):
        print('Failed to load metaparameters_schema.json, output: {}'.format(output))
        return None

    if "return_code" in output and output["return_code"] != 0:
        print('Error, unable to find metaparameters_schema')
        return None

    return json.loads(output)

def is_schema_configuration_valid(json_schema, config):
    global num_issues

    # Check for $schema
    if '$schema' not in json_schema:
        print(
            'Missing "$schema" in root schema, see "https://github.com/usnistgov/trojai-example/blob/master/metaparameters_schema.json" for an example')
        num_issues += 1

    # Add additional properties false to main schema and any defs if they exist
    json_schema['additionalProperties'] = False

    if '$defs' in json_schema:
        def_dict = json_schema['$defs']
        if isinstance(def_dict, dict):
            for key in def_dict.keys():
                def_dict[key]['additionalProperties'] = False

    try:
        jsonschema.validate(instance=config, schema=json_schema)
    except jsonschema.exceptions.SchemaError as e:
        print(e)
        num_issues += 1
    except jsonschema.exceptions.ValidationError as ex:
        print(ex)
        num_issues += 1
    except Exception as exc:
        print(exc)
        num_issues += 1

    try:
        build_json_config(json_schema, config, print_warnings=True)
    except Exception as e:
        print('There was an issue parsing your schema. We may not have support for one of your options, please send us your schema to check.\nException: {}'.format(e))
        num_issues += 1


    print('There were {} issues while parsing your jsonschema.'.format(num_issues))

    return num_issues == 0

def is_container_configuration_valid(container_filepath):
    import logging
    json_schema = collect_json_metaparams_schema(container_filepath)
    json_original_config = collect_json_metaparams(container_filepath)

    if json_schema is None:
        logging.info("Failed to load json_schema from container {}".format(container_filepath))
        return False

    if json_original_config is None:
        logging.info("Failed to load original config from container {}".format(container_filepath))
        return False

    return is_schema_configuration_valid(json_schema, json_original_config)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate singularity container or json schema to verify submission for TrojAI. '
                                                 'Use either --container-filepath or --jsonschema-filepath and --config-filepath to evaluate the schema. '
                                                 'If you use container-filepath, you will need the singularity program in your path.')
    parser.add_argument('--container-filepath', type=str, help='The filepath to the singularity container.')
    parser.add_argument('--jsonschema-filepath', type=str, help='The filepath to the jsonschema.')
    parser.add_argument('--config-filepath', type=str, help="The filepath to the program's configuration file")

    args = parser.parse_args()

    container_filepath = args.container_filepath
    jsonschema_filepath = args.jsonschema_filepath
    config_filepath = args.config_filepath

    json_schema = None
    json_original_config = None

    # Verify arguments and load json files
    if container_filepath is not None:
        if not is_container_configuration_valid(container_filepath):
            exit(1)
    elif jsonschema_filepath is not None and config_filepath is not None:
        with open(jsonschema_filepath) as f:
            json_schema = json.load(f)
        with open(config_filepath) as f:
            json_original_config = json.load(f)

        if json_schema is None or json_original_config is None:
            print('Failed to load schema and configuration files.')
            exit(1)

        # param_lookup = create_param_lookup(json_schema)
        # new_config = perturb_config(param_lookup, json_original_config)
        print('Done')

        if not is_schema_configuration_valid(json_schema, json_original_config):
            exit(1)
    else:
        parser.print_help()
        exit(1)

    exit(0)
