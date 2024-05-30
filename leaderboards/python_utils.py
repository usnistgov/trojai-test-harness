from typing import Dict


def update_object_values(object_inst: object, config_dict: Dict) -> bool:
    config_keys = object_inst.__dict__.keys()
    is_updated = False

    for key in config_keys:
        if key in config_dict:
            if hasattr(object_inst, key):
                setattr(object_inst, key, config_dict[key])
                is_updated = True

    return is_updated

def get_value(my_dict, key):
    if key in my_dict:
        return my_dict[key]
    else:
        return None