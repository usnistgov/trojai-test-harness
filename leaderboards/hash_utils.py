import os
import hashlib
import logging

def compute_hash(container_filepath: str, buf_size: int=65536):
    sha256 = hashlib.sha256()
    pre, ext = os.path.splitext(container_filepath)
    output_filepath = pre + '.sha256'

    if not os.path.exists(output_filepath):
        with open(container_filepath, 'rb') as f:
            while True:
                data = f.read(buf_size)
                if not data:
                    break
                sha256.update(data)

        with open(output_filepath, 'w') as f:
            f.write(sha256.hexdigest())

def load_hash(container_filepath: str):
    pre, ext = os.path.splitext(container_filepath)
    hash_filepath = pre + '.sha256'

    if not os.path.exists(hash_filepath):
        logging.warning('Failed to find hash filepath: {}'.format(hash_filepath))
        return None

    with open(hash_filepath, 'r') as f:
        hash_value = f.read()

    return hash_value