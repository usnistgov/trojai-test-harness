import os

from drive_io import DriveIO


def upload(token, filepath):

    if os.path.exists(filepath) and os.path.exists(token):
        print("Starting upload")
        g_drive = DriveIO(token)
        g_drive.upload(filepath)
    else:
        print(filepath + " or " + token + " do not exist")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Upload a folder to the trojai google drive.')

    parser.add_argument('--token-pickle-filepath', type=str,
                        help='Path token.pickle file holding the oauth keys.',
                        default='token.pickle')
    parser.add_argument('--filepath', type=str,
                        help='The file or directory to upload',
                        required=True)

    args = parser.parse_args()

    token = args.token_pickle_filepath
    filepath = args.filepath

    upload(token, filepath)

