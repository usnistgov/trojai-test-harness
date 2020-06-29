import pickle

from google_auth_oauthlib.flow import InstalledAppFlow
from drive_io import DriveIO


def create_auth_token(credentials_filepath, token_pickle_filepath):
    flow = InstalledAppFlow.from_client_secrets_file(credentials_filepath, DriveIO.SCOPES)
    creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open(token_pickle_filepath, 'wb') as token:
        pickle.dump(creds, token)
    return creds


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Build token.pickle file for authenticating Google Drive API access.')

    parser.add_argument('--token-pickle-filepath', type=str,
                        help='Path token.pickle file holding the oauth keys. If token.pickle is missing, but credentials have been provided, token.pickle will be generated after opening a web-browser to have the user accept the app permissions',
                        default='token.pickle')
    parser.add_argument('--credentials-filepath',
                        type=str,
                        help='Path to the credentials.json file holding the Google Cloud Project with API access to trojai@nist.gov Google Drive.',
                        default='credentials.json')

    args = parser.parse_args()
    token = args.token_pickle_filepath
    credentials = args.credentials_filepath
    create_auth_token(credentials, token)