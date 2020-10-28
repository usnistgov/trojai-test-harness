# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import pickle

from google_auth_oauthlib.flow import InstalledAppFlow
from actor_executor.drive_io import DriveIO


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