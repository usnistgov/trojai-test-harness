import os
import time

from drive_io import DriveIO

def connection_test(token, email):
    if os.path.exists(token):
        g_drive = DriveIO(token)
        while(True):
            files = g_drive.query_by_email(email)
            for f in files:
                print(f)
    else:
        print(token + " does not exist")


def main(token, upload_time, filepath, share_email):

    if os.path.exists(filepath) and os.path.exists(token):
        print("Starting upload loop")
        count = 1
        g_drive = DriveIO(token)
        while(True):
            g_drive.upload_and_share(filepath, share_email)
            print("Upload " + str(count) + " complete, sleeping for " + str(upload_time) + " seconds")
            time.sleep(upload_time)
            count = count + 1
    else:
        print(filepath + " or " + token + " do not exist")


if __name__ == "__main__":
    import argparse
    import time_utils

    parser = argparse.ArgumentParser(description='Infinitely loops copying a file to Google Drive and shares with trojai.')

    parser.add_argument('--token-pickle-filepath', type=str,
                        help='Path token.pickle file holding the oauth keys.',
                        default='token.pickle')
    parser.add_argument('--upload-time', type=time_utils.parse_time,
                        help='Amount of time to wait between uploads',
                        default="30m")
    parser.add_argument('--filepath', type=str,
                        help='The file to upload continuously',
                        required=True)
    parser.add_argument('--share-email', type=str,
                        help='Email to share files too',
                        default='trojai@nist.gov')

    args = parser.parse_args()

    token = args.token_pickle_filepath
    upload_time = args.upload_time
    filepath = args.filepath
    share_email = args.share_email

    connection_test(token, share_email)
    # main(token, upload_time, filepath, share_email)






