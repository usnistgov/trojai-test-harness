import os
import io
import pickle
import random
import mimetypes
import time
import logging
from typing import List

import socket
socket.setdefaulttimeout(300)  # set timeout to 5 minutes (300s)

from google_drive_file import GoogleDriveFile

from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

# Limit logging from Drive API
logging.getLogger('googleapiclient.discovery').setLevel(logging.WARNING)
logging.getLogger('google_auth_oauthlib.flow').setLevel(logging.WARNING)
logging.getLogger('google.auth.transport.requests').setLevel(logging.WARNING)
logging.getLogger('googleapiclient.http').setLevel(logging.WARNING)
logging.getLogger('googleapiclient.errors').setLevel(logging.WARNING)


class DriveIO(object):
    # If modifying these scopes, delete the file token.pickle.
    SCOPES = ['https://www.googleapis.com/auth/drive']

    def __init__(self, token_pickle_filepath):
        self.token_pickle_filepath = token_pickle_filepath
        self.page_size = 100
        self.max_retry_count = 4
        self.__get_service(self.token_pickle_filepath)

    def __get_service(self, token_pickle_filepath):
        logging.debug('Starting connection to Google Drive.')
        creds = None
        try:
            # The file token.pickle stores the user's access and refresh tokens, and is
            # created automatically when the authorization flow completes for the first
            # time.
            if os.path.exists(token_pickle_filepath):
                logging.debug('Found token file: {}'.format(token_pickle_filepath))
                with open(token_pickle_filepath, 'rb') as token:
                    creds = pickle.load(token)
            logging.debug('Token credentials loaded')
            # If there are no (valid) credentials available, let the user log in.
            if not creds:
                logging.error('Credentials could not be loaded. Rebuild token using create_auth_token.py')
                raise RuntimeError('Credentials could not be loaded. Rebuild token using create_auth_token.py')

            # check if the credentials are not valid
            if not creds.valid:
                if creds.expired and creds.refresh_token:
                    logging.debug('Credentials exists, but are no longer valid, attempting to refresh.')
                    creds.refresh(Request())
                    logging.debug('Credentials refreshed successfully.')
                    # Save the credentials for the next run
                    with open(token_pickle_filepath, 'wb') as token:
                        pickle.dump(creds, token)
                    logging.debug('Credentials refreshed and saved to "{}".'.format(token_pickle_filepath))
                else:
                    logging.error('Could not refresh credentials. Rebuild token using create_auth_token.py.')
                    raise RuntimeError('Could not refresh credentials. Rebuild token using create_auth_token.py.')

            logging.debug('Building Drive service from credentials.')
            self.service = build('drive', 'v3', credentials=creds, cache_discovery=False)
            # Turn off cache discover to prevent logging warnings
            # https://github.com/googleapis/google-api-python-client/issues/299

            logging.debug('Querying Drive to determine account owner details.')
            response = self.service.about().get(fields="user").execute(num_retries=self.max_retry_count)
            self.user_details = response.get('user')
            self.email_address = self.user_details.get('emailAddress')
            logging.info('Connected to Drive for user: "{}" with email "{}".'.format(self.user_details.get('displayName'), self.email_address))
        except:
            logging.error('Failed to connect to Drive.')
            raise

    def __query_worker(self, query: str) -> List[GoogleDriveFile]:
        # https://developers.google.com/drive/api/v3/search-files
        # https://developers.google.com/drive/api/v3/reference/query-ref
        try:
            logging.debug('Querying Drive API with "{}".'.format(query))
            retry_count = 0
            while True:
                try:
                    # Call the Drive v3 API, blocking through pageSize records for each call
                    page_token = None
                    items = list()
                    while True:
                        # name, id, modifiedTime, sharingUser
                        response = self.service.files().list(q=query,
                                                             pageSize=self.page_size,
                                                             fields="nextPageToken, files(name, id, modifiedTime, owners)",
                                                             pageToken=page_token,
                                                             spaces='drive').execute(num_retries=self.max_retry_count)
                        items.extend(response.get('files'))
                        page_token = response.get('nextPageToken', None)
                        if page_token is None:
                            break
                    break  # successfully download file list, break exponential backoff scheme loop
                except HttpError as e:
                    retry_count = retry_count + 1
                    if e.resp.status in [104, 404, 408, 410] and retry_count <= self.max_retry_count:
                        # Start the upload from the beginning.
                        logging.info('Drive Query Error, restarting query from beginning (attempt {}/{}) with exponential backoff.'.format(retry_count, self.max_retry_count))
                        sleep_time = random.random() * 2 ** retry_count
                        time.sleep(sleep_time)
                    else:
                        raise

            logging.debug('Downloaded list of {} files from Drive account.'.format(len(items)))
            file_list = list()
            for item in items:
                owner = item['owners'][0]  # user first owner by default
                g_file = GoogleDriveFile(owner['emailAddress'], item['name'], item['id'], item['modifiedTime'])
                file_list.append(g_file)
        except:
            logging.error('Failed to connect to and list files from Drive.')
            raise

        return file_list

    def query_by_filename(self, file_name: str) -> List[GoogleDriveFile]:
        query = "name = '{}' and trashed = false".format(file_name)
        file_list = self.__query_worker(query)
        return file_list

    def query_by_email_and_filename(self, email: str, file_name: str) -> List[GoogleDriveFile]:
        query = "name = '{}' and '{}' in owners and trashed = false".format(file_name, email)
        file_list = self.__query_worker(query)
        return file_list

    def query_by_email(self, email: str) -> List[GoogleDriveFile]:
        query = "'{}' in owners and trashed = false".format(email)
        file_list = self.__query_worker(query)
        return file_list

    def download(self, g_file: GoogleDriveFile, output_dirpath: str) -> None:
        retry_count = 0
        logging.info('Downloading file: "{}" from Drive'.format(g_file))
        while True:
            try:
                request = self.service.files().get_media(fileId=g_file.id)
                file_data = io.FileIO(os.path.join(output_dirpath, g_file.name), 'wb')
                downloader = MediaIoBaseDownload(file_data, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk(num_retries=self.max_retry_count)
                    logging.debug("  downloaded {:d}%".format(int(status.progress() * 100)))
                return  # download completed successfully
            except HttpError as e:
                retry_count = retry_count + 1
                if e.resp.status in [104, 404, 408, 410] and retry_count <= self.max_retry_count:
                    # Start the upload from the beginning.
                    logging.info('Download Error, restarting download from beginning (attempt {}/{}) with exponential backoff.'.format(retry_count, self.max_retry_count))
                    sleep_time = random.random() * 2 ** retry_count
                    time.sleep(sleep_time)
                else:
                    raise
            except:
                logging.error('Failed to download file "{}" from Drive.'.format(g_file.name))
                raise

    def upload(self, file_path: str) -> str:
        _, file_name = os.path.split(file_path)
        logging.info('Uploading file: "{}" to Drive'.format(file_name))
        m_type = mimetypes.guess_type(file_name)[0]

        # TODO ensure file_path is a file, and not a folder

        for retry_count in range(self.max_retry_count):
            try:
                existing_files_list = self.query_by_email_and_filename(self.email_address, file_name)

                existing_file_id = None
                if len(existing_files_list) > 0:
                    existing_file_id = existing_files_list[0].id

                file_metadata = {'name': file_name}
                media = MediaFileUpload(file_path, mimetype=m_type, resumable=True)

                if existing_file_id is not None:
                    logging.info("Updating existing file '{}' on Drive.".format(file_name))
                    request = self.service.files().update(fileId=existing_file_id, body=file_metadata, media_body=media)
                else:
                    logging.info("Uploading new file '{}' to Drive.".format(file_name))
                    request = self.service.files().create(body=file_metadata, media_body=media, fields='id')

                response = None
                # loop while there are additional chunks
                while response is None:
                    status, response = request.next_chunk(num_retries=self.max_retry_count)
                    if status:
                        logging.debug("  uploaded {:d}%".format(int(status.progress() * 100)))

                file = request.execute()
                return file.get('id')  # upload completed successfully

            except HttpError as e:
                if e.resp.status in [104, 404, 408, 410] and retry_count <= self.max_retry_count:
                    # Start the upload from the beginning.
                    logging.info('Upload Error, restarting upload from beginning (attempt {}/{}) with exponential backoff.'.format(retry_count, self.max_retry_count))
                    sleep_time = random.random() * 2 ** retry_count
                    time.sleep(sleep_time)
                else:
                    raise
            except:
                logging.error("Failed to upload file '{}' to Drive.".format(file_name))
                raise

    def share(self, file_id: str, share_email: str) -> None:
        if share_email is not None:
            # update the permissions to share the log file with the team, using short exponential backoff scheme
            user_permissions = {'type': 'user', 'role': 'reader', 'emailAddress': share_email}
            for retry_count in range(self.max_retry_count):
                sleep_time = random.random() * 2 ** retry_count
                time.sleep(sleep_time)
                try:
                    self.service.permissions().create(fileId=file_id, body=user_permissions, fields='id', sendNotificationEmail=False).execute()
                    logging.info('Successfully shared file {} with {}.'.format(file_id, share_email))
                    return  # permissions were successfully modified if no exception
                except:
                    if retry_count <= 4:
                        logging.info('Failed to modify permissions on try, performing random exponential backoff.')
                    else:
                        logging.error("Failed to share uploaded file '{}' with '{}'.".format(file_id, share_email))
                        raise

    def upload_and_share(self, file_path: str, share_email: str) -> None:
        file_id = self.upload(file_path)
        self.share(file_id, share_email)

    def submission_download(self, email: str, output_dirpath: str, metadata_filepath: str, sts: bool) -> GoogleDriveFile:
        actor_file_list = self.query_by_email(email)

        # filter list based on file prefix
        gdrive_file_list = list()
        for g_file in actor_file_list:
            if sts and g_file.name.startswith('test'):
                gdrive_file_list.append(g_file)
            if not sts and not g_file.name.startswith('test'):
                gdrive_file_list.append(g_file)

        # ensure submission is unique (one and only one possible submission file from a team email)
        if len(gdrive_file_list) < 1:
            msg = "Actor does not have submission from email {}.".format(email)
            logging.error(msg)
            raise IOError(msg)
        if len(gdrive_file_list) > 1:
            msg = "Actor submitted {} file from email {}.".format(len(gdrive_file_list), email)
            logging.error(msg)
            raise IOError(msg)

        submission = gdrive_file_list[0]
        submission.save_json(metadata_filepath)
        logging.info('Downloading "{}" from Actor "{}" last modified time "{}".'.format(submission.name, submission.email, submission.modified_epoch))
        self.download(submission, output_dirpath)
        return submission


