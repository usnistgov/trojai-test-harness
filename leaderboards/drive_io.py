# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
import json
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import io
import random
import mimetypes
import time
import logging
from typing import List
from leaderboards import time_utils
from threading import Thread, Lock, Condition
from queue import Queue, Empty

import socket
socket.setdefaulttimeout(120)  # set timeout to 5 minutes (300s)

from leaderboards.google_drive_file import GoogleDriveFile


# Limit logging from Drive API
logging.getLogger('googleapiclient.discovery').setLevel(logging.WARNING)
logging.getLogger('google_auth_oauthlib.flow').setLevel(logging.WARNING)
logging.getLogger('google.auth.transport.requests').setLevel(logging.WARNING)
logging.getLogger('googleapiclient.http').setLevel(logging.WARNING)
logging.getLogger('googleapiclient.errors').setLevel(logging.WARNING)



class UploadWorker(Thread):
    def __init__(self, g_drive: 'DriveIO', queue: Queue, lock: Lock, cond: Condition):
        super().__init__()
        self.g_drive = g_drive
        self.queue = queue
        self.lock = lock
        self.cond = cond
        self.daemon = True
        self.work_done = False

    def run(self):
        while True:
            file_path = None
            folder_id = None
            with self.lock:
                try:
                    file_path, folder_id = self.queue.get(block=False)
                    if file_path is None:
                        self.cond.notify_all()
                        return
                except Empty:
                    self.cond.wait()
                    continue

            self.g_drive.upload(file_path, folder_id)
            print('Finished uploading {}'.format(file_path))



    @staticmethod
    def init_workers(num_threads: int, g_drive: 'DriveIO'):
        workers = []
        for _ in range(num_threads):
            worker = UploadWorker(g_drive, g_drive.upload_queue, g_drive.upload_queue_lock, g_drive.upload_queue_cond)
            worker.start()
            workers.append(worker)
        g_drive.is_upload_workers_initialized = True
        return workers


class DriveIO(object):
    # If modifying these scopes, delete the file token.pickle.
    SCOPES = ['https://www.googleapis.com/auth/drive']

    def __init__(self, token_pickle_filepath, upload_queue: Queue = None, upload_queue_lock: Lock = None, upload_queue_cond: Condition = None):
        self.token_pickle_filepath = token_pickle_filepath
        self.page_size = 100
        self.max_retry_count = 4
        # self.request_count = 0
        self.upload_queue = upload_queue
        self.upload_queue_lock = upload_queue_lock
        self.upload_queue_cond = upload_queue_cond
        self.is_upload_workers_initialized = False

        self.__get_service(self.token_pickle_filepath)

        self.folder_cache = {}
        self.folder_times = {}
        # Cache is stale after 120 seconds
        self.stale_time_limit = 120

    def __get_service(self, token_json_filepath):
        from googleapiclient.discovery import build
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        logging.debug('Starting connection to Google Drive.')
        creds = None
        try:
            # The file token.pickle stores the user's access and refresh tokens, and is
            # created automatically when the authorization flow completes for the first
            # time.
            if os.path.exists(token_json_filepath):
                creds = Credentials.from_authorized_user_file(token_json_filepath, DriveIO.SCOPES)

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
                    with open(token_json_filepath, 'w') as token:
                        json.dump(creds, token)
                        # pickle.dump(creds, token)
                    logging.debug('Credentials refreshed and saved to "{}".'.format(token_json_filepath))
                else:
                    logging.error('Could not refresh credentials. Rebuild token using create_auth_token.py.')
                    raise RuntimeError('Could not refresh credentials. Rebuild token using create_auth_token.py.')

            logging.debug('Building Drive service from credentials.')
            self.service = build('drive', 'v3', credentials=creds, cache_discovery=False)
            # Turn off cache discover to prevent logging warnings
            # https://github.com/googleapis/google-api-python-client/issues/299

            logging.debug('Querying Drive to determine account owner details.')
            response = self.service.about().get(fields="user").execute(num_retries=self.max_retry_count)
            # self.request_count += 1
            self.user_details = response.get('user')
            self.email_address = self.user_details.get('emailAddress')
            logging.info('Connected to Drive for user: "{}" with email "{}".'.format(self.user_details.get('displayName'), self.email_address))
        except Exception as e:
            logging.error('Failed to connect to Drive.')
            logging.error('Exception: {}'.format(e))
            raise

    def __query_worker(self, query: str) -> List[GoogleDriveFile]:
        from googleapiclient.errors import HttpError

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
                        # self.request_count += 1
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

    def query_folder(self, folder_name: str, parent_id='root') -> List[GoogleDriveFile]:
        query = "name = '{}' and trashed = false and mimeType = 'application/vnd.google-apps.folder' and '{}' in parents".format(folder_name, parent_id)
        folder_list = self.__query_worker(query)
        return folder_list

    def query_by_filename(self, file_name: str) -> List[GoogleDriveFile]:
        query = "name = '{}' and trashed = false".format(file_name)
        file_list = self.__query_worker(query)
        return file_list

    def query_by_email_and_filename(self, email: str, file_name: str, folder_id=None) -> List[GoogleDriveFile]:
        if folder_id is None:
            query = "name = '{}' and '{}' in owners and trashed = false".format(file_name, email)
        else:
            query = "name = '{}' and '{}' in owners and trashed = false and '{}' in parents".format(file_name, email, folder_id)
        file_list = self.__query_worker(query)
        return file_list

    def query_by_email(self, email: str) -> List[GoogleDriveFile]:
        query = "'{}' in owners and trashed = false".format(email)
        file_list = self.__query_worker(query)
        return file_list

    def download(self, g_file: GoogleDriveFile, output_dirpath: str) -> None:
        from googleapiclient.http import MediaIoBaseDownload
        from googleapiclient.errors import HttpError

        retry_count = 0
        logging.info('Downloading file: "{}" from Drive'.format(g_file))
        while True:
            try:
                request = self.service.files().get_media(fileId=g_file.id)
                # self.request_count += 1
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

    def create_folder(self, folder_name, parent_id='root') -> str:
        from googleapiclient.errors import HttpError
        folder_key = '{}_{}'.format(folder_name, parent_id)
        if folder_key in self.folder_cache.keys():
            # Check staleness
            if folder_key in self.folder_times.keys():
                folder_time = self.folder_times[folder_key]

                cur_epoch = time_utils.get_current_epoch()
                if folder_time + self.stale_time_limit > cur_epoch:
                    return self.folder_cache[folder_key]
                else:
                    del self.folder_cache[folder_key]
                    del self.folder_times[folder_key]

        file_metadata = {
            'name': folder_name,
            'parents': [parent_id],
            'mimeType': 'application/vnd.google-apps.folder'
        }

        retry_count = 0
        logging.debug('Creating google drive folder {}'.format(folder_name))

        while True:
            try:
                existing_folders = self.query_folder(folder_name, parent_id=parent_id)

                if len(existing_folders) > 0:
                    return existing_folders[0].id

                file = self.service.files().create(body=file_metadata, fields='id').execute()
                # self.request_count += 1
                self.folder_cache[folder_key] = file.get('id')
                self.folder_times[folder_key] = time_utils.get_current_epoch()
                return file.get('id')

            except HttpError as e:
                retry_count = retry_count + 1
                if e.resp.status in [104, 404, 408, 410] and retry_count <= self.max_retry_count:
                    logging.info('Folder creation error, restarting (attempt {}/{}) with exponential backoff'.format(retry_count, self.max_retry_count))
                    slee_time = random.random() * 2 ** retry_count
                    time.sleep(slee_time)
                else:
                    raise
            except:
                logging.error('Failed to create folder  "{}" from Drive.'.format(folder_name))
                raise

    def enqueue_file_upload(self, file_paths: List[str], folder_id=None):
        if self.is_upload_workers_initialized:
            with self.upload_queue_lock:
                for i in range(len(file_paths)):
                    self.upload_queue.put([file_paths[i], folder_id])
                self.upload_queue_cond.notify_all()
        else:
            logging.error('Upload workers were not correctly initialized prior to enqueueing files for upload')


    def upload(self, file_path: str, folder_id=None, skip_existing=False) -> str:
        from googleapiclient.http import MediaFileUpload
        from googleapiclient.errors import HttpError

        _, file_name = os.path.split(file_path)
        logging.info('Uploading file: "{}" to Drive'.format(file_name))
        m_type = mimetypes.guess_type(file_name)[0]

        # ensure file_path is a regular file
        if not os.path.isfile(file_path):
            logging.error('Upload file_path = "{}" is not a regular file, aborting upload.'.format(file_path))
            raise RuntimeError('Upload file_path = "{}" is not a regular file, aborting upload.'.format(file_path))

        for retry_count in range(self.max_retry_count):
            try:
                existing_files_list = self.query_by_email_and_filename(self.email_address, file_name, folder_id=folder_id)

                existing_file_id = None
                if len(existing_files_list) > 0:
                    existing_file_id = existing_files_list[0].id

                if existing_file_id is not None and skip_existing:
                    logging.info('Skipping upload {}, it already exists on gdrive'.format(file_name))
                    return existing_file_id

                if folder_id is None:
                    file_metadata = {'name': file_name}
                else:
                    if existing_file_id is not None:
                        file_metadata = {'name': file_name}
                    else:
                        file_metadata = {'name': file_name, 'parents': [folder_id]}

                media = MediaFileUpload(file_path, mimetype=m_type, resumable=True)

                if existing_file_id is not None:
                    logging.info("Updating existing file '{}' on Drive.".format(file_name))
                    request = self.service.files().update(fileId=existing_file_id, body=file_metadata, media_body=media)
                    # self.request_count += 1
                else:
                    logging.info("Uploading new file '{}' to Drive.".format(file_name))
                    request = self.service.files().create(body=file_metadata, media_body=media, fields='id')
                    # self.request_count += 1

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
                if retry_count == 0:
                    sleep_time = 0.1
                else:
                    sleep_time = random.random() * 2 ** retry_count
                time.sleep(sleep_time)
                try:
                    self.service.permissions().create(fileId=file_id, body=user_permissions, fields='id', sendNotificationEmail=False).execute()
                    # self.request_count += 1
                    logging.info('Successfully shared file {} with {}.'.format(file_id, share_email))
                    return  # permissions were successfully modified if no exception
                except:
                    if retry_count <= 4:
                        logging.info('Failed to modify permissions on try, performing random exponential backoff.')
                    else:
                        logging.error("Failed to share uploaded file '{}' with '{}'.".format(file_id, share_email))
                        raise

    def remove_all_sharing_permissions(self, file_id: str) -> None:
        permissions = self.service.permissions().list(fileId=file_id).execute()
        # self.request_count += 1
        permissions = permissions['permissions']

        for permission in permissions:
            if permission['role'] != 'owner':
                for retry_count in range(self.max_retry_count):
                    if retry_count == 0:
                        sleep_time = 0.1
                    else:
                        sleep_time = random.random() * 2 ** retry_count
                    time.sleep(sleep_time)

                    try:
                        self.service.permissions().delete(fileId=file_id, permissionId=permission['id']).execute()
                        # self.request_count += 1
                        logging.info("Successfully removed share permission '{}' from file {}.".format(permission, file_id))
                        break  # break retry loop
                    except:
                        if retry_count <= 4:
                            logging.info('Failed to modify permissions on try, performing random exponential backoff.')
                        else:
                            logging.error("Failed to remove share permission '{}' from file '{}'.".format(permission, file_id))
                            raise

    def upload_and_share(self, file_path: str, share_email: str) -> None:
        file_id = self.upload(file_path)
        # unshare to remove all permissions except for owner, to ensure that if the file is deleted on the receivers end, that they get a new copy of it.
        self.remove_all_sharing_permissions(file_id)
        self.share(file_id, share_email)

    def upload_and_share_multiple(self, file_path: str, share_emails: list) -> None:
        file_id = self.upload(file_path)
        for email in share_emails:
            self.share(file_id, email)

    def submission_download(self, email: str, output_dirpath: str, metadata_filepath: str, requested_leaderboard_name: str, requested_data_split_name) -> GoogleDriveFile:
        actor_file_list = self.query_by_email(email)

        # filter list based on file prefix
        gdrive_file_list = list()
        for g_file in actor_file_list:
            filename = g_file.name
            filename_split = filename.split('_')

            if len(filename_split) <= 2:
                continue

            leaderboard_name = filename_split[0]
            data_split_name = filename_split[1]

            if leaderboard_name == requested_leaderboard_name and data_split_name == requested_data_split_name:
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


    def create_external_root_folder(self):
        return self.create_folder('trojai_admin_leaderboard')

    def create_actor_root_folder(self, actor_name):
        return self.create_folder('trojai_results_{}'.format(actor_name))

    def create_leaderboard_summary_folder(self):
        trojai_summary_folder_id = self.create_external_root_folder()
        return self.create_folder('leaderboard_summary_data', parent_id=trojai_summary_folder_id)

    def create_actor_summary_folder(self):
        trojai_summary_folder_id = self.create_external_root_folder()
        return self.create_folder('actor_summary_data', parent_id=trojai_summary_folder_id)

    def get_submission_actor_and_external_folder_ids(self, actor_name: str, leaderboard_name: str, data_split_name: str):
        try:
            # Setup the external folder for an actor, leaderboard name, and data split
            actor_plots_folder_id = self.create_actor_summary_folder()
            root_external_folder_id = self.create_folder('{}'.format(actor_name), parent_id=actor_plots_folder_id)
            external_actor_submission_folder_id = self.create_folder('{}_{}'.format(leaderboard_name, data_split_name), parent_id=root_external_folder_id)

            # Setup folder for actor
            root_actor_folder_id = self.create_actor_root_folder(actor_name)
            actor_submission_folder_id = self.create_folder('{}_{}'.format(leaderboard_name, data_split_name), parent_id=root_actor_folder_id)
        except:
            logging.error('Failed to create google drive actor directories')
            actor_submission_folder_id = None
            external_actor_submission_folder_id = None

        return actor_submission_folder_id, external_actor_submission_folder_id
