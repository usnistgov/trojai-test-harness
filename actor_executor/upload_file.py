import os

from drive_io import DriveIO


def upload(token, filepath):

    if os.path.exists(filepath) and os.path.exists(token):
        print("Starting upload")
        g_drive = DriveIO(token)
        g_drive.upload(filepath)
    else:
        print(filepath + " or " + token + " do not exist")


# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser(description='Upload a folder to the trojai google drive.')
#
#     parser.add_argument('--token-pickle-filepath', type=str,
#                         help='Path token.pickle file holding the oauth keys.',
#                         default='token.pickle')
#     parser.add_argument('--filepath', type=str,
#                         help='The file or directory to upload',
#                         required=True)
#
#     args = parser.parse_args()
#
#     token = args.token_pickle_filepath
#     filepath = args.filepath
#
#     upload(token, filepath)


token = '/home/mmajurski/nist/token.pickle'

# fp = '/mnt/scratch/trojai/data/round1/round1-holdout-dataset.tar.gz'
# upload(token, fp)
#
# fp = '/mnt/scratch/trojai/data/round1/round1-test-dataset.tar.gz'
# upload(token, fp)

fp = '/mnt/scratch/trojai/data/round1/trojai-round1-dataset-train.tar.gz'
upload(token, fp)




# fp = '/mnt/scratch/trojai/data/round2/round2-train-dataset/'
# fns = ['id-000000xx.tar.gz', 'id-000001xx.tar.gz', 'id-000002xx.tar.gz', 'id-000003xx.tar.gz', 'id-000004xx.tar.gz', 'id-000005xx.tar.gz','id-000006xx.tar.gz','id-000007xx.tar.gz','id-000008xx.tar.gz','id-000009xx.tar.gz', 'id-000010xx.tar.gz', 'id-000011xx.tar.gz']
#
#
# for fn in fns:
#     print(fn)
#     cur_fp = os.path.join(fp, fn)
#     upload(token, cur_fp)

