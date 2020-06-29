# Setup Drive API Access

Goto the Developer console
https://console.developers.google.com/

- Create a new project "trojai"

- Enable APIs - Enable Google Drive API

- Configure OAuth2 Screen

  Click on 'Credentials' in the LHS page menu.
  + Click on 'Configure Consent Screen'
  + Select 'Internal' for 'User Type' and click on 'Create'
  + Set the 'Application name' field to 'TrojAI'
  + Add the following scope: '../auth/drive'
  + Click on 'Save'

- Create Credentials (OAuth2)
  + Click on 'Credentials'
  + Click on 'Create Credentials'
  + Choose 'OAuth Client ID'
  + Choose 'Other' as 'Application type'
  + Set 'Name' to 'TrojAI'
  + Click on 'Save' or 'Create'

	Creates 'OAuth client' and gives 'Client ID' and 'Client Secret',
    which look like a public-private key pair.

  + Click OK

- Download credential, rename to credentials.json

-------------------------------------------------------------------------------

# libvert

Install libvert on the system.

`sudo apt install libvirt-dev`


# Setup Python for Google Drive Access

Based on https://developers.google.com/drive/api/v3/quickstart/python


`conda create --name drive python=3.7`

`conda activate drive`

`pip install --upgrade -r requirements.txt`


python3 -m venv test-env
source ~/test-env/bin/activate
pip install --upgrade wheel google-api-python-client google-auth-httplib2 google-auth-oauthlib jsonpickle pid numpy pytablewriter dominate GitPython httplib2==0.15

-------------------------------------------------------------------------------

# JSON CLI tool

- *NOT NEEDED*

Install `jq` for the shell:

`jq command -- apt install jq`

-------------------------------------------------------------------------------

# Create and Downlaod OAuth Token

- Run the following command; make sure paths are *right*.

```
	pushd trojai/src/te-scripts/actor_executor
	python3 create_auth_token.py \
	--credentials-filepath ~/Projects/trojai/OAuth-creds/credentials.json \
	--token-pickle-filepath ~/Projects/trojai/OAuth-creds/token.pickle
```

# Add Actor

```
python actor_manager.py --add-actor="<team name>,<submitting email>" --config-file=/mnt/trojainas/configRound0.json --log-file=./actor-log
```

# Start Test Loop

- Run the following command; make sure paths are *right*.

```
	pushd trojai/src/te-scripts/actor_executor
	python3 infinite_submitter.py \
	--token-pickle-filepath ~/Projects/trojai/OAuth-creds/token.pickle \
	--filepath ~/Projects/trojai/fake_trojan_detector.sif 
```


-------------------------------------------------------------------------------

# Restrictions for Files Shared with Trojai Drive


# FAQ

## Which VMs Go Where

STS (slurm queue test) has vms: 61, db

ES (slurm queue production) has vms: 3b, 60, 86, da 

## To share a file with TrojAI Google Drive user

1. Upload the file in question to your Google Drive account
2. Right click on the file and select "Share"
3. Enter "trojai@nist.gov" and click Done

## To Stop sharing a file

1. Right click on the file in question and select "Share".
2. Click on "Advanced" (bottom right of the dialog box).
3. Remove the people the file is shared with as required.
4. Click "Save changes"

## File Shared with Trojai Drive user does not show up in Trojai's Drive 'Shared with me' folder.

If trojai@nist.gov removes a file shared with it, that file will not show up in the trojai@nist.gov Drive again. The file will need to be deleted and a new copy uploaded and shared with trojai@nist.gov. Using the drive_io python tools, if the file exists in the target Google Drive account it will modify that existing file, not replace it, preventing the file from showing up for trojai Drive if it was removed on the trojai side. Delete the file from the source drive, re-upload a new copy, and share the new copy with trojai@nist.gov to re-share the file.