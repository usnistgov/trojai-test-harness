import smtplib
import subprocess

ALERT_THRESHOLD = 80  # 80 percent, as usage is reported in percentage [0,100]

SERVER = "smtp.nist.gov"
FROM = 'trojai@nist.gov'


def send(to: str, subject: str, message: str):
    try:
        mail_message = "From: {}\r\nTo: {}\r\nSubject: {}\r\n\r\n\r\n{}".format(FROM, to, subject, message)

        # Send the mail
        server = smtplib.SMTP(SERVER)
        server.sendmail(FROM, to, mail_message)
        server.quit()
    except:
        pass


out = subprocess.Popen(["df", "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = out.communicate()

if stderr != b'':
    # subprocess call produced an error
    subject = "V100 trojainas capacity monitor is down"
    msg = "V100 trojainas capacity monitor is down. Error = {}".format(stderr)
    send('trojai@nist.gov', subject, msg)

# decode the subprocess results
output = stdout.decode('utf-8').strip()
output = str.split(output,'\n')[1]

# find the disk usage percentage
percent = [val for val in output.split() if '%' in val]
percent = int(percent[0].replace('%',''))

if percent >= ALERT_THRESHOLD:
    msg = "V100 trojainas is {} percent full".format(percent)
    send('trojai@nist.gov', msg, msg)