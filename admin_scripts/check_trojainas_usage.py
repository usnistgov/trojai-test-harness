# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

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