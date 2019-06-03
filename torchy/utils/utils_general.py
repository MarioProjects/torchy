import os
import socket

from slackclient import SlackClient


def get_hostname():
    return socket.gethostname()


def time_to_human(start, end, verbose=False):
    #### start and end --> time.time()
    #### returns string or print
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    # print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    if verbose:
        print("{:0>1} hr {:0>2} min".format(int(hours), int(minutes)))
    else:
        return "{:0>1} hr {:0>2} min".format(int(hours), int(minutes))


def hours_elapsed(start, end):
    #### start and end --> time.time()
    #### returns hours as int
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return int(hours)


''' ######################################################################## '''
''' ####################### MESSAGING - LOGGING ############################ '''
''' ######################################################################## '''

# Tutorial slack-token and logging https://github.com/MarioProjects/Python-Slack-Logging
# echo "export SLACK_TOKEN='my-slack-token'" >> ~/.bashrc
SLACK_TOKEN = os.environ.get('SLACK_TOKEN')


def slack_message(message, channel):
    if SLACK_TOKEN == None:
        print("No Slack Token found in system!")
    else:
        sc = SlackClient(SLACK_TOKEN)
        sc.api_call('chat.postMessage', channel=channel,
                    text=message, username='My Sweet Bot',
                    icon_emoji=':robot_face:')
