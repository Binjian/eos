import argparse
import datetime
import os
import subprocess
import time

# resumption settings
parser = argparse.ArgumentParser(
    'DDPG with reduced observations (no expected velocity) Suite'
)

parser.add_argument(
    '-a',
    '--agent',
    type=str,
    default='ddpg',
    help="RL agent choice: 'ddpg' for DDPG; 'rdpg' for Recurrent DPG",
)

parser.add_argument(
    '-u',
    '--ui',
    type=str,
    default='cloud',
    help="User Inferface: 'mobile' for mobile phone (for training); 'local' for local hmi; 'cloud' for no UI",
)
parser.add_argument(
    '-r',
    '--resume',
    help='resume the last training with restored model, checkpoint and pedal map',
    action='store_true',
)
parser.add_argument(
    '-t',
    '--record_table',
    help='record action table during training',
    action='store_true',
)
parser.add_argument(
    '-p',
    '--path',
    type=str,
    help='relative path to be saved, for create subfolder for different drivers',
)
parser.add_argument(
    '-v',
    '--vehicle',
    type=str,
    default='.',
    help="vehicle ID like 'VB7' or 'MP3' or VIN 'HMZABAAH1MF011055'",
)
parser.add_argument(
    '-d',
    '--driver',
    type=str,
    default='.',
    help="driver ID like 'longfei.zheng' or 'jiangbo.wei'",
)
parser.add_argument(
    '-m',
    '--remotecan',
    type=str,
    default='10.0.64.78:5000',
    help='url for remote can server, e.g. 10.10.0.6:30865, or name, e.g. baiduyun_k8s, newrizon_test',
)
parser.add_argument(
    '-w',
    '--web',
    type=str,
    default='10.0.64.78:9876',
    help='url for web ui server, e.g. 10.10.0.13:9876, or name, e.g. baiduyun_k8s, newrizon_test',
)
parser.add_argument(
    '-o',
    '--mongodb',
    type=str,
    default='mongo_local',
    help='url for mongodb server in format usr:password@host:port, e.g. admint:y02ydhVqDj3QFjT@10.10.0.4:23000, or simply name with synced default config, e.g. mongo_cluster, mongo_local',
)
args = parser.parse_args()

udpfileName = (
    os.getcwd()
    + '/../../data/udp-pcap/rl_agent_vb-'
    + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    + '.pcap'
)
# --cloud -t -p testremote -r
portNum = 8002  # port number
pid = os.fork()
if pid == 0:  # copy process
    time.sleep(1)
    if args.resume:
        os.execlp(
            'python',
            'python',
            '../rl_agent.py',
            '-a ',
            args.agent,
            '--ui',
            'local',
            '--resume',
            '--path',
            args.path,
            '--record_table',
            '-v',
            'VB7',
            '-d',
            'longfei',
            '-o',
            'mongo_intra',
        )

    else:
        os.execlp(
            'python',
            'python',
            '../realtime_train_infer_ddpg.py',
            '--cloud',
            '--web',
            '--path',
            args.path,
            '--record_table',
        )  #  run Simulation
else:
    p = subprocess.Popen(
        [
            'tcpdump',
            'udp',
            '-w',
            udpfileName,
            '-i',
            'lo',
            'port',
            str(portNum),
        ],
        stdout=subprocess.PIPE,
    )
    result = os.waitpid(-1, 0)
    p.terminate()
