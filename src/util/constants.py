import os
from glob import glob

COMBINED_DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "CICIoT2023", "data", "combined_data.csv")
RAW_DATA_FILES = glob(os.path.join(os.path.dirname(__file__), "..", "..", "CICIoT2023", "data", "original", "*.csv"))

BENIGN = "Benign"
ATTACK = "Attack"

CLASSES_8_MAPPING = {
    'DDoS-RSTFINFlood': 'DDoS',
    'DDoS-PSHACK_Flood': 'DDoS',
    'DDoS-SYN_Flood': 'DDoS',
    'DDoS-UDP_Flood': 'DDoS',
    'DDoS-TCP_Flood': 'DDoS',
    'DDoS-ICMP_Flood': 'DDoS',
    'DDoS-SynonymousIP_Flood': 'DDoS',
    'DDoS-ACK_Fragmentation': 'DDoS',
    'DDoS-UDP_Fragmentation': 'DDoS',
    'DDoS-ICMP_Fragmentation': 'DDoS',
    'DDoS-SlowLoris': 'DDoS',
    'DDoS-HTTP_Flood': 'DDoS',

    'DoS-UDP_Flood': 'DoS',
    'DoS-SYN_Flood': 'DoS',
    'DoS-TCP_Flood': 'DoS',
    'DoS-HTTP_Flood': 'DoS',


    'Mirai-greeth_flood': 'Mirai',
    'Mirai-greip_flood': 'Mirai',
    'Mirai-udpplain': 'Mirai',

    'Recon-PingSweep': 'Recon',
    'Recon-OSScan': 'Recon',
    'Recon-PortScan': 'Recon',
    'VulnerabilityScan': 'Recon',
    'Recon-HostDiscovery': 'Recon',

    'DNS_Spoofing': 'Spoofing',
    'MITM-ArpSpoofing': 'Spoofing',

    'BenignTraffic': 'Benign',

    'BrowserHijacking': 'Web',
    'Backdoor_Malware': 'Web',
    'XSS': 'Web',
    'Uploading_Attack': 'Web',
    'SqlInjection': 'Web',
    'CommandInjection': 'Web',

    'DictionaryBruteForce': 'BruteForce',    
}

CLASSES_2_MAPPING = {
    'DDoS-RSTFINFlood': 'Attack',
    'DDoS-PSHACK_Flood': 'Attack',
    'DDoS-SYN_Flood': 'Attack',
    'DDoS-UDP_Flood': 'Attack',
    'DDoS-TCP_Flood': 'Attack',
    'DDoS-ICMP_Flood': 'Attack',
    'DDoS-SynonymousIP_Flood': 'Attack',
    'DDoS-ACK_Fragmentation': 'Attack',
    'DDoS-UDP_Fragmentation': 'Attack',
    'DDoS-ICMP_Fragmentation': 'Attack',
    'DDoS-SlowLoris': 'Attack',
    'DDoS-HTTP_Flood': 'Attack',

    'DoS-UDP_Flood': 'Attack',
    'DoS-SYN_Flood': 'Attack',
    'DoS-TCP_Flood': 'Attack',
    'DoS-HTTP_Flood': 'Attack',


    'Mirai-greeth_flood': 'Attack',
    'Mirai-greip_flood': 'Attack',
    'Mirai-udpplain': 'Attack',

    'Recon-PingSweep': 'Attack',
    'Recon-OSScan': 'Attack',
    'Recon-PortScan': 'Attack',
    'VulnerabilityScan': 'Attack',
    'Recon-HostDiscovery': 'Attack',

    'DNS_Spoofing': 'Attack',
    'MITM-ArpSpoofing': 'Attack',

    'BenignTraffic': 'Benign',

    'BrowserHijacking': 'Attack',
    'Backdoor_Malware': 'Attack',
    'XSS': 'Attack',
    'Uploading_Attack': 'Attack',
    'SqlInjection': 'Attack',
    'CommandInjection': 'Attack',

    'DictionaryBruteForce': 'Attack',
}

X_COLUMNS = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
       'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
       'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
       'ece_flag_number', 'cwr_flag_number', 'ack_count',
       'syn_count', 'fin_count', 'urg_count', 'rst_count', 
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP',
       'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min',
       'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue',
       'Radius', 'Covariance', 'Variance', 'Weight', 
]

CLASSES_34_Y_COLUMN = 'label'
CLASSES_8_Y_COLUMN = 'class_8'
CLASSES_2_Y_COLUMN = 'class_2'

