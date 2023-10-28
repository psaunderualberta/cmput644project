import os
from glob import glob

MAX_TREE_SIZE = 10

COMBINED_DATA_FILES = glob(os.path.join(os.path.dirname(__file__), "..", "..", "CICIoT2023", "data", "combined_data", "*.csv"))
SHORTENED_DATA_FILES = glob(os.path.join(os.path.dirname(__file__), "..", "..", "CICIoT2023", "data", "shortened_data", "*.csv"))
RAW_DATA_FILES = glob(os.path.join(os.path.dirname(__file__), "..", "..", "CICIoT2023", "data", "original", "*.csv"))

BENIGN = "Benign"
ATTACK_CLASS = 1
ATTACK = "Attack"
BENIGN_CLASS = 0

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
    'DDoS-RSTFINFlood': ATTACK_CLASS,
    'DDoS-PSHACK_Flood': ATTACK_CLASS,
    'DDoS-SYN_Flood': ATTACK_CLASS,
    'DDoS-UDP_Flood': ATTACK_CLASS,
    'DDoS-TCP_Flood': ATTACK_CLASS,
    'DDoS-ICMP_Flood': ATTACK_CLASS,
    'DDoS-SynonymousIP_Flood': ATTACK_CLASS,
    'DDoS-ACK_Fragmentation': ATTACK_CLASS,
    'DDoS-UDP_Fragmentation': ATTACK_CLASS,
    'DDoS-ICMP_Fragmentation': ATTACK_CLASS,
    'DDoS-SlowLoris': ATTACK_CLASS,
    'DDoS-HTTP_Flood': ATTACK_CLASS,

    'DoS-UDP_Flood': ATTACK_CLASS,
    'DoS-SYN_Flood': ATTACK_CLASS,
    'DoS-TCP_Flood': ATTACK_CLASS,
    'DoS-HTTP_Flood': ATTACK_CLASS,


    'Mirai-greeth_flood': ATTACK_CLASS,
    'Mirai-greip_flood': ATTACK_CLASS,
    'Mirai-udpplain': ATTACK_CLASS,

    'Recon-PingSweep': ATTACK_CLASS,
    'Recon-OSScan': ATTACK_CLASS,
    'Recon-PortScan': ATTACK_CLASS,
    'VulnerabilityScan': ATTACK_CLASS,
    'Recon-HostDiscovery': ATTACK_CLASS,

    'DNS_Spoofing': ATTACK_CLASS,
    'MITM-ArpSpoofing': ATTACK_CLASS,

    'BenignTraffic': BENIGN_CLASS,

    'BrowserHijacking': ATTACK_CLASS,
    'Backdoor_Malware': ATTACK_CLASS,
    'XSS': ATTACK_CLASS,
    'Uploading_Attack': ATTACK_CLASS,
    'SqlInjection': ATTACK_CLASS,
    'CommandInjection': ATTACK_CLASS,

    'DictionaryBruteForce': ATTACK_CLASS,
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

__normalized_columns = [col.replace(' ', '_').lower() for col in X_COLUMNS]
__grammar_columns = "\n\t| ".join([f"{col}" for col in __normalized_columns])
HEURISTIC_GRAMMAR = r"""
    ?heuristic: "(" binary ")"
        | "(" unary ")"
        | """ + "\n\t| ".join([f"{col} -> terminal" for col in __normalized_columns]) + r"""
        | FLOAT -> decimal
        
    """ + "\n".join([f"?{col}: \"{col}\"" for col in __normalized_columns]) + r"""
    
    ?unary: unary_op heuristic 
    ?unary_op: "neg"    -> neg
        | "abs"         -> abs
        | "sqrt"        -> sqrt
        | "sqr"         -> sqr

    ?binary: binary_op heuristic heuristic 
    ?binary_op: "+"     -> plus
        | "/"           -> div
        | "*"           -> mul
        | "-"           -> sub
        | "max"         -> max
        | "min"         -> min

    %import common.SIGNED_FLOAT -> FLOAT
    %import common.WS
    %ignore WS
"""
