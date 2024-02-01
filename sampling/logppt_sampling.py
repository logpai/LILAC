import json
import os
import pandas as pd
import re
import string
from sklearn.utils import shuffle
import textdistance
import random
import heapq
from collections import Counter, defaultdict, deque, OrderedDict
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
import time
import calendar
import argparse

datasets = [
    "Apache",
    "BGL",
    "Hadoop",
    "HDFS",
    "HealthApp",
    "HPC",
    "Linux",
    "Mac",
    "OpenSSH",
    "OpenStack",
    "Proxifier",
    "Spark",
    "Thunderbird",
    "Zookeeper"
]

# datasets = [
#     "Apache",
# ]

benchmark = {
    'HDFS': {
        'log_file': 'HDFS/HDFS_full.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
    },

    'Hadoop': {
        'log_file': 'Hadoop/Hadoop_full.log',
        'log_format': '<SessionId> <Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
    },

    'Spark': {
        'log_file': 'Spark/Spark_full.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>',
    },

    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_full.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
    },

    'BGL': {
        'log_file': 'BGL/BGL_full.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    },

    'HPC': {
        'log_file': 'HPC/HPC_full.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
    },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird_full.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
    },

    'Windows': {
        'log_file': 'Windows/Windows_full.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
    },

    'Linux': {
        'log_file': 'Linux/Linux_full.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
    },

    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_full.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
    },

    'Apache': {
        'log_file': 'Apache/Apache_full.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
    },

    'Proxifier': {
        'log_file': 'Proxifier/Proxifier_full.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
    },

    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH_full.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
    },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_full.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
    },

    'Mac': {
        'log_file': 'Mac/Mac_full.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
    }
}


def generate_logformat_regex(log_format):
    """ Function to generate regular expression to split log messages
    """
    headers = []
    splitters = re.split(r'(<[^<>]+>)', log_format)
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(' +', '\\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    regex = re.compile('^' + regex + '$')
    return headers, regex


def log_to_dataframe(log_file, log_format):
    """ Function to transform log file to dataframe
    """
    headers, regex = generate_logformat_regex(log_format)
    log_messages = []
    line_count = 0
    with open(log_file, 'r', encoding='utf8', errors='ignore') as fin:
        for line in fin.readlines():
            try:
                match = regex.search(line.strip())
                message = [match.group(header) for header in headers]
                log_messages.append(message)
                line_count += 1
            except Exception as _:
                pass
    logdf = pd.DataFrame(log_messages, columns=headers)
    logdf.insert(0, 'LineId', None)
    logdf['LineId'] = [i + 1 for i in range(line_count)]
    return logdf


def lcs_distance(x, y):
    seq1 = x.split()
    seq2 = y.split()
    lengths = [[0 for j in range(len(seq2) + 1)] for i in range(len(seq1) + 1)]
    # row 0 and column 0 are initialized to 0 already
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            if seq1[i] == seq2[j]:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])

    return 1 - 2 * lengths[-1][-1] / (len(seq1) + len(seq2))


def lev_distance(x, y):
    return textdistance.levenshtein.normalized_distance(x, y)


def euc_distance(x, y):
    return textdistance.cosine.normalized_distance(x, y)


def jaccard_distance(x, y):
    return textdistance.jaccard.normalized_distance(x.split(), y.split())


def ratcliff_distance(x, y):
    return textdistance.ratcliff_obershelp.normalized_distance(x, y)


def min_distance(c_set, t_set):
    D = []
    for c_inst in c_set:
        min_candidate_distance = 1e10
        for t_inst in t_set:
            min_candidate_distance = min(min_candidate_distance, jaccard_distance(c_inst, t_inst))
        D.append(min_candidate_distance)
    return D


def adaptive_random_sampling(logs, k, n_candidate=128):
    sample_set = []
    T = []
    for r in range(k):
        print("r: ", r)
        if len(sample_set) == 0:
            i = max(range(0, len(logs)), key=lambda x: (len(logs[x][0].split()), logs[x][2]))
            T.append(logs[i][0])
            sample_set.append(logs[i][1])
            #del logs[i]
            continue
        candidate_set = [(x, logs[x]) for x in range(len(logs)) if x in random.sample(range(len(logs)), n_candidate)]
        candidate_set = sorted(candidate_set, key=lambda x: x[1][2], reverse=True)
        candidate_distance = min_distance([x[1][0] for x in candidate_set], T)
        best_candidate = max(range(len(candidate_distance)), key=candidate_distance.__getitem__)
        T.append(candidate_set[best_candidate][1][0])
        sample_set.append(candidate_set[best_candidate][1][1])
        #del logs[candidate_set[best_candidate][0]]
    return sample_set
# shot * (candidate + candidate*log_candidate, candidate * shot * distance)


class Vocab:
    def __init__(self, stopwords=["<*>"]):
        stopwords = [
            "a",
            "an",
            "and",
            "i",
            "ie",
            "so",
            "to",
            "the",

        ] + list(calendar.day_name) + list(calendar.day_abbr) \
          + list(calendar.month_name) + list(calendar.month_abbr)
        self.token_counter = Counter()
        self.stopwords = frozenset(set(stopwords))
        #print(self.__filter_stopwords(['LDAP', 'Built', 'with']))

    def build(self, sequences):
        print("Build vocab with examples: ", len(sequences))
        for sequence in sequences:
            sequence = self.__filter_stopwords(sequence)
            #print(sequence)
            self.update(sequence)

    def update(self, sequence):
        sequence = self.__filter_stopwords(sequence)
        self.token_counter.update(sequence)

    def topk_tokens(self, sequence, topk=3):
        sequence = self.__filter_stopwords(sequence)
        token_count = [(token, self.token_counter[token]) for token in set(sequence)]
        topk_tuples = heapq.nlargest(topk, token_count, key=lambda x: x[1])
        topk_keys = tuple([t[0] for t in topk_tuples])
        return topk_keys

    def __len__(self):
        return len(self.token_counter)

    def __filter_stopwords(self, sequence):
        return [
            token
            for token in sequence
            if (len(token) > 2) and (token not in self.stopwords)
        ]



def adaptive_random_sampling(logs, k, n_candidate=128):
    sample_set = []
    T = []
    for r in range(k):
        if len(sample_set) == 0:
            i = max(range(0, len(logs)), key=lambda x: (len(logs[x][0].split()), logs[x][2]))
            T.append(logs[i][0])
            sample_set.append(logs[i][1])
            del logs[i]
            continue
        candidate_set = [(x, logs[x]) for x in range(len(logs)) if x in random.sample(range(len(logs)), n_candidate)]
        candidate_set = sorted(candidate_set, key=lambda x: x[1][2], reverse=True)
        candidate_distance = min_distance([x[1][0] for x in candidate_set], T)
        best_candidate = max(range(len(candidate_distance)), key=candidate_distance.__getitem__)
        T.append(candidate_set[best_candidate][1][0])
        sample_set.append(candidate_set[best_candidate][1][1])
        del logs[candidate_set[best_candidate][0]]
    return sample_set



def common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-full', '--full_data',
                        help="Set this if you want to test on full dataset",
                        default=False, action='store_true')
    args = parser.parse_args()
    return args


def clean(s):
    """ Preprocess log message
    Parameters
    ----------
    s: str, raw log message
    Returns
    -------
    str, preprocessed log message without number tokens and special characters
    """
    s = re.sub(':|\(|\)|=|,|"|\{|\}|@|$|\[|\]|\||;|\.', ' ', s)
    s = " ".join([word.lower() if word.isupper() else word for word in s.strip().split()])
    s = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', s))
    s = " ".join([word for word in s.split() if not bool(re.search(r'\d', word))])
    trantab = str.maketrans(dict.fromkeys(list(string.punctuation)))
    s = s.translate(trantab)
    s = " ".join([word.lower().strip() for word in s.strip().split()])
    return s



if __name__ == '__main__':
    args = common_args()
    data_dir = "../full_dataset"
    if not os.path.exists(f"{data_dir}/logppt_samples"):
        os.makedirs(f"{data_dir}/logppt_samples")
    for dataset in datasets:
        print(dataset)
        todo = False
        for shot in [8, 16, 32, 64, 128]:
            if not os.path.exists(f"{data_dir}/logppt_samples/{dataset}/{shot}shot.json"):
                todo = True
                break
        if not todo:
            continue
        log_file = benchmark[dataset]['log_file']
        if not args.full_data:
            data_dir = data_dir.replace("full", "2k")
            log_file = log_file.replace("full", "2k")
        print(data_dir, log_file)
        labelled_logs = pd.read_csv(f'{data_dir}/{log_file}_structured.csv')
        k_rate = 0.2 if args.full_data else 1
        length = int(k_rate * len(labelled_logs))
        labelled_logs = labelled_logs[:length]
        # labelled_logs = labelled_logs[:length].drop_duplicates(['Content'], keep='first')
        print("Original logs: ", len(labelled_logs))
        print("Unique templates: ", labelled_logs['EventTemplate'].nunique())

        content = [(clean(x), i, len(x)) for i, x in enumerate(labelled_logs['Content'].tolist())]
        content = [x for x in content if len(x[0].split()) > 1]
        #content = content * 10
        print("Content OK")
        keywords_list = []
        os.makedirs(f"{data_dir}/logppt_samples/{dataset}/", exist_ok=True)


        for shot in [8, 16, 32, 64, 128]:
            begin_time = time.time()
            sampled_ids = adaptive_random_sampling(content, shot)

            labeled_samples = [(row['Content'], row['EventTemplate']) for _, row in labelled_logs.take(sampled_ids).iterrows()]
            labeled_samples = [{"query": x[0], "answer": x[1]} for x in labeled_samples]
            sampled_templates = set([row['EventTemplate'] for _,row in labelled_logs.take(sampled_ids).iterrows()])
            end_time = time.time()
            with open("logppt.txt", "a") as fa:
                fa.write(f"{dataset} {shot}-shot cover {len(sampled_templates)} used {end_time - begin_time}\n")
            print(f"{shot}-shot sampling covers sampled templates: {len(sampled_templates)}")
            print(f"Time cost: {end_time - begin_time}")
            with open(f"{data_dir}/logppt_samples/{dataset}/{shot}shot.json", "w") as f:
                for s in labeled_samples[:shot]:
                    f.write(json.dumps(s) + "\n")

        #labelled_logs = lab
        # print("Removed length: ", len(labelled_logs))
        # train_df = labelled_logs.sample(n=2000)
        # contents = {}
        # for i, x in enumerate(labelled_logs['Content'].to_list()):
        #     x, fx = clean(x)
        #     if len(x.split()) > 1:
        #         contents[i] = (x, fx)
        # # content = {i: clean(x) if len(x.split()) > 1 for i, x in enumerate(labelled_logs['Content'].tolist())}
        # # for shot in [4, 8, 16, 32]:
        # os.makedirs(f"{data_dir}/logppt_samples/{dataset}/", exist_ok=True)
        
        # hierichical_clusters = hierichical_clustering(contents)
        # for shot in [8, 16, 32, 64, 128]:
        # # for shot in [32]:
        #     if os.path.exists(f"{data_dir}/logppt_samples/{dataset}/{shot}shot.json"):
        #         continue
        #     sampled_ids = hierichical_distribute(hierichical_clusters, shot, labelled_logs['Content'].to_list())
        #     sampled_templates = set([row['EventTemplate'] for _,row in labelled_logs.take(sampled_ids).iterrows()])
        #     print(f"{shot}-shot sampling covers sampled templates: {len(sampled_templates)}")
        #     candidate_samples = [(row['Content'], row['EventTemplate']) for _, row in labelled_logs.take(sampled_ids).iterrows()]
        #     candidate_samples = [{"query": x[0], "answer": x[1].replace('<*>', '{variables}')} for x in candidate_samples]
        #     with open(f"{data_dir}/logppt_samples/{dataset}/{shot}shot.json", "w") as f:
        #         for s in candidate_samples[:shot]:
        #             f.write(json.dumps(s) + "\n")

            
