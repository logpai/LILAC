import textdistance
import re
import string
import random

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


def clean(s):
    """ Preprocess log message
    Parameters
    ----------
    s: str, raw log message
    Returns
    -------
    str, preprocessed log message without number tokens and special characters
    """
    # s = re.sub(r'(\d+\.){3}\d+(:\d+)?', " ", s)
    # s = re.sub(r'(\/.*?\.[\S:]+)', ' ', s)
    return s
    s = re.sub(':|\(|\)|=|,|"|\{|\}|@|$|\[|\]|\||;|\.', ' ', s)
    s = " ".join([word.lower() if word.isupper() else word for word in s.strip().split()])
    s = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', s))
    s = " ".join([word for word in s.split() if not bool(re.search(r'\d', word))])
    trantab = str.maketrans(dict.fromkeys(list(string.punctuation)))
    s = s.translate(trantab)
    s = " ".join([word.lower().strip() for word in s.strip().split()])
    return s


def prompt_select(prompts, log, demonstration, selection_method="LILAC"):
    #examples = [{'query' : 'Log message: `authentication  failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=adsl-70-242-75-179.dsl.ksc2mo.swbell.net`',
    #             'answer' : 'Log template: `authentication  failure; logname= uid={uid} euid={euid} tty={ssh_type} ruser= rhost={rhost}`'}]
    if demonstration == 0:
        return []
    examples = []
    if selection_method == "random":
        print("random selection...")
        result = random.choices(prompts, k=demonstration)
    else:
        L = prompts
        log = clean(log)
        for d in L:
            d['similarity'] = jaccard_distance(clean(d['query']), log)
        sorted_dicts = sorted(L, key=lambda d: d['similarity'], reverse=False)
        result = sorted_dicts[:demonstration]
        
    for x in result:
        examples.append({'query': f"Log message: `{x['query']}`",
                         'answer': f"Log template: `{x['answer']}`"})
    # print(examples)
    return examples
