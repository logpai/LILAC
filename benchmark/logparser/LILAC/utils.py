import pickle
import json

def cache_to_file(log_tuples, cached_file):
    with open(cached_file, "w") as fw:
        for tuples in log_tuples:
            fw.write(f"{tuples[1]} {tuples[0]}\n")


def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print("No file:", file_path)
        return None
    except Exception as e:
        print("Load Error:", str(e))
        return None


def save_pickle(data, file_path):
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
        #print("Save success:", file_path)
    except Exception as e:
        print("Save error:", str(e))

def load_tuple_list(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    tuple_list = []
    for line in lines:
        idx, s = line.strip().split(' ', 1)
        tuple_list.append((s, int(idx)))
    return tuple_list


def read_json_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            json_dict = json.loads(line)
            data.append(json_dict)
    return data