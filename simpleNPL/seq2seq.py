import numpy as np
import os

import cntk as C
import requests

C.cntk_py.set_fixed_random_seed(1)
# ============= dataset preparation =============
def download(url, filename):
    """ utility function to download a file """
    response = requests.get(url, stream=True)
    with open(filename, "wb") as handle:
        for data in response.iter_content():
            handle.write(data)

MODEL_DIR = "."
DATA_DIR = os.path.join('..', 'Examples', 'SequenceToSequence', 'CMUDict', 'Data')
# If above directory does not exist, just use current.
if not os.path.exists(DATA_DIR):
    DATA_DIR = '.'

dataPath = {
  'validation': 'tiny.ctf',
  'training': 'cmudict-0.7b.train-dev-20-21.ctf',
  'testing': 'cmudict-0.7b.test.ctf',
  'vocab_file': 'cmudict-0.7b.mapping',
}

for k in sorted(dataPath.keys()):
    path = os.path.join(DATA_DIR, dataPath[k])
    if os.path.exists(path):
        print("Reusing locally cached:", path)
    else:
        print("Starting download:", dataPath[k])
        url = "https://github.com/Microsoft/CNTK/blob/release/2.4/Examples/SequenceToSequence/CMUDict/Data/%s?raw=true"%dataPath[k]
        download(url, path)
        print("Download completed")
    dataPath[k] = path

# ============== functions ====================
def get_vocab(path):
    vocab = [w.strip() for w in open(path).readlines()]
    i2w = {i:w for i, w in enumerate(vocab)}
    w2i = {w:i for i, w in enumerate(vocab)}
    return (vocab, i2w, w2i)

def create_reader(path, is_training):
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        features = StreamDef(field='S0', shape=input_vocab_dim, is_sparse=True),
        labels   = StreamDef(field='S1', shape=label_vocab_dim, is_sparse=True)
        )), randomize = is_training, max_sweeps = INFINITELY_REPEAT if is_training else 1)

# ============= configure =====================
input_vocab_dim = 69
label_vocab_dim = 69
