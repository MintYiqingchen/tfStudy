import numpy as np
import os

import cntk as C
from cntk.io import *
from utils import display_model
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
def create_model():
    embed = C.layers.Embedding(embedding_dim, name="embed") if use_embedding else identity
    # encoder
    # tricks: 1.use stabilizer to accelerate convergence
    # 2. use backwards to enhance the influence of former sequence part
    with C.layers.default_options(enable_self_stabilization=True, go_backwards=not use_attention):
        LastRecurrence = C.layers.Fold if not use_attention else C.layers.Recurrence
        encoder = C.layers.Sequential([
            embed, C.layers.Stabilizer(),
            C.layers.For(range(num_layers), lambda: C.layers.Recurrence(C.layers.LSTM(hidden_dim))),
            LastRecurrence(C.layers.LSTM(hidden_dim),return_full_state=True),
            (C.layers.Label('encode_h'), C.layers.Label('encode_c'))
            ], name='encode')

    # decoder
    # -train: input is attention vector and groundtruth(history)
    # -test: input is attention vector and last prediction(history)
    with C.layers.default_options(enable_self_stabilization=True):
        stab_in = C.layers.Stabilizer()
        rec_blocks = [C.layers.LSTM(hidden_dim) for i in range(num_layers)]
        stab_out = C.layers.Stabilizer()
        proj_out = C.layers.Dense(label_vocab_dim, name='out_proj')

        if use_attention:
            attention_model = C.layers.AttentionModel(attention_dim, name="attention_model")

        @C.Function
        def decoder(history, datain):
            encode_out = encoder(datain)
            r = embed(history)
            r = stab_in(r)
            for i in range(num_layers):
                rec_block = rec_blocks[i]
                if use_attention:
                    if i==0:
                        @C.Function
                        def lstm_with_attention(dh, dc, x):
                            # encoder hidden state, decoder hidden state
                            tmp = encode_out.outputs[0].owner
                            print(tmp)
                            h_att = attention_model(encode_out.outputs[0], dh)
                            x = C.splice(x, h_att)
                            return rec_block(dh,dc,x)
                        r = C.layers.Recurrence(lstm_with_attention)(r)
                    else:
                        r = C.layers.Recurrence(rec_block)(r)
                else:
                    r=C.layers.RecurrenceFrom(rec_block)(*(encode.outputs+(r,)))
            r=stab_out(r)
            r=proj_out(r)
            r=C.layers.Label('out_proj_out')(r)
            return r
        return decoder
def create_model_train(s2smodel):
    @C.Function
    def model_train(datain, labels):
        past_labels = C.layers.Delay(initial_state=sentence_start)(labels)
        return s2smodel(past_labels, datain)
    return model_train
def create_model_test(s2smodel):
    @C.Function
    def model_test(input:InputSequence[C.layers.Tensor[input_vocab_dim]]):
        # calculate until the output satisfied until_predicate
        unfold = C.layers.UnfoldFrom(lambda history:s2smodel(history, input) >> C.hardmax,
            until_predicate=lambda w: w[..., sentence_end_idx],
            length_increase=length_increase)
        return unfold(initial_state=sentence_start, dynamic_axes_like=input)
    return model_test
def create_criterion_function(model):
    @C.Function
    def criterion(input:InputSequence[C.layers.Tensor[input_vocab_dim]]
            ,labels:LabelSequence[C.layers.Tensor[label_vocab_dim]]):
        postprocessed_labels = C.sequence.slice(labels, 1, 0) # <s> A B C </s> --> A B C </s>
        z = model(input, postprocessed_labels)
        ce = C.cross_entropy_with_softmax(z, postprocessed_labels)
        errs = C.classification_error(z, postprocessed_labels)
        return (ce, errs)
    return criterion
# dummy for printing the input sequence below. Currently needed because input is sparse.
def create_sparse_to_dense(input_vocab_dim):
    I = C.Constant(np.eye(input_vocab_dim))
    @C.Function
    def no_op(input:InputSequence[C.layers.SparseTensor[input_vocab_dim]]):
        return C.times(input, I)
    return no_op
def train(train_reader, valid_reader, vocab, i2w, s2smodel, max_epochs, epoch_size):
    model_train = create_model_train(s2smodel)
    criterion = create_criterion_function(model_train)
    # also wire in a greedy decoder so that we can properly log progress on a validation example
    # This is not used for the actual training process.
    model_greedy = create_model_test(s2smodel)
    # Instantiate the trainer object to drive the model training
    minibatch_size = 72
    lr = 0.001 if use_attention else 0.005
    learner = C.fsadagrad(model_train.parameters,
                          #apply the learning rate as if it is a minibatch of size 1
                          lr = C.learning_parameter_schedule_per_sample([lr]*2+[lr/2]*3+[lr/4], epoch_size),
                          momentum = C.momentum_schedule(0.9366416204111472, minibatch_size=minibatch_size),
                          gradient_clipping_threshold_per_sample=2.3,
                          gradient_clipping_with_truncation=True)
    trainer = C.Trainer(None, criterion, learner)
    
    # records
    total_samples = 0
    mbs = 0
    eval_freq = 100

    # print out some useful training information
    C.logging.log_number_of_parameters(model_train) ; print()
    progress_printer = C.logging.ProgressPrinter(freq=30, tag='Training')

    # a hack to allow us to print sparse vectors
    sparse_to_dense = create_sparse_to_dense(input_vocab_dim)

    for epoch in range(max_epochs):
        while total_samples < (epoch+1) * epoch_size:
            # get next minibatch of training data
            mb_train = train_reader.next_minibatch(minibatch_size)

            # do the training
            trainer.train_minibatch({criterion.arguments[0]: mb_train[train_reader.streams.features],
                                     criterion.arguments[1]: mb_train[train_reader.streams.labels]})

            progress_printer.update_with_trainer(trainer, with_metric=True) # log progress

            # every N MBs evaluate on a test sequence to visually show how we're doing
            if mbs % eval_freq == 0:
                mb_valid = valid_reader.next_minibatch(1)

                # run an eval on the decoder output model (i.e. don't use the groundtruth)
                e = model_greedy(mb_valid[valid_reader.streams.features])
                print(format_sequences(sparse_to_dense(mb_valid[valid_reader.streams.features]), i2w))
                print("->")
                print(format_sequences(e, i2w))

                # visualizing attention window
                if use_attention:
                    debug_attention(model_greedy, mb_valid[valid_reader.streams.features])

            total_samples += mb_train[train_reader.streams.labels].num_samples
            mbs += 1

        # log a summary of the stats for the epoch
        progress_printer.epoch_summary(with_metric=True)

    # done: save the final model
    model_path = "model_%d.cmf" % epoch
    print("Saving final model to '%s'" % model_path)
    s2smodel.save(model_path)
    print("%d epochs complete." % max_epochs)

# Given a vocab and tensor, print the output
def format_sequences(sequences, i2w):
    return [" ".join([i2w[np.argmax(w)] for w in s]) for s in sequences]

# to help debug the attention window
def debug_attention(model, input):
    q = C.combine([model, model.attention_model.attention_weights])
    #words, p = q(input) # Python 3
    words_p = q(input)
    words = words_p[0]
    p     = words_p[1]
    output_seq_len = words[0].shape[0]
    p_sq = np.squeeze(p[0][:output_seq_len,:,:]) # (batch, output_len, input_len, 1)
    opts = np.get_printoptions()
    np.set_printoptions(precision=5)
    print(p_sq)
    np.set_printoptions(**opts)

def evaluation_decoder(reader, s2smodel, i2w):
    model = create_model_test(s2smodel)
    progress_printer = C.logging.ProgressPrinter(tag="Evaluation")
    sparse_to_dense = create_sparse_to_dense(input_vocab_dim)
    minibatch_size = 512
    num_total = 0
    num_wrong = 0
    while True:
        mb = reader.next_minibatch(minibatch_size)
        if not mb: # finish when end of test set reached
            break
        e = model_decoding(mb[reader.streams.features])
        outputs = format_sequences(e, i2w)
        labels  = format_sequences(sparse_to_dense(mb[reader.streams.labels]), i2w)
        # prepend sentence start for comparison
        outputs = ["<s> " + output for output in outputs]

        num_total += len(outputs)
        num_wrong += sum([label != output for output, label in zip(outputs, labels)])

    rate = num_wrong / num_total
    print("string error rate of {:.1f}% in {} samples".format(100 * rate, num_total))
    return rate

def do_test():
    model_path='model_0.cmf'
    model = C.Function.load(model_path)
    test_reader = create_reader(dataPath['testing'], False)
    evaluate_decoder(test_reader, model, i2w)
# ============= configure =====================
input_vocab_dim = 69
label_vocab_dim = 69

hidden_dim = 512
num_layers = 2
attention_dim = 128
use_attention = True
use_embedding = True
embedding_dim = 200
length_increase = 1.5

InputSequence = C.layers.SequenceOver[C.Axis('inputAxis')]
LabelSequence = C.layers.SequenceOver[C.Axis('labelAxis')]

vocab, i2w, _ = get_vocab(dataPath['vocab_file'])
train_reader = create_reader(dataPath['training'], True)
valid_reader = create_reader(dataPath['validation'], True)

sentence_start = C.Constant(np.array([w=='<s>' for w in vocab], dtype=np.float))
sentence_end_idx = vocab.index('</s>') # first </s>

if __name__ == '__main__':
    model = create_model()
    a = model.find_by_name('encode_h')
    #x = x.root_function
    print(a)
    #display_model(model)
    #train(train_reader, valid_reader, vocab, i2w, model, max_epochs=1, epoch_size=25000)