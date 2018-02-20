from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import requests
import os, math
import cntk as C
import numpy as np
import cntk.tests.test_utils
cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1) # fix a random seed for CNTK components

def download(url, filename):
    """ utility function to download a file """
    response = requests.get(url, stream=True)
    with open(filename, "wb") as handle:
        for data in response.iter_content():
            handle.write(data)

locations = ['Tutorials/SLUHandsOn', 'Examples/LanguageUnderstanding/ATIS/BrainScript']

data = {
  'train': { 'file': 'atis.train.ctf', 'location': 0 },
  'test': { 'file': 'atis.test.ctf', 'location': 0 },
  'query': { 'file': 'query.wl', 'location': 1 },
  'slots': { 'file': 'slots.wl', 'location': 1 },
  'intent': { 'file': 'intent.wl', 'location': 1 }  
}

for item in data.values():
    location = locations[item['location']]
    path = os.path.join('..', location, item['file'])
    if os.path.exists(path):
        print("Reusing locally cached:", item['file'])
        # Update path
        item['file'] = path
    elif os.path.exists(item['file']):
        print("Reusing locally cached:", item['file'])
    else:
        print("Starting download:", item['file'])
        url = "https://github.com/Microsoft/CNTK/blob/release/2.3.1/%s/%s?raw=true"%(location, item['file'])
        download(url, item['file'])
        print("Download completed")

# --------------- function implement -------------
def LookAhead(x):
    xn = C.sequence.future_value(x)
    return C.splice(x,xn)
def create_model():
    x = C.placeholder()
    with C.layers.default_options(initial_state=0.1):
        e = C.layers.Embedding(emb_dim, name='embed')(x)
        negRnn = C.layers.Recurrence(C.layers.LSTM(hidden_dim), go_backwards=True)(e)
        posRnn = C.layers.Recurrence(C.layers.LSTM(hidden_dim), go_backwards=False)(e)
        h = C.splice(posRnn, negRnn)
        out = C.layers.Dense(num_labels, name='classify')(h)
        return out
def create_reader(path, is_training):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
         query         = C.io.StreamDef(field='S0', shape=vocab_size,  is_sparse=True),
         intent        = C.io.StreamDef(field='S1', shape=num_intents, is_sparse=True),  
         slot_labels   = C.io.StreamDef(field='S2', shape=num_labels,  is_sparse=True)
     )), randomize=is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)
def create_criterion_function_preferred(model, labels):
    ce   = C.cross_entropy_with_softmax(model, labels)
    errs = C.classification_error      (model, labels)
    return ce, errs # (model, labels) -> (loss, error metric)
def train(reader, model_func, max_epochs=10, task='slot_tagging'):
    
    # Create the containers for input feature (x) and the label (y)
    x = C.sequence.input_variable(vocab_size)
    y = C.sequence.input_variable(num_labels)
    # Instantiate the model function; x is the input (feature) variable 
    model = model_func(x)
    # Instantiate the loss and error function
    loss, label_error = create_criterion_function_preferred(model, y)

    # training config
    epoch_size = 18000        # 18000 samples is half the dataset size 
    minibatch_size = 70
    
    # LR schedule over epochs 
    # In CNTK, an epoch is how often we get out of the minibatch loop to
    # do other stuff (e.g. checkpointing, adjust learning rate, etc.)
    lr_per_sample = [3e-4]*4+[1.5e-4]
    lr_per_minibatch = [lr * minibatch_size for lr in lr_per_sample]
    lr_schedule = C.learning_parameter_schedule(lr_per_minibatch, epoch_size=epoch_size)
    
    # Momentum schedule
    momentums = C.momentum_schedule(0.9048374180359595, minibatch_size=minibatch_size)
    
    # We use a the Adam optimizer which is known to work well on this dataset
    # Feel free to try other optimizers from 
    # https://www.cntk.ai/pythondocs/cntk.learner.html#module-cntk.learner
    learner = C.adam(parameters=model.parameters,
                     lr=lr_schedule,
                     momentum=momentums,
                     gradient_clipping_threshold_per_sample=15, 
                     gradient_clipping_with_truncation=True)

    # Setup the progress updater
    progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=max_epochs)
    
    # Uncomment below for more detailed logging
    #progress_printer = ProgressPrinter(freq=100, first=10, tag='Training', num_epochs=max_epochs) 

    # Instantiate the trainer
    trainer = C.Trainer(model, (loss, label_error), learner, progress_printer)

    # process minibatches and perform model training
    C.logging.log_number_of_parameters(model)
    
    # Assign the data fields to be read from the input
    if task == 'slot_tagging':
        data_map={x: reader.streams.query, y: reader.streams.slot_labels}
    else:
        data_map={x: reader.streams.query, y: reader.streams.intent} 
    t = 0
    for epoch in range(max_epochs):         # loop over epochs
        epoch_end = (epoch+1) * epoch_size
        while t < epoch_end:                # loop over minibatches on the epoch
            data = reader.next_minibatch(minibatch_size, input_map= data_map)  # fetch minibatch
            trainer.train_minibatch(data)               # update model with it
            t += data[y].num_samples                    # samples so far
        trainer.summarize_training_progress()

def evaluate(reader, model_func, task='slot_tagging'):
    x = C.sequence.input_variable(vocab_size)
    y = C.sequence.input_variable(num_labels)
    model = model_func(x)
    # Create the loss and error functions
    loss, label_error = create_criterion_function_preferred(model, y)

    # process minibatches and perform evaluation
    progress_printer = C.logging.ProgressPrinter(tag='Evaluation', num_epochs=0)
    
    # Assign the data fields to be read from the input
    if task == 'slot_tagging':
        data_map={x: reader.streams.query, y: reader.streams.slot_labels}
    else:
        data_map={x: reader.streams.query, y: reader.streams.intent} 

    while True:
        minibatch_size = 500
        data = reader.next_minibatch(minibatch_size, input_map= data_map)  # fetch minibatch
        if not data:                                 # until we hit the end
            break

        evaluator = C.eval.Evaluator(loss, progress_printer)
        evaluator.test_minibatch(data)
     
    evaluator.summarize_test_progress()
def print_data(data):
    v = list(data.values())[0]
    print("{} word number:{} sentence number:{}".format(v.data, v.num_samples, v.num_sequences)) 

def display_model(model):
    from IPython.display import SVG, display
    svg = C.logging.graph.plot(model, "tmp1.svg")
    display(SVG(filename="tmp1.svg"))
# --------------- global setting -----------------
# number of words in vocab, slot labels, and intent labels
vocab_size = 943 ; num_labels = 129 ; num_intents = 26    

# model dimensions
input_dim  = vocab_size
label_dim  = num_labels
emb_dim    = 150
hidden_dim = 150

if __name__ == '__main__':
    model = create_model()
    reader = create_reader(data['train']['file'], is_training=True)
    train(reader, model)
    reader = create_reader(data['test']['file'], is_training=False)
    evaluate(reader, model)