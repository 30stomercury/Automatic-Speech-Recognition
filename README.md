# A TensorFlow Implementation of [Listen Attention and Tell](https://arxiv.org/abs/1508.01211)

This is a tensorflow implementation of end-to-end ASR. Though there are several fantastic github repos in tensorflow, I tried to implemented LAS in tensorflow instead of using `tf.contrib.seq2seq` API

### Requirements
```
virtualenv --python=python3 venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### Prepare libirspeech train/dev/test data
```
sh prepare_libri_data.sh LibriSpeech-100 (OR LibriSpeech-360)
```

### Prepare tedlium train/dev/test data
```
sh prepare_ted_data.sh TED-LIUMv1 (OR TED-LIUMv2)
```
### Train

### Tensorboard
```
tensorboard --logdir ./summary
```

### TODO
- [ ] Evaluate performance on subword mode: Subword las training, subword RNNLM. 
- [ ] Evaluate performance on joint CTC training, decoding.
- [ ] Add other attention mechanisms.
