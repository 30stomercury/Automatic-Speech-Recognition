# Automatic_Speech_Recognition

### Prepare data
#### Prepare libirspeech train/dev/test data
```
sh prepare_libri_data.sh LibriSpeech-100
```
or 
```
sh prepare_libri_data.sh LibriSpeech-360
```

#### Prepare tedlium train/dev/test data
```
sh prepare_ted_data.sh TED-LIUMv1
```
or
```
sh prepare_ted_data.sh TED-LIUMv2
```

### Requirements
```
virtualenv --python=python3 venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### Tensorboard
```
tensorboard --logdir ./summary
```

### TODO
- [ ] Evaluate with WER.  
- [ ] Add beam search decoder.  
- [ ] Add CTC loss.
