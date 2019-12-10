# Automatic_Speech_Recognition

### Prepare train/dev/test data
```
sh prepare_data.sh
```

### Requirements
```
virtualenv --python=python3 venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### TODO
[] Remove teacher in inference.  
[] Calculate audio length to char length conversion rate.  
[] Modify while loop maxlen in inference phase.  
[] Evaluate with WER.  
[] Add beam search decoder.  
[] Add CTC loss.
