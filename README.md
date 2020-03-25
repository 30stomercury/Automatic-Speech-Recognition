# A TensorFlow Implementation of [Listen Attention and Spell](https://arxiv.org/abs/1508.01211)

This is a tensorflow implementation of end-to-end ASR. Though there are several fantastic github repos in tensorflow, I tried to implemented it **without using `tf.contrib.seq2seq` API**. In addition, the performance on LibriSpeech dev/test datasets are evaluated with.

## Overview

![](demo/las.png)

* Components:
    - Char/Subword text encoding.
    - MFCC/fbank acoustic features.
    - LAS training (visualized with tensorboard: loss, sample text outputs, features, alignments).  
    - Joint CTC-Attention training. (See notes)
    - Batch testing using greedy decoder.
    - RNNLM.
    - Beam search decoder.

## Remarks

Note that this project is still in progress.
* Notes
    - Currently, I only test this model on: MFCC 39 (13+delta+accelerate) features + character text encoding + training without CTC loss.
    - BPE and CTC related parts are not yet fully tested.
    - Volume augmentation is currently commented out because it shows little improvements.

* Improvements
    - Augmentation include speed. (IMPORTANT) 
    - Label smoothing. (IMPORANT)
    - Bucketing.

In my experience, add more data is the best policy.

## Requirements
```
pip3 install virtualenv
virtualenv --python=python3 venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Prepare libirspeech train/dev/test data
```
sh prepare_libri_data.sh LibriSpeech-100 (OR LibriSpeech-360)
```

## Prepare tedlium train/dev/test data
```
sh prepare_ted_data.sh TED-LIUMv1 (OR TED-LIUMv2)
```
## Train

## Tensorboard
```
tensorboard --logdir ./summary
```

## TODO
- [ ] Evaluate performance with subword unit: Subword las training, subword-based RNNLM. 
- [ ] Evaluate performance on joint CTC training, decoding.
- [ ] Test on TEDLIUM dataset.
- [ ] Add other attention mechanisms such as multi-head attention. 
- [ ] Add scheduled sampling.
