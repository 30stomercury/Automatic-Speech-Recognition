# Automatic_Speech_Recognition

### Prepare train/dev/test data
```
mkdir data

wget http://www.openslr.org/resources/12/train-clean-100.tar.gz
wget http://www.openslr.org/resources/12/dev-clean.tar.gz
wget http://www.openslr.org/resources/12/test-clean.tar.gz

tar -xvf train-clean-100.tar.gz
mv LibriSpeech data/LibriSpeech_train

tar -xvf dev-clean.tar.gz
mv LibriSpeech data/LibriSpeech_dev

tar -xvf test-clean.tar.gz
mv LibriSpeech data/LibriSpeech_test
```
