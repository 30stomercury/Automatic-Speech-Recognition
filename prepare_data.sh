DIR="data/"


if [ -d "$DIR" ]
then
    # Download dataset if $DIR exists. #
    echo "Download dataset to ${DIR}..."
else
    mkdir data
fi

if [ $1 = "LibriSpeech-100" ]
then
    echo "Downloading LibriSpeech-100"
    wget http://www.openslr.org/resources/12/train-clean-100.tar.gz
    tar -xvf train-clean-100.tar.gz

elif [ $1 = "LibriSpeech-360" ]
then
    echo "Downloading LibriSpeech-360"
    wget http://www.openslr.org/resources/12/train-clean-360.tar.gz
    tar -xvf train-clean-360.tar.gz

else
    echo "Please choose LibriSpeech-100 or -360"
fi

echo "Downloading LibriSpeech test/dev set"

wget http://www.openslr.org/resources/12/dev-clean.tar.gz
wget http://www.openslr.org/resources/12/test-clean.tar.gz

mkdir data/$1
mv LibriSpeech data/$1/LibriSpeech_train

tar -xvf dev-clean.tar.gz
mv LibriSpeech data/$1/LibriSpeech_dev

tar -xvf test-clean.tar.gz
mv LibriSpeech data/$1/LibriSpeech_test
