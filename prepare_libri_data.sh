DIR="data/"
SOURCE=http://www.openslr.org/resources/12


if [ -d "$DIR" ]
then
    # Download dataset if $DIR exists. #
    echo "Download dataset to ${DIR}..."
else
    mkdir data
fi

for target in train-clean-100.tar.gz train-clean-360.tar.gz train-other-500.tar.gz
do
    echo "Downloading $target"
    wget $SOURCE/$target
    echo "Extracting..."
    tar -xvf $target

    hr=${target:12:3}
    mkdir data/$hr
    mv LibriSpeech data/$hr/LibriSpeech_train
done

echo "Downloading LibriSpeech test/dev set"

wget $SOURCE/dev-clean.tar.gz
wget $SOURCE/test-clean.tar.gz


tar -xvf dev-clean.tar.gz
mv LibriSpeech data/LibriSpeech_eval/LibriSpeech_dev

tar -xvf test-clean.tar.gz
mv LibriSpeech data/LibriSpeech_eval/LibriSpeech_test


