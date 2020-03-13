DIR="data/"


if [ -d "$DIR" ]
then
    # Download dataset if $DIR exists. #
    echo "Download dataset to ${DIR}..."
else
    mkdir data
fi

if [ $1 = "TED-LIUMv1" ]
then
    echo "Downloading TED-LIUMv1"
    wget http://www.openslr.org/resources/7/TEDLIUM_release1.tar.gz
    echo "Extracting..."
    tar -xvf TEDLIUM_release1.tar.gz

elif [ $1 = "TED-LIUMv2" ]
then
    echo "Downloading TED-LIUMv2"
    wget http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz
    echo "Extracting..."
    tar -xvf TEDLIUM_release1.tar.gz

else
    echo "Please choose TED-LIUMv1 or TED-LIUMv2"
fi


echo "install sox..."
sudo apt-get install sox

echo "convert sph to wav"
find -type f -name '*.sph' | awk '{printf "sox -t sph %s -b 16 -t wav %s\n", $0, $0".wav" }' | bash

echo "Clean test, dev, train texts..."
for set in dev test train; do
    dir=data/TEDLIUM_release1/$set
    mkdir -p $dir
    {
        # process stms
        cat data/TEDLIUM_release1/$set/stm/*.stm | \
               sed -e 's/<sil>//g'   \
                   -e 's/([0-9])//g' \
                   -e 's/{SMACK}//g' \
                   -e 's/{NOISE}//g' \
                   -e 's/{BREATH}//g' \
                   -e 's/{COUGH}//g' > data/TEDLIUM_release1/$set/stm/clean.stm
    }
done

