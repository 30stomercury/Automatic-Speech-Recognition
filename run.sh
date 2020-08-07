# setup
unit="subword"
size=5000
feat_type="mfcc"

# train
epoch=100

# inference
restore=100
convert_rate=0.12
beam_size=8

# save dir
subword_dir="subword/bpe_5k/"
feat_dir="data/features_mfcc_bpe_5k/"
tfrecord_dir="data/tfrecord_mfcc_bpe_5k/"
model_dir="model/las/"
summary_dir="summary/summary/"

# corpus dir
libri_100hr_dir="data/100/LibriSpeech_train/train-clean-100/"
libri_360hr_dir="data/360/LibriSpeech_train/train-clean-360/"
libri_500hr_dir="data/500/LibriSpeech_train/train-other-500/"



if [ $unit = "subword" ]
then
    echo "$0: Train subword"
    python3 train_subword.py --save_dir $subword_dir \
	    		     --size $size \
			     --train_100hr_corpus_dir $libri_100hr_dir \
			     --train_360hr_corpus_dir $libri_360hr_dir \
			     --train_500hr_corpus_dir $libri_500hr_dir 
fi


echo "$0: Data Preparation"
python3 preprocess.py --unit $unit \
		      --feat_type $feat_type \
		      --train_100hr_corpus_dir $libri_100hr_dir \
		      --train_360hr_corpus_dir $libri_360hr_dir \
		      --train_500hr_corpus_dir $libri_500hr_dir \
		      --dev_data_dir data/eval/LibriSpeech_dev/dev-clean/ \
		      --test_data_dir data/eval/LibriSpeech_test/test-clean/ \
		      --feat_dir $feat_dir \
		      --subword_dir $subword_dir \
		      --augmentation False


echo "$0: TFrecord Preparation"
python3 create_tfrecord.py --unit $unit \
			   --feat_dir $feat_dir \
			   --save_dir $tfrecord_dir

echo "$0: LAS Training"
python3 train.py --lr 0.0001 \
		 --epoch $epoch \
		 --feat_dim 13 \
		 --enc_units 512 \
		 --dec_units 1024 \
		 --embedding_size 256 \
		 --attention_size 128 \
		 --num_enc_layers 4 \
		 --num_dec_layers 2 \
		 --mode loc \
		 --dropout_rate 0 \
		 --grad_clip 5 \
		 --schduled_sampling False \
		 --augmentation False \
		 --save_dir $model_dir \
		 --summary_dir $summary_dir \
		 --subword_dir $subword_dir \
		 --verbose 1 

echo "$0: Decoding"
python3 decode.py --feat_dim 13 \
		  --enc_units 512 \
       	 	  --dec_units 1024 \
		  --embedding_size 256 \
		  --attention_size 128 \
		  --num_enc_layers 4 \
		  --num_dec_layers 2 \
		  --mode loc \
		  --save_dir $model_dir \
		  --subword_dir $subword_dir
		  --feat_dir $feat_dir \
		  --restore_epoch $restore \
		  --convert_rate $convert_rate \
	          --verbose 1 
