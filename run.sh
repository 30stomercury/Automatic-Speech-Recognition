unit="subword"
size=5000

# save dir
subword_dir="subword/bpe_5k/"
feat_dir="data/features_mfcc_bpe_5k/"
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
			   --feat_dir $feat_dir

python3 train_tfrecord.py --lr 0.00025 \
			  --epoch 3000 \
			  --enc_units 512 \
			  --dec_units 512 \
			  --embedding_size 256 \
			  --attention_size 128 \
			  --feat_dim 13 \
			  --dropout_rate 0 \
			  --num_enc_layers 4 \
			  --num_dec_layers 1 \
			  --subword_dir $subword_dir \
			  --summary_dir $summary_dir \
			  --save_dir $model_dir \
			  --max_step 1500000 \
			  --min_rate 0.8 \
			  --warmup_step 1000000 \
			  --augmentation False \
			  --verbose 1 
