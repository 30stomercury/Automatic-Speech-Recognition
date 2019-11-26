from tensor2tensor.layers import common_audio
#from tensor2tensor.layers import common_layers
import tensorflow as tf
import librosa
import numpy as np
from pathlib import Path
import time

libri_path = './data/LibriSpeech/dev-clean'
dir_path = Path(libri_path)
audio_list = [f for f in dir_path.glob('**/*.flac') if f.is_file()] 
audio, fs = librosa.load(audio_list[2])
audio = np.array(audio)
print(audio.shape, fs)
max_len = len(audio)
audio = np.hstack((audio, np.zeros((max_len - audio.shape[0]), dtype=np.float32)))
print(audio.shape)
audio = tf.expand_dims(audio, 0)
mel_fbanks = common_audio.compute_mel_filterbank_features(audio, dither=0.0, sample_rate=fs, frame_step=10, num_mel_bins=40, apply_mask=True)
#fbank_size = common_layers.shape_list(mel_fbanks)
output = tf.reduce_sum(mel_fbanks, -1)
sess = tf.Session()
print(sess.run(output).shape)
print(sess.run(output))
#print(sess.run(output)[:,870,:])
#print(sess.run(output)[:,872:,:])
waveforms = audio
wav_lens = tf.reduce_max(
  tf.expand_dims(tf.range(tf.shape(waveforms)[1]), 0) *
  tf.to_int32(tf.not_equal(waveforms, 0.0)),
  axis=-1) + 1 
print(sess.run(wav_lens))

def load_audio(path, sess, prepro_batch=128, frame_step=10, sample_rate=22050, feat_type='fbank'):
    
    input_audio = tf.placeholder(dtype=tf.float32, shape=[None, None])
    # extract audio feature:
    if feat_type == 'fbank':
        mel_fbanks = common_audio.compute_mel_filterbank_features(
            input_audio, dither=0.0, sample_rate=sample_rate, frame_step=frame_step, num_mel_bins=40, apply_mask=True)
        mel_fbanks = tf.reduce_sum(mel_fbanks, -1)

    def extract_feat(audio_batch, len_batch, fs):
        max_len = max(len_batch)
        audio_padded = np.zeros([prepro_batch, max_len], dtype=np.float32)
        for i in range(len(audio_batch)):
            audio_padded[i][:len(audio_batch[i])] = audio_batch[i]
        feat = sess.run(mel_fbanks, feed_dict={input_audio: audio_padded})
        # find the feature length:
        feat_len = np.array(len_batch) // int(fs * frame_step / 1e3) + 1
        feat_len = feat_len.astype(np.int32)
        return feat, feat_len
        
    audio_batch = []
    len_batch = []
    feats_list = []
    length_list = []

    dir_path = Path(path)
    audio_list = [f for f in dir_path.glob('**/*.flac') if f.is_file()] 
    for p in audio_list:
        audio, fs = librosa.load(p)
        audio_batch.append(audio)
        len_batch.append(len(audio))
        if len(audio_batch) == prepro_batch:
            feat, feat_len = extract_feat(audio_batch, len_batch, fs)
            # remove paddings of audios batch:
            for index, l in enumerate(feat_len):
                #feats.append(mel_fbanks[index][:l])
                feats_list.append(feat[index][:l])
            length_list = np.concatenate([length_list, feat_len])
            audio_batch = []
            len_batch = []
            print("Processed samples: {}/{}".format(len(feats_list), len(audio_list)))

    if len(audio_batch) % prepro_batch != 0:
        feat, feat_len = extract_feat(audio_batch, len_batch, fs)
        # remove paddings of rest audios:
        for index, l in enumerate(feat_len):
            #feats.append(mel_fbanks[index][:l])
            feats_list.append(feat[index][:l])
        length_list = np.concatenate([length_list, feat_len])
    return feats_list, length_list.astype(np.int32)

def batch_gen(feats_list, length_list, batch_size=5):

    assert len(feats_list) == len(length_list)

    def generator():
        buff = []
        for i, x in enumerate(feats_list):
            if i % batch_size == 0:
                max_len = max(length_list[i:i+batch_size])
            
            if i % batch_size == 0 and buff:
                yield np.stack(buff, 0)
                buff = []
            padded = np.zeros([max_len - x.shape[0], x.shape[1]], dtype=np.float32)
            x_padded = np.concatenate([x, padded], 0)
            buff.append(x_padded)

        if buff:
            yield np.stack(buff, 0)

    shapes = ([None, None, None])
    dataset = tf.data.Dataset.from_generator(generator,
                                             output_types=tf.float32, 
                                             output_shapes=shapes)
    dataset.batch(batch_size).prefetch(1)
    dataset = dataset.shuffle(128*batch_size)
    dataset = dataset.repeat() 
    iter_ = dataset.make_initializable_iterator()
    x = iter_.get_next()
    return iter_, x, dataset


#X = load_audio(audio_list[:500], prepro_batch=20)
#X_out = sess.run(X)
#print(len(X))
#print(len(X_out[2]), X_out[2])

#def load_text(path):
if __name__ == "__main__":
    s = time.time()
    libri_path = './data/LibriSpeech/dev-clean'
    X, X_len = load_audio(libri_path, sess, prepro_batch=32)
    print(X[0].shape, X[1].shape)
    #X_out = sess.run(X)
    iter_, x, dataset = batch_gen(X, X_len)
    print("output_shape", dataset.output_shapes)
    sess.run(tf.global_variables_initializer())
    sess.run(iter_.initializer)
    print(sess.run(x).shape)
    for _ in range(100):
        print(sess.run(x).shape)
    print(time.time()-s)
