import tensorflow as tf
import numpy as np


def label_smoothing(inputs, epsilon=0.01):
    """label smoothing

    Reference: https://github.com/Kyubyong/transformer/blob/master/tf1.2_legacy/modules.py
    """

    K = inputs.get_shape().as_list()[-1] 
    return ((1-epsilon) * inputs) + (epsilon / K)


def convert_idx_to_token_tensor(inputs, id_to_token, unit="char"):
    """Converts int32 tensor to string tensor.

    Reference:
        https://github.com/Kyubyong/transformer/blob/master/utils.py
    """

    def my_func(inputs):
        sent = "".join(id_to_token[elem] for elem in inputs)
        sent = sent.split("<EOS>")[0].strip()
        if unit == "char":
            # replace <SPACE> with " "
            sent = sent.replace("<SPACE>", " ") 
        elif unit == "subword":
            # replace the suffix </w> with " "
            sent = sent.replace("</w>", " ") 
        return " ".join(sent.split())

    return tf.py_func(my_func, [inputs], tf.string)

def convert_idx_to_string(inputs, id_to_token, unit="char"):
    """Converts int32 ndarray to string. (char or subword tokens)"""

    sent =  "".join(id_to_token[elem] for elem in inputs)
    sent = sent.split("<EOS>")[0].strip()
    if unit == "char":
        # replace <SPACE> with " "
        sent = sent.replace("<SPACE>", " ") 
    elif unit == "subword":
        # replace the suffix </w> with " "
        sent = sent.replace("</w>", " ") 
    return " ".join(sent.split())

def wer(s1,s2):
    
    e, length = edit_distance(s1, s2)
    
    return e / length

def edit_distance(s1,s2):

    d = np.zeros([len(s1)+1,len(s2)+1])
    d[:,0] = np.arange(len(s1)+1)
    d[0,:] = np.arange(len(s2)+1)

    for j in range(1,len(s2)+1):
        for i in range(1,len(s1)+1):
            if s1[i-1] == s2[j-1]:
                d[i,j] = d[i-1,j-1]
            else:
                d[i,j] = min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+1)

    return d[-1,-1], len(s1)

def get_save_vars():
    """Get all variables needed to be save."""

    var_list = [var for var in tf.global_variables() if "moving" in var.name]                       # include moving var, mean
    var_list += tf.trainable_variables()                                                            # trainable var
    var_list += [var for var in tf.global_variables() if "Adam" in var.name or "step" in var.name]  # adam parms
    var_list += [var for var in tf.global_variables() if "beta1_power" in var.name or "beta2_power" in var.name]

    return var_list
