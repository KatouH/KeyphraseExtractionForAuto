from bert import extract_features
import tensorflow as tf
import json
import os

_get_path = lambda path:os.path.normpath(os.path.join(os.getcwd(),os.path.dirname(__file__),path))

bert_config = _get_path("../../Model/chinese_L-12_H-768_A-12/bert_config.json")
init_checkpoint = _get_path("../../Model/chinese_L-12_H-768_A-12/bert_model.ckpt")
vocab_file = _get_path("../../Model/chinese_L-12_H-768_A-12/vocab.txt")
input_file = _get_path("../../DataSet/train1.txt")
output_file = _get_path("../../Temp/output.json")
max_seq_length = 100
batch_size = 8
layers = "-2"

extract_features.FLAGS.layers = layers
extract_features.FLAGS.bert_config_file=bert_config
extract_features.FLAGS.init_checkpoint=init_checkpoint
extract_features.FLAGS.input_file=input_file
extract_features.FLAGS.output_file=output_file
extract_features.FLAGS.vocab_file=vocab_file
extract_features.FLAGS.max_seq_length=max_seq_length
extract_features.FLAGS.batch_size=batch_size

extract_features.main(None)

bert_res = list()
with tf.gfile.GFile(extract_features.FLAGS.output_file,'r') as reader:
    for line in reader.readlines():
        bert_res.append(json.loads(line))
