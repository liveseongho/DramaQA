# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""A simple demonstration of running VGGish in inference mode.

This is intended as a toy example that demonstrates how the various building
blocks (feature extraction, model definition and loading, postprocessing) work
together in an inference context.

A WAV file (assumed to contain signed 16-bit PCM samples) is read in, converted
into log mel spectrogram examples, fed into VGGish, the raw embedding output is
whitened and quantized, and the postprocessed embeddings are optionally written
in a SequenceExample to a TFRecord file (using the same format as the embedding
features released in AudioSet).

Usage:
  # Run a WAV file through the model and print the embeddings. The model
  # checkpoint is loaded from vggish_model.ckpt and the PCA parameters are
  # loaded from vggish_pca_params.npz in the current directory.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file

  # Run a WAV file through the model and also write the embeddings to
  # a TFRecord file. The model checkpoint and PCA parameters are explicitly
  # passed in as well.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file \
                                    --tfrecord_file /path/to/tfrecord/file \
                                    --checkpoint /path/to/model/checkpoint \
                                    --pca_params /path/to/pca/params

  # Run a built-in input (a sine wav) through the model and print the
  # embeddings. Associated model files are read from the current directory.
  $ python vggish_inference_demo.py
"""

from __future__ import print_function

import glob, os
import numpy as np
import tensorflow.compat.v1 as tf

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

import json

PCA_PARAMS = 'vggish_pca_params.npz'
CHECKPOINT = 'vggish_model.ckpt'

config = json.load(open('../preprocess_config.json', 'r', encoding='utf-8'))
segmented_audio_dir = config['segmented_audio_dir']
vggish_dir = config['vggish_dir']


os.makedirs(vggish_dir, exist_ok=True)

def main(_):
    # In this simple example, we run the examples from a single audio file through
    # the model. If none is provided, we generate a synthetic input.

    # Prepare a postprocessor to munge the model embeddings.
    pproc = vggish_postprocess.Postprocessor(PCA_PARAMS)

    with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, CHECKPOINT)
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)


        for idx, wav_file in enumerate(sorted(glob.glob(segmented_audio_dir + '*.wav'))):

            batch = vggish_input.wavfile_to_examples(wav_file)
            if batch.shape[0] == 0:
                print('batch:', batch.shape)
                print(batch)

            # Run inference and postprocessing.
            [embedding_batch] = sess.run([embedding_tensor],
                                         feed_dict={features_tensor: batch})

            # print('embedding_batch:', embedding_batch.shape)
            postprocessed_batch = pproc.postprocess(embedding_batch)
            # print('postprocessed_batch:', postprocessed_batch.shape)

            npy_path = vggish_dir + \
                        wav_file.split('/')[-1].replace('.wav', '.npy')

            np.save(npy_path, postprocessed_batch)


            if idx % 1 == 0:
                print('idx:', idx, '\t', wav_file, ':', postprocessed_batch.shape)



if __name__ == '__main__':
    tf.app.run()
