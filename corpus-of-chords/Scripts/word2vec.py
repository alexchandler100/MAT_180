import io
import tensorflow as tf
import numpy as np
from keras import layers
from keras import preprocessing
import random
import dataProsessing as dp
import MidiWriting as mw

import json as json

#this class, and only this class, shamelessly copy-pasted from the tensorflow W2Vec tutorial
class Word2Vec(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, num_ns):
    super(Word2Vec, self).__init__()
    self.target_embedding = layers.Embedding(vocab_size,
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding")
    self.context_embedding = layers.Embedding(vocab_size,
                                       embedding_dim,
                                       input_length=num_ns+1)

  def call(self, pair):
    target, context = pair
   
    if len(target.shape) == 2:
      target = tf.squeeze(target, axis=1)
    
    word_emb = self.target_embedding(target)
    
    context_emb = self.context_embedding(context)
    
    dots = tf.einsum('be,bce->bc', word_emb, context_emb)
    
    return dots


def generateTrainingData(sequence, vocab_size , window_size, negative_samples):
    #takes vectorized text, generates context,target,label tensors
    positive_skips = preprocessing.sequence.skipgrams(sequence, vocabulary_size=vocab_size, window_size =window_size,negative_samples=0)[0]
    target_tensor = []
    contexts_tensor = []
    labels_tensor = []

    dictionary = {}

    for s in positive_skips:
        if s[0] not in dictionary:
            dictionary[s[0]] = [s[1]]
        else:
            dictionary[s[0]] += [s[1]]
    for target, context in dictionary.items():
        
        negatives = [i for i in range(vocab_size) if i not in context]
        for c in context:
            target_tensor.append(tf.constant(target))
            contexts_tensor.append(tf.constant([c] + random.sample(negatives,negative_samples)))
            labels_tensor.append(tf.constant([1]+ [0 for i in range(negative_samples)]))

    
    return target_tensor , contexts_tensor, labels_tensor






