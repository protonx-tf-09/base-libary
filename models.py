import tensorflow as tf
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from layers import EncoderLayer
from layers import positional_encoding

class TransformerClassifier(tf.keras.Model):
    def __init__(self, num_encoder_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(TransformerClassifier, self).__init__()
        self.num_encoder_layers = num_encoder_layers
        self.embedding = Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_encoder_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        self.global_average_pooling = GlobalAveragePooling1D()
        self.final_layer = Dense(1, activation='sigmoid')

    def call(self, x, training):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_encoder_layers):
            x = self.enc_layers[i](x, training, None)
        x = self.global_average_pooling(x)
        return self.final_layer(x)