import os
from argparse import ArgumentParser
from models import TransformerClassifier

import tensorflow as tf
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D

if __name__ == "__main__":
    parser = ArgumentParser()
    
    # FIXME
    # Arguments users used when running command lines
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--num-encoder-layers", default=6, type=int)
    parser.add_argument("--d-model", default=256, type=int)
    parser.add_argument("--num-heads", default=2, type=int)
    parser.add_argument("--dff", default=512, type=int)
    parser.add_argument("--vocab-size", default=10000, type=int)
    parser.add_argument("--maxlen", default=200, type=int)
    parser.add_argument("--num-epochs", default=10, type=int)

    home_dir = os.getcwd()
    args = parser.parse_args()

    num_encoder_layers = args.num_encoder_layers
    d_model = args.d_model
    num_encoder_layers = args.num_encoder_layers
    num_heads = args.num_heads
    dff = args.dff
    vocab_size = args.vocab_size
    maxlen = args.maxlen
    batch_size = args.batch_size
    num_epochs = args.num_epochs


    # FIXME
    # Project Description

    print('---------------------Welcome to ${name}-------------------')
    print('Github: ${account}')
    print('Email: ${email}')
    print('---------------------------------------------------------------------')
    print('Training ${name} model with hyper-params:') # FIXME
    print('===========================')


    # Process data

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)
    
    # FIXME
    # Do Prediction


    # Instantiate the model
    model = TransformerClassifier(
        num_encoder_layers=num_encoder_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=vocab_size,
        maximum_position_encoding=maxlen
    )

    # Compile the model
    model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=[BinaryAccuracy()])

    # Train the model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_test, y_test))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')


    # I try to hack you!!!!
