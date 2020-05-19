'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''
import numpy as np
import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import argparse
# The following import and function call are the only additions to code required
# to automatically log metrics and parameters to MLflow.
import mlflow.keras

mlflow.keras.autolog()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=10, help='epoch')
    parser.add_argument('--max_words', default=1000, type=int, help='max words')
    parser.add_argument('--batch_size', default=32, help='batch size')
    args = vars(parser.parse_args())
    epoch = int(args['epoch'])
    max_words = int(args['max_words'])
    batch_size = int(args['batch_size'])

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words,
                                                         test_split=0.2)

    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    num_classes = np.max(y_train) + 1
    print(num_classes, 'classes')

    print('Vectorizing sequence data...')
    tokenizer = Tokenizer(num_words=max_words)
    x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
    x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    print('Building model...')
    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epoch,
                    verbose=1,
                    validation_split=0.1)
    score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # The Estimator periodically generates "INFO" logs; make these logs visible.
