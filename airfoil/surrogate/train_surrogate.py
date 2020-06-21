
"""
Trains a surrogate model

Author(s): Wei Chen (wchen459@umd.edu)
"""

import argparse
import numpy as np
import tensorflow as tf

from surrogate_model import Model, preprocess, postprocess

import sys
sys.path.append('../..')
from utils import ElapsedTimer


if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('mode', type=str, default='train', help='train or evaluate')
    parser.add_argument('--save_interval', type=int, default=1000, help='save interval')
    args = parser.parse_args()
    assert args.mode in ['train', 'evaluate']
    
    train_steps = 10000
    batch_size = 256
    lr = 0.0001
    
    # Read dataset
    xs_train_fname = '../data/xs_train.npy'
    ys_train_fname = '../data/ys_train.npy'
    xs_test_fname = '../data/xs_test.npy'
    ys_test_fname = '../data/ys_test.npy'
    X_train = np.load(xs_train_fname)
    Y_train = np.load(ys_train_fname)
    X_test = np.load(xs_test_fname)
    Y_test = np.load(ys_test_fname)
    
    # Scale y
    Y = np.concatenate([Y_train, Y_test])
    min_y = np.min(Y)
    max_y = np.max(Y)
    Y_train = (Y_train-min_y)/(max_y-min_y)
    Y_test = (Y_test-min_y)/(max_y-min_y)
    
#    X_train = np.random.rand(8429, 1, 2)
#    Y_train = np.mean(X_train, axis=(1,2))
#    X_test = np.random.rand(2107, 1, 2)
#    Y_test = np.mean(X_test, axis=(1,2))
    
    directory = './trained_surrogate'
    with tf.Session() as sess:
        
        model = Model(sess, X_train.shape[1])
        if args.mode == 'train':
            # Train
            timer = ElapsedTimer()
            model.train(X_train, Y_train, X_test, Y_test, batch_size=batch_size, train_steps=train_steps, lr=lr,
                        save_interval=args.save_interval, directory=directory)
            elapsed_time = timer.elapsed_time()
            runtime_mesg = 'Wall clock time for training: %s' % elapsed_time
            print(runtime_mesg)
        else:
            model.restore(directory=directory)
            
        n = 5
        ind = np.random.choice(X_test.shape[0], size=n, replace=False)
        scores = model.predict(X_test[ind])
        print(scores)
        print(Y_test[ind])
            
#        g = tf.gradients(model.y_pred, model.x, unconnected_gradients='zero')
#        airfoils = preprocess(np.array(X_test[:5], ndmin=3))
#        gradients = sess.run(g, feed_dict={model.x: airfoils, model.training: False})[0]
#        print(gradients)
#        print(gradients.shape)
        
#        model.restore_frozen_graph(directory=directory)
#        
#    # Test model
#    n = 5
##    scores = model.predict(X_test[:n])
#    with tf.Session(graph=model.frozen_graph) as sess:
#        airfoils = preprocess(np.array(X_test[:n], ndmin=3))
#        scores = sess.run(model.y_pred, feed_dict={model.x: airfoils, model.training: False})
#    print(postprocess(scores))
#    print(Y_test[:n])
#    g = tf.gradients(model.y_pred, model.x, unconnected_gradients='zero')
#    with tf.Session(graph=model.frozen_graph) as sess:
#        gradients = sess.run(g, feed_dict={model.x: airfoils, model.training: False})[0]
#    print(gradients)
#    print(gradients.shape)
