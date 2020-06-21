"""
BezierGAN for capturing the airfoil manifold

Author(s): Wei Chen (wchen459@umd.edu)
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

#from normalization import WeightNorm


def preprocess(X):
    X = np.expand_dims(X, axis=-1)
    return X.astype(np.float32)

def postprocess(X):
    X = np.squeeze(X)
    return X

EPSILON = 1e-7

class Model(object):
    
    def __init__(self, sess, n_points=64):

        self.sess = sess
        self.X_shape = (n_points, 2, 1)
        
    def residual_block(self, x_init, channels, kernel_size, kernel_initializer, kernel_regularizer, 
                       training=True, downsample=False, scope='bottle_resblock'):
        
        with tf.variable_scope(scope):
            
#            x = tf.layers.batch_normalization(x_init, momentum=0.9, training=training)
#            shortcut = tf.nn.leaky_relu(x, alpha=0.2)
#    
#            x = tf.layers.conv2d(shortcut, channels, kernel_size=1, strides=1, padding='same', 
#                                 kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
#            x = tf.layers.batch_normalization(x, momentum=0.9, training=training)
#            x = tf.nn.leaky_relu(x, alpha=0.2)
#    
#            if downsample :
#                x = tf.layers.conv2d(x, channels, kernel_size, strides=(2,1), padding='same', 
#                                     kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
#                shortcut = tf.layers.conv2d(shortcut, channels*4, kernel_size=1, strides=(2,1), padding='same', 
#                                            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
#    
#            else :
#                x = tf.layers.conv2d(x, channels, kernel_size, strides=1, padding='same', 
#                                     kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
#                shortcut = tf.layers.conv2d(shortcut, channels*4, kernel_size=1, strides=1, padding='same', 
#                                            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
#    
#            x = tf.layers.batch_normalization(x, momentum=0.9, training=training)
#            x = tf.nn.leaky_relu(x, alpha=0.2)
#            x = tf.layers.conv2d(x, channels*4, kernel_size=1, strides=1, padding='same', 
#                                 kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
#    
#            return x + shortcut
            
            x = tf.layers.batch_normalization(x_init, momentum=0.9, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
    
    
            if downsample :
                x = tf.layers.conv2d(x, channels, kernel_size, strides=(2,1), padding='same', 
                                     kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
                x_init = tf.layers.conv2d(x_init, channels, kernel_size=1, strides=(2,1), padding='same', 
                                          kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
    
            else :
                x = tf.layers.conv2d(x, channels, kernel_size, strides=1, padding='same', 
                                     kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
    
            x = tf.layers.batch_normalization(x, momentum=0.9, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.conv2d(x, channels, kernel_size, strides=1, padding='same', 
                                 kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
            
            return x + x_init
        
    def net(self, x, training=True, reuse=tf.AUTO_REUSE):
        
        depth = 16
#        dropout = 0.4
        kernel_size = (4,2)
        residual_list = [2, 2, 2, 2]
#        residual_list = [3, 4, 6, 3]
        weight_init = tf.contrib.layers.variance_scaling_initializer()
        weight_regularizer = tf.contrib.layers.l2_regularizer(0.0001)
        
        with tf.variable_scope('net', reuse=reuse):
        
##            x = WeightNorm(tf.layers.Conv2D(depth*1, kernel_size, strides=(2,1), padding='same', activation='relu'))(x)
#            x = tf.layers.conv2d(x, depth*1, kernel_size, strides=(2,1), padding='same')
#            x = tf.layers.batch_normalization(x, momentum=0.9, training=training)
#            x = tf.nn.leaky_relu(x, alpha=0.2)
##            x = tf.layers.dropout(x, dropout, training=training)
#            
##            x = WeightNorm(tf.layers.Conv2D(depth*2, kernel_size, strides=(2,1), padding='same', activation='relu'))(x)
#            x = tf.layers.conv2d(x, depth*2, kernel_size, strides=(2,1), padding='same')
#            x = tf.layers.batch_normalization(x, momentum=0.9, training=training)
#            x = tf.nn.leaky_relu(x, alpha=0.2)
##            x = tf.layers.dropout(x, dropout, training=training)
#            
##            x = WeightNorm(tf.layers.Conv2D(depth*4, kernel_size, strides=(2,1), padding='same', activation='relu'))(x)
#            x = tf.layers.conv2d(x, depth*4, kernel_size, strides=(2,1), padding='same')
#            x = tf.layers.batch_normalization(x, momentum=0.9, training=training)
#            x = tf.nn.leaky_relu(x, alpha=0.2)
##            x = tf.layers.dropout(x, dropout, training=training)
#            
##            x = WeightNorm(tf.layers.Conv2D(depth*8, kernel_size, strides=(2,1), padding='same', activation='relu'))(x)
#            x = tf.layers.conv2d(x, depth*8, kernel_size, strides=(2,1), padding='same')
#            x = tf.layers.batch_normalization(x, momentum=0.9)#, training=training)
#            x = tf.nn.leaky_relu(x, alpha=0.2)
##            x = tf.layers.dropout(x, dropout, training=training)
#            
##            x = WeightNorm(tf.layers.Conv2D(depth*16, kernel_size, strides=(2,1), padding='same', activation='relu'))(x)
#            x = tf.layers.conv2d(x, depth*16, kernel_size, strides=(2,1), padding='same')
#            x = tf.layers.batch_normalization(x, momentum=0.9, training=training)
#            x = tf.nn.leaky_relu(x, alpha=0.2)
##            x = tf.layers.dropout(x, dropout, training=training)
#            
##            x = WeightNorm(tf.layers.Conv2D(depth*32, kernel_size, strides=(2,1), padding='same', activation='relu'))(x)
#            x = tf.layers.conv2d(x, depth*32, kernel_size, strides=(2,1), padding='same')
#            x = tf.layers.batch_normalization(x, momentum=0.9, training=training)
#            x = tf.nn.leaky_relu(x, alpha=0.2)
##            x = tf.layers.dropout(x, dropout, training=training)
#            
#            x = tf.layers.flatten(x)
##            x = WeightNorm(tf.layers.Dense(1024, activation='relu'))(x)
#            x = tf.layers.dense(x, 1024)
#            x = tf.layers.batch_normalization(x, momentum=0.9, training=training)
#            x = tf.nn.leaky_relu(x, alpha=0.2)
#            
##            y = WeightNorm(tf.layers.Dense(1, activation='sigmoid'))(x)
#            y = tf.layers.dense(x, 1)
#            y = tf.nn.sigmoid(y, name='y')
            
            x = tf.layers.conv2d(x, depth*1, kernel_size, strides=1, padding='same', 
                                 kernel_initializer=weight_init, kernel_regularizer=weight_regularizer)

            for i in range(residual_list[0]) :
                x = self.residual_block(x, channels=depth, kernel_size=kernel_size, training=training, downsample=False, 
                                        kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, 
                                        scope='resblock0_' + str(i))

            ########################################################################################################

            x = self.residual_block(x, channels=depth*2, kernel_size=kernel_size, training=training, downsample=True, 
                                    kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, 
                                    scope='resblock1_0')

            for i in range(1, residual_list[1]) :
                x = self.residual_block(x, channels=depth*2, kernel_size=kernel_size, training=training, downsample=False, 
                                        kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, 
                                        scope='resblock1_' + str(i))

            ########################################################################################################

            x = self.residual_block(x, channels=depth*4, kernel_size=kernel_size, training=training, downsample=True, 
                                    kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, 
                                    scope='resblock2_0')

            for i in range(1, residual_list[2]) :
                x = self.residual_block(x, channels=depth*4, kernel_size=kernel_size, training=training, downsample=False, 
                                        kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, 
                                        scope='resblock2_' + str(i))

            ########################################################################################################

            x = self.residual_block(x, channels=depth*8, kernel_size=kernel_size, training=training, downsample=True, 
                                    kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, 
                                    scope='resblock_3_0')

            for i in range(1, residual_list[3]) :
                x = self.residual_block(x, channels=depth*8, kernel_size=kernel_size, training=training, downsample=False, 
                                        kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, 
                                        scope='resblock_3_' + str(i))

            ########################################################################################################

            x = tf.layers.batch_normalization(x, momentum=0.9, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)

            x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, 128, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer)
            x = tf.layers.batch_normalization(x, momentum=0.9, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            
            y = tf.layers.dense(x, 1, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer)
            y = tf.nn.sigmoid(y, name='y')
            
#            init = tf.constant_initializer([[0.5],[0.5]])
            
#            x = tf.layers.flatten(x)
#            y = tf.layers.dense(x, 1, use_bias=False)#, kernel_initializer=init)
            
            return y
        
    def train(self, X_train, Y_train, X_test, Y_test, train_steps=2000, batch_size=256, lr=0.001, save_interval=0, directory='.'):
            
        X_train = preprocess(X_train)
        X_test = preprocess(X_test)
        Y_train = Y_train.reshape(-1,1)
        Y_test = Y_test.reshape(-1,1)
        
        # Inputs
        self.x = tf.placeholder(tf.float32, shape=(None,)+self.X_shape, name='x')
        self.training = tf.placeholder(tf.bool, name='training')
        
        # Targets
        y_true = tf.placeholder(tf.float32, shape=(None,1))
        
        # Outputs
        self.y_pred = self.net(self.x, self.training)
        
        # Loss
        loss = tf.reduce_mean(tf.abs(y_true-self.y_pred))
        
        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
#        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        
        # Training operations
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)
        
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        
        # Create summaries to monitor losses
        tf.summary.scalar('loss', loss)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        # Run the initializer
        self.sess.run(init)
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter('{}/logs'.format(directory), graph=self.sess.graph)
    
        for t in range(train_steps):
    
            ind = np.random.choice(X_train.shape[0], size=batch_size, replace=False)
            X_batch = X_train[ind]
            Y_batch = Y_train[ind]
            l, _, summary_str = self.sess.run([loss, train_op, merged_summary_op], 
                                              feed_dict={self.x: X_batch, y_true: Y_batch, self.training: True})
            
            l_test = self.sess.run(loss, feed_dict={self.x: X_test, y_true: Y_test, self.training: False})
        
#            print('#####################################')
#            for v in tf.trainable_variables():
#                print(v.name)
#                print(self.sess.run(v))
#            print('#####################################')
            
            assert not np.isnan(l) and not np.isnan(l_test)
            
            summary_writer.add_summary(summary_str, t+1)
            
            # Show messages
            log_mesg = "%d: train %f  test %f" % (t+1, l, l_test)
            print(log_mesg)
            
            if save_interval>0 and (t+1)%save_interval==0 or t+1 == train_steps:    
                # Save the final variables to disk.
                checkpoint_path = saver.save(self.sess, '{}/model'.format(directory))
                print('Model saved in path: %s' % checkpoint_path)
        
#        proto_save_path = tf.train.write_graph(self.sess.graph.as_graph_def(), directory, 'model.pbtxt', as_text=True)
#        print('Proto file saved in path: %s' % proto_save_path)
#        
#        self.freeze_graph(directory)
        
#    def freeze_graph(self, directory):
#        
#        # Freeze graph
#        output_graph_filename = '{}/frozen_graph.pb'.format(directory)
#        
#        checkpoint_path = '{}/model'.format(directory)
#        input_saver_def_path = ''
#        input_binary = True
#        output_node_names = "net/y"
#        restore_op_name = "save/restore_all"
#        filename_tensor_name = "save/Const:0"
#        clear_devices = False
#        input_meta_graph = checkpoint_path + ".meta"
#        
#        freeze_graph.freeze_graph(
#            '', input_saver_def_path, input_binary, checkpoint_path,
#            output_node_names, restore_op_name, filename_tensor_name,
#            output_graph_filename, clear_devices, '', '', '', input_meta_graph)
#
#        print('Frozen graph saved in path : %s' % output_graph_filename)
#        
##        # Optimize for inference
##        inputGraph = tf.GraphDef()
##        with tf.gfile.Open(output_graph_filename, 'rb') as f:
##            data2read = f.read()
##            inputGraph.ParseFromString(data2read)
##          
##        outputGraph = optimize_for_inference_lib.optimize_for_inference(
##                      inputGraph,
##                      ['x', 'training'], # an array of the input node(s)
##                      ['net/y'], # an array of output nodes
##                      tf.float32.as_datatype_enum)
##        
##        # Save the optimized graph
##        optimized_frozen_graph_filename = '{}/frozen_graph_optimized.pb'.format(directory)
##        with tf.gfile.GFile(optimized_frozen_graph_filename, "w") as f:
##            f.write(outputGraph.SerializeToString())
##
##        print('Otimized frozen graph saved in path : %s' % optimized_frozen_graph_filename)
                    
    def restore(self, directory='.'):
        
        print('Loading model ...')
        
        # Load meta graph and restore weights
        saver = tf.train.import_meta_graph('{}/model.meta'.format(directory))
        saver.restore(self.sess, tf.train.latest_checkpoint('{}/'.format(directory)))
        
        # Access and create placeholders variables            
        graph = tf.get_default_graph()
        self.x = graph.get_tensor_by_name('x:0')
        self.training = graph.get_tensor_by_name('training:0')
        self.y_pred = graph.get_tensor_by_name('net/y:0')
        
        return graph
        
#    def restore_frozen_graph(self, directory='.'):
#        
#        print('Loading frozen graph ...')
#        frozen_graph_filename = '{}/frozen_graph.pb'.format(directory)
#        
#        with tf.gfile.GFile(frozen_graph_filename,'rb') as f:
#            graph_def = tf.GraphDef()
#            graph_def.ParseFromString(f.read())
#            
#        with tf.Graph().as_default() as graph:
#            self.x, self.training, self.y_pred = tf.import_graph_def(
#                                                    graph_def,
#                                                    input_map=None,
#                                                    return_elements=['x:0', 'training:0', 'net/y:0'],
#                                                    name='')
#            self.frozen_graph = graph
            
    def predict(self, airfoils):
        airfoils = preprocess(np.array(airfoils, ndmin=3))
        scores = self.sess.run(self.y_pred, feed_dict={self.x: airfoils, self.training: False})
        return postprocess(scores)
    