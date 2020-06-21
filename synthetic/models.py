"""
GANs

Author(s): Wei Chen (wchen459@umd.edu)
"""

import numpy as np
import tensorflow as tf

from visualization import visualize_2d

import sys
sys.path.append('..')
from utils import safe_remove


EPSILON = 1e-7


class BaseModel(object):
    
    def __init__(self, noise_dim, data_dim, lambda0=1., lambda1=0.01):

        self.noise_dim = noise_dim
        self.data_dim = data_dim
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        
    def generator(self):
        pass
        
    def discriminator(self):
        pass
    
    def compute_diversity_loss(self, x, equation, val_scale):
            
#        x_norm = tf.linalg.l2_normalize(x, 1)
#        S = tf.tensordot(x_norm, tf.transpose(x_norm), 1) # similarity matrix (cosine)
#        D = None
        
        r = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
        D = r - 2*tf.matmul(x, tf.transpose(x)) + tf.transpose(r)
        S = tf.exp(-0.5*tf.square(D)) # similarity matrix (rbf)
#        S = 1/(1+D)
        
        y = val_scale * equation(x)
        
        if self.lambda0 == 'inf':
            
            eig_val, _ = tf.self_adjoint_eig(S)
            loss = -10*tf.reduce_mean(y)
            
            Q = None
            L = None
        
        elif self.lambda0 == 'naive':
            
            eig_val, _ = tf.self_adjoint_eig(S)
            loss = -tf.reduce_mean(tf.log(tf.maximum(eig_val, EPSILON)))-10*tf.reduce_mean(y)
            
            Q = None
            L = None
            
        else:
            
            Q = tf.tensordot(tf.expand_dims(y, 1), tf.expand_dims(y, 0), 1) # quality matrix
            if self.lambda0 == 0.:
                L = S
            else:
                L = S * tf.pow(Q, self.lambda0)
            
            eig_val, _ = tf.self_adjoint_eig(L)
            loss = -tf.reduce_mean(tf.log(tf.maximum(eig_val, EPSILON)))
        
        return loss, D, S, Q, L, y
        
    def train(self):
        pass
                    
    def restore(self, save_dir='.'):
        
        self.sess = tf.Session()
        # Load meta graph and restore weights
        saver = tf.train.import_meta_graph('{}/model.meta'.format(save_dir))
        saver.restore(self.sess, tf.train.latest_checkpoint('{}/'.format(save_dir)))
        
        # Access and create placeholders variables            
        graph = tf.get_default_graph()
        self.x = graph.get_tensor_by_name('data:0')
        self.z = graph.get_tensor_by_name('noise:0')
        self.x_fake = graph.get_tensor_by_name('Generator/gen:0')

    def synthesize(self, noise):
        if isinstance(noise, int):
            N = noise
            noise = np.random.normal(scale=0.5, size=(N, self.noise_dim))
        X = self.sess.run(self.x_fake, feed_dict={self.z: noise})
        return X


class GAN(BaseModel):
        
    def generator(self, z, reuse=tf.AUTO_REUSE):
        
        with tf.variable_scope('Generator', reuse=reuse):
            
            h = tf.layers.dense(z, 128)
            h = tf.nn.leaky_relu(h, alpha=0.2)
#            h = tf.nn.tanh(h)
            
            h = tf.layers.dense(h, 128)
            h = tf.nn.leaky_relu(h, alpha=0.2)
#            h = tf.nn.tanh(h)
            
            h = tf.layers.dense(h, 128)
            h = tf.nn.leaky_relu(h, alpha=0.2)
#            h = tf.nn.tanh(h)
            
            h = tf.layers.dense(h, self.data_dim)
#            x = tf.identity(h, name='gen')
            x = tf.nn.tanh(h, name='gen')
            
            return x
        
    def discriminator(self, x, reuse=tf.AUTO_REUSE):
        
        with tf.variable_scope('Discriminator', reuse=reuse):
            
            h = tf.layers.dense(x, 128)
            h = tf.nn.leaky_relu(h, alpha=0.2)
#            h = tf.nn.tanh(h)
            
            h = tf.layers.dense(h, 128)
            h = tf.nn.leaky_relu(h, alpha=0.2)
#            h = tf.nn.tanh(h)
            
            log_d = tf.layers.dense(h, 1)
            
            return log_d
        
    def train(self, data_obj, func_obj, val_scale, train_steps=10000, batch_size=32, 
              disc_lr=2e-4, gen_lr=2e-4, save_interval=0, save_dir='.'):
        
        safe_remove('{}/logs'.format(save_dir))
        
        # Inputs
        self.x = tf.placeholder(tf.float32, shape=[None, self.data_dim], name='data')
        self.z = tf.placeholder(tf.float32, shape=[None, self.noise_dim], name='noise')
        
        # Outputs
        d_real = self.discriminator(self.x)
        self.x_fake = self.generator(self.z)
        d_fake = self.discriminator(self.x_fake)
        
        # Losses
        # Cross entropy losses for D
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))
        # Cross entropy losses for G
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))
        dpp_loss, D, S, Q, L, y = self.compute_diversity_loss(self.x_fake, func_obj.equation, val_scale)
        mean_y = tf.reduce_mean(y)
        g_dpp_loss = g_loss + self.lambda1 * dpp_loss
        
        # Optimizers
        d_optimizer = tf.train.AdamOptimizer(learning_rate=disc_lr, beta1=0.5)
        g_optimizer = tf.train.AdamOptimizer(learning_rate=gen_lr, beta1=0.5)
        
        # Generator variables
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        # Discriminator variables
        dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
        
        # Training operations
        d_train = d_optimizer.minimize(d_loss_real+d_loss_fake, var_list=dis_vars)
        g_train = g_optimizer.minimize(g_dpp_loss, var_list=gen_vars)
        
#        d_grads_real = d_optimizer.compute_gradients(d_loss_real, dis_vars)
#        d_grads_fake = d_optimizer.compute_gradients(d_loss_fake, dis_vars)
#        g_grads = g_optimizer.compute_gradients(g_loss, gen_vars)
#        dpp_grads = g_optimizer.compute_gradients(dpp_loss, gen_vars)
        
#        def clip_gradient(optimizer, loss, var_list):
#            grads_and_vars = optimizer.compute_gradients(loss, var_list)
#            clipped_grads_and_vars = [(grad, var) if grad is None else 
#                                      (tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars]
#            train_op = optimizer.apply_gradients(clipped_grads_and_vars)
#            return train_op
#        
#        d_train = clip_gradient(d_optimizer, d_loss_real+d_loss_fake, dis_vars)
#        g_train = clip_gradient(g_optimizer, g_dpp_loss, gen_vars)
        
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        
        # Create summaries to monitor losses
        tf.summary.scalar('D_loss_for_real', d_loss_real)
        tf.summary.scalar('D_loss_for_fake', d_loss_fake)
        tf.summary.scalar('G_loss', g_loss)
        tf.summary.scalar('DPP_loss', dpp_loss)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        # Start training
        self.sess = tf.Session()
        
        # Run the initializer
        self.sess.run(init)
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter('{}/logs'.format(save_dir), graph=self.sess.graph)
        
        data = data_obj.data
    
        for t in range(train_steps):
    
#            print('#################################### D_vars ####################################')
#            for var, val in zip(dis_vars, self.sess.run(dis_vars)):
#                print('D_vars before update: '+var.name, val)
                
            ind = np.random.choice(data.shape[0], size=batch_size, replace=False)
            X_real = data[ind]
            noise = np.random.normal(scale=0.5, size=(batch_size, self.noise_dim))
            _, dlr, dlf = self.sess.run([d_train, d_loss_real, d_loss_fake], feed_dict={self.x: X_real, self.z: noise})
            
#            print('#################################### D_grads_real ####################################')
#            for var, val in zip(dis_vars, self.sess.run(d_grads_real, feed_dict={self.x: X_real})):
#                print('D_grads_real: '+var.name, val)
#            print('#################################### D_grads_fake ####################################')
#            for var, val in zip(dis_vars, self.sess.run(d_grads_fake, feed_dict={self.z: noise})):
#                print('D_grads_fake: '+var.name, val)
#            print('#################################### D_vars ####################################')
#            for var, val in zip(dis_vars, self.sess.run(dis_vars)):
#                print('D_vars after update: '+var.name, val)
            
#            X_fake = self.sess.run(self.x_fake, feed_dict={self.z: noise})
#            print('************************************ X_fake ************************************')
#            print('Before G update:', X_fake)
            
#            D_batch, S_batch, Q_batch, L_batch = self.sess.run([D, S, Q, L], feed_dict={self.z: noise})
#            print('************************************ SQL ************************************')
#            print('D before G update:', D_batch)
#            print(np.min(D_batch), np.max(D_batch))
#            print('S before G update:', S_batch)
#            print('Q before G update:', Q_batch)
#            print('L before G update:', L_batch)
#            print(np.min(L_batch), np.max(L_batch))
#            print('Singular values:', np.linalg.svd(L_batch, compute_uv=False))
                
#            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% G_vars %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
#            for var, val in zip(gen_vars, self.sess.run(gen_vars)):
#                print('G_vars before update: '+var.name, val)
                
            noise = np.random.normal(scale=0.5, size=(batch_size, self.noise_dim))
            _, gl, dppl, my = self.sess.run([g_train, g_loss, dpp_loss, mean_y], feed_dict={self.z: noise})
            
#            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% G_grads %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
#            for var, val in zip(gen_vars, self.sess.run(g_grads, feed_dict={self.z: noise})):
#                print(var.name, val)
#            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DPP_grads %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
#            for var, val in zip(gen_vars, self.sess.run(dpp_grads, feed_dict={self.z: noise})):
#                print(var.name, val)
#            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% G_vars %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
#            for var, val in zip(gen_vars, self.sess.run(gen_vars)):
#                print('G_vars after update: '+var.name, val)
            
#            X_fake = self.sess.run(self.x_fake, feed_dict={self.z: noise})
#            print('************************************ X_fake ************************************')
#            print('After G update:', X_fake)
            
#            D_batch, S_batch, Q_batch, L_batch = self.sess.run([D, S, Q, L], feed_dict={self.z: noise})
#            print('************************************ SQL ************************************')
#            print('D after G update:', D_batch)
#            print(np.min(D_batch), np.max(D_batch))
#            print('S after G update:', S_batch)
#            print('Q after G update:', Q_batch)
#            print('L after G update:', L_batch)
#            print(np.min(L_batch), np.max(L_batch))
#            print('Singular values:', np.linalg.svd(L_batch, compute_uv=False))
            
            summary_str = self.sess.run(merged_summary_op, feed_dict={self.x: X_real, self.z: noise})
            summary_writer.add_summary(summary_str, t+1)
            
            # Show messages
            log_mesg = "%d: [D] real %f fake %f" % (t+1, dlr, dlf)
            log_mesg = "%s  [G] fake %f dpp %f y %f" % (log_mesg, gl, dppl, my)
            print(log_mesg)
            
            assert not (np.isnan(dlr) or np.isnan(dlf) or np.isnan(gl) or np.isnan(dppl))
            
            if save_interval>0 and (t+1)%save_interval==0 or t+1==train_steps:
                # Save the variables to disk.
                save_path = saver.save(self.sess, '{}/model'.format(save_dir))
                print('Model saved in path: %s' % save_path)
                print('Plotting results ...')
                gen_data = self.synthesize(1000)
                visualize_2d(data, func=func_obj.evaluate, gen_data=gen_data, save_path='{}/{}_synthesized.svg'.format(save_dir, t+1))
        
    
