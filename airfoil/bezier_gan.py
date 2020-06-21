"""
BezierGAN for capturing the airfoil manifold

Author(s): Wei Chen (wchen459@umd.edu)
"""

import numpy as np
import tensorflow as tf

#from shape_plot import plot_grid
from surrogate.surrogate_model import Model as SM


EPSILON = 1e-7

        
class BezierGAN(object):
    
    def __init__(self, latent_dim=5, noise_dim=100, n_points=64, bezier_degree=16, bounds=(0.0, 1.0), 
                 lambda0=1., lambda1=0.01):

        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        
        self.n_points = n_points
        self.X_shape = (n_points, 2, 1)
        self.bezier_degree = bezier_degree
        self.bounds = bounds
        
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        
    def generator(self, c, z, reuse=tf.AUTO_REUSE, training=True):
        
        depth_cpw = 32*8
        dim_cpw = int((self.bezier_degree+1)/8)
        kernel_size = (4,3)
#        noise_std = 0.01
        
        with tf.variable_scope('Generator', reuse=reuse):
                
            if self.noise_dim == 0:
                cz = c
            else:
                cz = tf.concat([c, z], axis=-1)
            
            cpw = tf.layers.dense(cz, 1024)
            cpw = tf.layers.batch_normalization(cpw, momentum=0.9)#, training=training)
            cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
    
            cpw = tf.layers.dense(cpw, dim_cpw*3*depth_cpw)
            cpw = tf.layers.batch_normalization(cpw, momentum=0.9)#, training=training)
            cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
            cpw = tf.reshape(cpw, (-1, dim_cpw, 3, depth_cpw))
    
            cpw = tf.layers.conv2d_transpose(cpw, int(depth_cpw/2), kernel_size, strides=(2,1), padding='same')
            cpw = tf.layers.batch_normalization(cpw, momentum=0.9)#, training=training)
            cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
#            cpw += tf.random_normal(shape=tf.shape(cpw), stddev=noise_std)
            
            cpw = tf.layers.conv2d_transpose(cpw, int(depth_cpw/4), kernel_size, strides=(2,1), padding='same')
            cpw = tf.layers.batch_normalization(cpw, momentum=0.9)#, training=training)
            cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
#            cpw += tf.random_normal(shape=tf.shape(cpw), stddev=noise_std)
            
            cpw = tf.layers.conv2d_transpose(cpw, int(depth_cpw/8), kernel_size, strides=(2,1), padding='same')
            cpw = tf.layers.batch_normalization(cpw, momentum=0.9)#, training=training)
            cpw = tf.nn.leaky_relu(cpw, alpha=0.2)
#            cpw += tf.random_normal(shape=tf.shape(cpw), stddev=noise_std)
            
            # Control points
            cp = tf.layers.conv2d(cpw, 1, (1,2), padding='valid') # batch_size x (bezier_degree+1) x 2 x 1
            cp = tf.nn.tanh(cp)
            cp = tf.squeeze(cp, axis=-1, name='control_point') # batch_size x (bezier_degree+1) x 2
            
            # Weights
            w = tf.layers.conv2d(cpw, 1, (1,3), padding='valid')
            w = tf.nn.sigmoid(w) # batch_size x (bezier_degree+1) x 1 x 1
            w = tf.squeeze(w, axis=-1, name='weight') # batch_size x (bezier_degree+1) x 1
            
            # Parameters at data points
            db = tf.layers.dense(cz, 1024)
            db = tf.layers.batch_normalization(db, momentum=0.9)#, training=training)
            db = tf.nn.leaky_relu(db, alpha=0.2)
            
            db = tf.layers.dense(db, 256)
            db = tf.layers.batch_normalization(db, momentum=0.9)#, training=training)
            db = tf.nn.leaky_relu(db, alpha=0.2)
            
            db = tf.layers.dense(db, self.X_shape[0]-1)
            db = tf.nn.softmax(db) # batch_size x (n_data_points-1)
            
#            db = tf.random_gamma([tf.shape(cz)[0], self.X_shape[0]-1], alpha=100, beta=100)
#            db = tf.nn.softmax(db) # batch_size x (n_data_points-1)
            
            ub = tf.pad(db, [[0,0],[1,0]], constant_values=0) # batch_size x n_data_points
            ub = tf.cumsum(ub, axis=1)
            ub = tf.minimum(ub, 1)
            ub = tf.expand_dims(ub, axis=-1) # 1 x n_data_points x 1
            
            # Bezier layer
            # Compute values of basis functions at data points
            num_control_points = self.bezier_degree + 1
            lbs = tf.tile(ub, [1, 1, num_control_points]) # batch_size x n_data_points x n_control_points
            pw1 = tf.range(0, num_control_points, dtype=tf.float32)
            pw1 = tf.reshape(pw1, [1, 1, -1]) # 1 x 1 x n_control_points
            pw2 = tf.reverse(pw1, axis=[-1])
            lbs = tf.add(tf.multiply(pw1, tf.log(lbs+EPSILON)), tf.multiply(pw2, tf.log(1-lbs+EPSILON))) # batch_size x n_data_points x n_control_points
            lc = tf.add(tf.lgamma(pw1+1), tf.lgamma(pw2+1))
            lc = tf.subtract(tf.lgamma(tf.cast(num_control_points, dtype=tf.float32)), lc) # 1 x 1 x n_control_points
            lbs = tf.add(lbs, lc) # batch_size x n_data_points x n_control_points
            bs = tf.exp(lbs)
            # Compute data points
            cp_w = tf.multiply(cp, w)
            dp = tf.matmul(bs, cp_w) # batch_size x n_data_points x 2
            bs_w = tf.matmul(bs, w) # batch_size x n_data_points x 1
            dp = tf.div(dp, bs_w) # batch_size x n_data_points x 2
            dp = tf.expand_dims(dp, axis=-1, name='fake_image') # batch_size x n_data_points x 2 x 1
            
            return dp, cp, w, ub, db
        
    def discriminator(self, x, reuse=tf.AUTO_REUSE, training=True):
        
        depth = 64
        dropout = 0.4
        kernel_size = (4,2)
        
        with tf.variable_scope('Discriminator', reuse=reuse):
        
            x = tf.layers.conv2d(x, depth*1, kernel_size, strides=(2,1), padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)#, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=training)
            
            x = tf.layers.conv2d(x, depth*2, kernel_size, strides=(2,1), padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)#, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=training)
            
            x = tf.layers.conv2d(x, depth*4, kernel_size, strides=(2,1), padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)#, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=training)
            
            x = tf.layers.conv2d(x, depth*8, kernel_size, strides=(2,1), padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)#, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=training)
            
            x = tf.layers.conv2d(x, depth*16, kernel_size, strides=(2,1), padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)#, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=training)
            
            x = tf.layers.conv2d(x, depth*32, kernel_size, strides=(2,1), padding='same')
            x = tf.layers.batch_normalization(x, momentum=0.9)#, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.layers.dropout(x, dropout, training=training)
            
            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, 1024)
            x = tf.layers.batch_normalization(x, momentum=0.9)#, training=training)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            
            d = tf.layers.dense(x, 1)
            
            q = tf.layers.dense(x, 128)
#            q = tf.layers.batch_normalization(q, momentum=0.9)#, training=training)
            q = tf.nn.leaky_relu(q, alpha=0.2)
            q_mean = tf.layers.dense(q, self.latent_dim)
            q_logstd = tf.layers.dense(q, self.latent_dim)
            q_logstd = tf.maximum(q_logstd, -16)
            # Reshape to batch_size x 1 x latent_dim
            q_mean = tf.reshape(q_mean, (-1, 1, self.latent_dim))
            q_logstd = tf.reshape(q_logstd, (-1, 1, self.latent_dim))
            q = tf.concat([q_mean, q_logstd], axis=1, name='predicted_latent') # batch_size x 2 x latent_dim
            
            return d, q
    
    def compute_diversity_loss(self, x, y):
            
        x = tf.layers.flatten(x)
        y = tf.squeeze(y)
        
        r = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
        D = r - 2*tf.matmul(x, tf.transpose(x)) + tf.transpose(r)
        S = tf.exp(-0.5*tf.square(D)) # similarity matrix (rbf)
#        S = 1/(1+D)
        
        if self.lambda0 == 'naive':
            
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
        
        return loss, D, S, Q, L
        
    def train(self, X_train, train_steps=2000, batch_size=32, disc_lr=2e-4, gen_lr=2e-4, save_interval=0, 
              directory='.', surrogate_dir='.'):
            
        X_train = np.expand_dims(X_train, axis=-1).astype(np.float32)
        
        # Inputs
        self.x = tf.placeholder(tf.float32, shape=(None,)+self.X_shape, name='real_image')
        self.c = tf.placeholder(tf.float32, shape=[None, self.latent_dim], name='latent_code')
        self.z = tf.placeholder(tf.float32, shape=[None, self.noise_dim], name='noise')
        
        # Targets
        q_target = tf.placeholder(tf.float32, shape=[None, self.latent_dim])
        
        # Outputs
        d_real, _ = self.discriminator(self.x)
        x_fake_train, cp_train, w_train, ub_train, db_train = self.generator(self.c, self.z)
        d_fake, q_fake_train = self.discriminator(x_fake_train)
        
        self.x_fake_test, self.cp, self.w, ub, db = self.generator(self.c, self.z, training=False)
        
        # Schedule for lambda1
        p = tf.cast(5, tf.float32)
        G_global_step = tf.Variable(0, name='G_global_step', trainable=False)
        lambda1 = self.lambda1 * tf.cast(G_global_step/(train_steps-1), tf.float32)**p
        disc_lr = tf.train.exponential_decay(disc_lr, G_global_step, 1000, 0.8, staircase=True)
        gen_lr = tf.train.exponential_decay(gen_lr, G_global_step, 1000, 0.8, staircase=True)
        
        # Combine with the surrogate model graph
        with tf.Session() as sess:
            surrogate_model = SM(sess, self.n_points)
            surrogate_graph = surrogate_model.restore(directory=surrogate_dir)
            output_node_names = 'net/y'
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                sess,  # The session is used to retrieve the weights
                surrogate_graph.as_graph_def(),  # The graph_def is used to retrieve the nodes
                output_node_names.split(",")  # The output node names are used to select the usefull nodes
            )
        graph = tf.get_default_graph()
        tf.graph_util.import_graph_def(frozen_graph, 
                                       input_map={'x:0': x_fake_train, 'training:0': False},
                                       name='surrogate')
        y = graph.get_tensor_by_name('surrogate/net/y:0')
        mean_y = tf.reduce_mean(y)
        
        # Losses
        # Cross entropy losses for D
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))
        # Cross entropy losses for G
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))
        dpp_loss, D, S, Q, L = self.compute_diversity_loss(x_fake_train, y*d_fake)
        g_dpp_loss = g_loss + lambda1 * dpp_loss
        # Regularization for w, cp, a, and b
        r_w_loss = tf.reduce_mean(w_train[:,1:-1], axis=[1,2])
        cp_dist = tf.norm(cp_train[:,1:]-cp_train[:,:-1], axis=-1)
        r_cp_loss = tf.reduce_mean(cp_dist, axis=-1)
        r_cp_loss1 = tf.reduce_max(cp_dist, axis=-1)
        ends = cp_train[:,0] - cp_train[:,-1]
        r_ends_loss = tf.norm(ends, axis=-1) + tf.maximum(0.0, -10*ends[:,1])
        r_db_loss = tf.reduce_mean(db_train*tf.log(db_train), axis=-1)
        r_loss = r_w_loss + r_cp_loss + 0*r_cp_loss1 + r_ends_loss + 0*r_db_loss
        r_loss = tf.reduce_mean(r_loss)
        # Gaussian loss for Q
        q_mean = q_fake_train[:, 0, :]
        q_logstd = q_fake_train[:, 1, :]
        epsilon = (q_target - q_mean) / (tf.exp(q_logstd) + EPSILON)
        q_loss = q_logstd + 0.5 * tf.square(epsilon)
        q_loss = tf.reduce_mean(q_loss)
        
        # Optimizers
        d_optimizer = tf.train.AdamOptimizer(learning_rate=disc_lr, beta1=0.5)
        g_optimizer = tf.train.AdamOptimizer(learning_rate=gen_lr, beta1=0.5)
        
        # Generator variables
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        # Discriminator variables
        dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
        
        # Training operations
        d_train_real = d_optimizer.minimize(d_loss_real, var_list=dis_vars)
        d_train_fake = d_optimizer.minimize(d_loss_fake + q_loss, var_list=dis_vars)
        g_train = g_optimizer.minimize(g_dpp_loss + 10*r_loss + q_loss, var_list=gen_vars, global_step=G_global_step)
        
#        for v in tf.trainable_variables():
#            print(v.name)
#        for v in dis_vars:
#            print(v.name)
#        for v in gen_vars:
#            print(v.name)
        
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        
#        # Create summaries to monitor losses
#        tf.summary.scalar('D_loss_for_real', d_loss_real)
#        tf.summary.scalar('D_loss_for_fake', d_loss_fake)
#        tf.summary.scalar('G_loss', g_loss)
#        tf.summary.scalar('DPP_loss', dpp_loss)
#        tf.summary.scalar('R_loss', r_loss)
#        tf.summary.scalar('Q_loss', q_loss)
#        # Merge all summaries into a single op
#        merged_summary_op = tf.summary.merge_all()
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        # Start training
        self.sess = tf.Session()
        
        # Run the initializer
        self.sess.run(init)
#        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter('{}/logs'.format(directory), graph=self.sess.graph)
    
        for t in range(train_steps):
    
            ind = np.random.choice(X_train.shape[0], size=batch_size, replace=False)
            X_real = X_train[ind]
            
#            y_batch = self.sess.run(y, feed_dict={x_fake_train: X_real})
#            print('************************************ y before update ************************************')
#            print(y_batch)
            
            _, dlr = self.sess.run([d_train_real, d_loss_real], feed_dict={self.x: X_real})
            latent = np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=(batch_size, self.latent_dim))
            noise = np.random.normal(scale=0.5, size=(batch_size, self.noise_dim))
            X_fake = self.sess.run(self.x_fake_test, feed_dict={self.c: latent, self.z: noise})
            
            if np.any(np.isnan(X_fake)):
                ind = np.any(np.isnan(X_fake), axis=(1,2,3))
                print(self.sess.run(ub, feed_dict={self.c: latent, self.z: noise})[ind])
                assert not np.any(np.isnan(X_fake))
                
            _, dlf, qdl, lrd = self.sess.run([d_train_fake, d_loss_fake, q_loss, disc_lr],
                                             feed_dict={x_fake_train: X_fake, q_target: latent})
                
            latent = np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=(batch_size, self.latent_dim))
            noise = np.random.normal(scale=0.5, size=(batch_size, self.noise_dim))
            
            _, gl, dppl, rl, qgl, my, lbd1, lrg = self.sess.run([g_train, g_loss, dpp_loss, r_loss, q_loss, mean_y, lambda1, gen_lr],
                                                                feed_dict={self.c: latent, self.z: noise, q_target: latent})
            
#            y_batch = self.sess.run(y, feed_dict={x_fake_train: X_real})
#            print('************************************ y after update ************************************')
#            print(y_batch)
            
#            D_batch, S_batch, Q_batch, L_batch = self.sess.run([D, S, Q, L], feed_dict={self.c: latent, self.z: noise})
#            print('************************************ SQL ************************************')
#            print('L-S:', L_batch-S_batch)
#            print(Q_batch)
            
#            summary_str = self.sess.run(merged_summary_op, feed_dict={self.x: X_real, x_fake_train: X_fake,
#                                                                      self.c: latent, self.z: noise, q_target: latent})
#            
#            summary_writer.add_summary(summary_str, t+1)
            
            # Show messages
            log_mesg = "%d: [D] real %f fake %f q %f lr %f" % (t+1, dlr, dlf, qdl, lrd)
            log_mesg = "%s  [G] fake %f dpp %f reg %f q %f y %f lambda1 %f lr %f" % (log_mesg, gl, dppl, rl, qgl, my, lbd1, lrg)
            print(log_mesg)
            
            if save_interval>0 and (t+1)%save_interval==0 or t+1==train_steps:
                
                # Save the variables to disk.
                save_path = saver.save(self.sess, '{}/model'.format(directory))
                print('Model saved in path: %s' % save_path)
#                print('Plotting results ...')
#                plot_grid(5, gen_func=self.synthesize, d=self.latent_dim, bounds=self.bounds,
#                          scale=.95, scatter=True, s=1, alpha=.7, fname='{}/synthesized'.format(directory))
        
        summary_writer.close()
                    
    def restore(self, directory='.'):
        
        self.sess = tf.Session()
        # Load meta graph and restore weights
        saver = tf.train.import_meta_graph('{}/model.meta'.format(directory))
        saver.restore(self.sess, tf.train.latest_checkpoint('{}/'.format(directory)))
        
        # Access and create placeholders variables            
        graph = tf.get_default_graph()
        self.x = graph.get_tensor_by_name('real_image:0')
        self.c = graph.get_tensor_by_name('latent_code:0')
        self.z = graph.get_tensor_by_name('noise:0')
        self.x_fake_test = graph.get_tensor_by_name('Generator_1/fake_image:0')
        self.cp = graph.get_tensor_by_name('Generator_1/control_point:0')
        self.w = graph.get_tensor_by_name('Generator_1/weight:0')

    def synthesize(self, latent, noise=None):
        if isinstance(latent, int):
            N = latent
            latent = np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=(N, self.latent_dim))
            noise = np.random.normal(scale=0.5, size=(N, self.noise_dim))
            X, P, W = self.sess.run([self.x_fake_test, self.cp, self.w], feed_dict={self.c: latent, self.z: noise})
        else:
            N = latent.shape[0]
            if noise is None:
                noise = np.zeros((N, self.noise_dim))
            X, P, W = self.sess.run([self.x_fake_test, self.cp, self.w], feed_dict={self.c: latent, self.z: noise})
        return np.squeeze(X)