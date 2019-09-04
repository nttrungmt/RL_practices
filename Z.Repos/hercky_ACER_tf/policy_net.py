import numpy as np
import tensorflow as tf
import sonnet as snt



class PolicyNet(snt.AbstractModule):
    """
    to model  f(.|\phi_{\theta'}(x_i))
    parameters denoted by \theta'
    
    Input: the observed state, x: 1 x INPUT_DIM
    
    Output: a distribution over action ( a normal distribution with fixed diagonal co-variance)
    
    """
    def __init__(self, 
                 hidden_size,
                 output_size,
                 co_var = 0.3,
                 name ="policy_net"):
        """
        hidden_size = size of the neural net
        output size = dimensionality of the action space
        """
        super(PolicyNet, self).__init__(name=name)
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._co_var = co_var #[co_var] * output_size
        self._name = name
        
    def _build(self, inputs):
        """Compute output Tensor from input Tensor."""        
        '''layer_1 = snt.Linear(self._hidden_size, name="layer_1")
        layer_mu = snt.Linear(self._output_size, name="layer_mu")
        layer_sigma = snt.Linear(self._output_size, name="layer_sigma")
        mlp_mu = snt.Sequential([layer_1,  tf.nn.relu , layer_mu])
        mlp_sigma = snt.Sequential([layer_1,  tf.nn.relu , layer_sigma])
        
        mu = mlp_mu(inputs)
        sigma = tf.add(tf.nn.softplus(mlp_sigma(inputs)), 1e-5)'''
        with tf.variable_scope(self._name):
            init_xavier = tf.contrib.layers.xavier_initializer()
            self.hidden1 = tf.layers.dense(inputs, self._hidden_size, activation=tf.nn.relu, kernel_initializer=init_xavier, name="hidden1")
            self.hidden2 = tf.layers.dense(inputs, self._hidden_size, activation=tf.nn.relu, kernel_initializer=init_xavier, name="hidden2")
            self.mu = tf.layers.dense(self.hidden1, self._output_size, activation=None, kernel_initializer=init_xavier, name="mu")
            self.sigma = tf.layers.dense(self.hidden2, self._output_size, activation=None, kernel_initializer=init_xavier, name="sigma")
            print('PolicyNet::_build: {}, {}'.format(self.mu, self.sigma))
            #sigma = tf.nn.softplus(sigma) + 1e-5
            self.pdparam = tf.concat([self.mu, self.mu * 0.0 + self.sigma], axis=1)
            self.dist = tf.contrib.distributions.MultivariateNormalDiag(self.mu, self.sigma)
        
        #dist = tf.contrib.distributions.Normal(loc=mu, scale = tf.ones_like(mu) * self._co_var)
        #dist = tf.contrib.distributions.MultivariateNormalDiag(loc=mu, scale_diag = tf.ones_like(mu) * self._co_var)
        #dist = tf.contrib.distributions.Normal(loc=mu, scale = sigma)
        #dist = tf.contrib.distributions.MultivariateNormalDiag(loc=mu, scale_diag = sigma)
        
        # mu : MB x ACTION_DIM
        # dist: MB x ACTION_DIM
        return self.pdparam, self.dist
    
    def update_local_params_op(self, target_name):
        """
        to copy the ops within a net
        
        returns the copy op
        """
        t_vars = tf.trainable_variables()
        
        local_vars = [var for var in t_vars if self._name in var.name]
        target_vars = [var for var in t_vars if target_name in var.name]
        
        copy_op = []
        
        for l_var,t_var in zip(local_vars,target_vars):
            copy_op.append(l_var.assign(t_var))

        return copy_op
    
    def local_params(self):
        """
        return the local params
        """
        t_vars = tf.trainable_variables()
        
        local_vars = [var for var in t_vars if self._name in var.name]
        
        return local_vars
    
    def soft_update_from_target_params(self, from_target_name, tau):
        """
        returns the op for soft updating the local params from target params 
        # target should be global
        # this call should be only of average net
        """
        t_vars = tf.trainable_variables()
        
        local_vars = [var for var in t_vars if self._name in var.name]
        target_vars = [var for var in t_vars if from_target_name in var.name]
        
        soft_update_op = []
        
        for l_var,t_var in zip(local_vars,target_vars):
            soft_update_op.append(l_var.assign(tf.multiply(l_var, tau) + tf.multiply(t_var, (1. - tau))))
            
        return soft_update_op
        


