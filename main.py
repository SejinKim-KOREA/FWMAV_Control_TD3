import tensorflow as tf
import numpy as np
import tflearn
import argparse

from copy import deepcopy
import random 
import math

# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 400)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 400)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(inputs, 400)
        t2 = tflearn.fully_connected(action, 400)
        net = tflearn.activation(
            tf.matmul(inputs, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        net = tflearn.fully_connected(net, 400)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 400)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

# ===========================
#   Agent Training
# ===========================

def train(sess, args, actor, critic, critic1):

    n_CFD = 1
    saver = tf.train.Saver()

    if os.path.isfile('./checkpoint') == 1:
        saver.restore(sess,"./Netsaver.ckpt")
    else:
        sess.run(tf.global_variables_initializer())
        # Initialize target network weights
        actor.update_target_network()
        critic.update_target_network()
        critic1.update_target_network()
    
    # State, reward 
    s = np.zeros((n_CFD+1, actor.s_dim))
    s2 = np.zeros((n_CFD+1, actor.s_dim))
    r = np.zeros(n_CFD+1)
    a = np.zeros((n_CFD+1, actor.a_dim))

    # Data reproduction 
    n_repro_x = 10
    n_repro_th = 10
    n_repro_mirror = 2

    # Initialize replay memory
    replay_buffer = []

    # Agent training starts !!!
    n_STROKE = np.zeros(n_CFD+1, dtype=int)
    SUCCESS_TIME = np.zeros(n_CFD+1)
 
    for kkk in range(restart+1,int(args['max_episode_len'])+1):
        # Wait state from CFD simulation

        # Read state
        s[CFD][0] = x_l
        s[CFD][1] = k
        s[CFD][actor.s_dim-11] = u_a
        s[CFD][actor.s_dim-10] = v_a
        s[CFD][actor.s_dim-9] = F_bx
        s[CFD][actor.s_dim-8] = F_by
        s[CFD][actor.s_dim-7] = x_r
        s[CFD][actor.s_dim-6] = y_r
        s[CFD][actor.s_dim-5] = u_b
        s[CFD][actor.s_dim-4] = v_b
        s[CFD][actor.s_dim-3] = math.sin(theta_b*math.pi)
        s[CFD][actor.s_dim-2] = math.cos(theta_b*math.pi)
        s[CFD][actor.s_dim-1] = omega

        # Choose action with added exploration noise
        a_noise = np.zeros(actor.a_dim)
        a[CFD] = actor.predict(np.reshape(s[CFD], (1, actor.s_dim)))[0]

        distance = (s[CFD][actor.s_dim-7]**2.0+s[CFD][actor.s_dim-6]**2.0)**0.5
        if distance > 0.01: 
            a_noise = np.random.normal(0.0, 0.1, [actor.a_dim])
            a[CFD] += a_noise
        
        if np.sum(n_STROKE, dtype=np.int32) <= 100: # warmup with big noise
            a_noise = np.random.normal(0.0, 1.0, [actor.a_dim])
            a[CFD] = a_noise                 

        a[CFD] = np.clip(a[CFD], -actor.action_bound[0], actor.action_bound[0])

        # Send action to CFD simulation
        if SUCCESS_TIME[CFD] < 1.0:
            # Send action

        # Learning while waitting state
        if np.sum(n_STROKE, dtype=np.int32) >= int(args['minibatch_size']):
            learning_count = 0
            for _ in range(1000):
                learning_count += 1
                batch = []
                batch = random.sample(replay_buffer, int(args['minibatch_size']))
                s_batch = np.array([_[0] for _ in batch])
                a_batch = np.array([_[1] for _ in batch])
                r_batch = np.array([_[2] for _ in batch])
                s2_batch = np.array([_[3] for _ in batch])                                            

                a2 = actor.predict_target(s2_batch)
                a2_noise = np.clip(np.random.normal(0.0, 0.2, [int(args['minibatch_size']),actor.a_dim]), -0.5, 0.5)
           
                a2 = np.clip(a2+a2_noise, -actor.action_bound[0], actor.action_bound[0])

                # Calculate targets
                target_q = critic.predict_target(s2_batch, a2)
                target_q1 = critic1.predict_target(s2_batch, a2)

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    y_i.append(r_batch[k] + critic.gamma * min(target_q[k],target_q1[k]))

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                predicted_q_value1, _ = critic1.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                if learning_count%2 == 0:
                    # Update the actor policy using the sampled gradient
                    a_outs = actor.predict(s_batch)
                    grads = critic.action_gradients(s_batch, a_outs)
                    actor.train(s_batch, grads[0])

                    # Update target networks
                    actor.update_target_network()
                    critic.update_target_network()
                    critic1.update_target_network()

        # Next state arrives from CFD simulation
        n_STROKE[CFD] += 1

        # Read next state
        s2[CFD][0] = x_l
        s2[CFD][1] = k
        s2[CFD][actor.s_dim-11] = u_a
        s2[CFD][actor.s_dim-10] = v_a
        s2[CFD][actor.s_dim-9] = F_bx
        s2[CFD][actor.s_dim-8] = F_by
        s2[CFD][actor.s_dim-7] = x_r
        s2[CFD][actor.s_dim-6] = y_r
        s2[CFD][actor.s_dim-5] = u_b
        s2[CFD][actor.s_dim-4] = v_b
        s2[CFD][actor.s_dim-3] = math.sin(theta_b*math.pi)
        s2[CFD][actor.s_dim-2] = math.cos(theta_b*math.pi)
        s2[CFD][actor.s_dim-1] = omega

        # Success check
        distance = (s2[CFD][actor.s_dim-7]**2.0+s2[CFD][actor.s_dim-6]**2.0)**0.5
        if distance <= 0.005:
            SUCCESS_TIME[CFD] += 1.0/(a[CFD][2]*200.0+100.0)/2.0
        else:
            SUCCESS_TIME[CFD] = 0.0

        # Save data to buffer with data reproduction method
        xl_origin1 = s[CFD][0]
        k_origin1 = s[CFD][1]
        uatn_origin1 = s[CFD][actor.s_dim-11]
        vatn_origin1 = s[CFD][actor.s_dim-10]
        fx_body_origin1 = s[CFD][actor.s_dim-9]
        fy_body_origin1 = s[CFD][actor.s_dim-8]
        x_origin1 = s[CFD][actor.s_dim-7]
        y_origin1 = s[CFD][actor.s_dim-6]
        u_origin1 = s[CFD][actor.s_dim-5]
        v_origin1 = s[CFD][actor.s_dim-4]
        theta_origin1 = math.acos(s[CFD][actor.s_dim-2])/math.pi
        if math.asin(s[CFD][actor.s_dim-3]) < 0.0:
            theta_origin1 *= -1.0
        omega_origin1 = s[CFD][actor.s_dim-1]

        a_xl_origin = a[CFD][0]
        a_alphal_origin = a[CFD][1]
        f_origin = a[CFD][2]

        xl_origin2 = s2[CFD][0]
        k_origin2 = s2[CFD][1]
        uatn_origin2 = s2[CFD][actor.s_dim-11]
        vatn_origin2 = s2[CFD][actor.s_dim-10]
        fx_body_origin2 = s2[CFD][actor.s_dim-9]
        fy_body_origin2 = s2[CFD][actor.s_dim-8]
        x_origin2 = s2[CFD][actor.s_dim-7]
        y_origin2 = s2[CFD][actor.s_dim-6]
        u_origin2 = s2[CFD][actor.s_dim-5]
        v_origin2 = s2[CFD][actor.s_dim-4]
        theta_origin2 = math.acos(s2[CFD][actor.s_dim-2])/math.pi
        if math.asin(s2[CFD][actor.s_dim-3]) < 0.0:
            theta_origin2 *= -1.0
        omega_origin2 = s2[CFD][actor.s_dim-1]

        x0 = -0.5 # X Starting point
        y0 = 0.5  # Y Starting point

        for i in range(1,n_repro_x+1):
            for j in range(1,n_repro_x+1):
                del_x = (-1.0+(2.0*float(i)-1.0)/float(n_repro_x)) - x0
                del_y = (-1.0+(2.0*float(j)-1.0)/float(n_repro_x)) - y0

                for k in range(1,n_repro_th+1):
                    rot_th = 2.0*math.pi*float(k-1)/float(n_repro_th)

                    for l in range(1,n_repro_mirror+1):
                        y_turn = 1.0
                        if n_repro_mirror == 2 and l == 2:
                            y_turn = -1.0

                        # state
                        s[0][0] = xl_origin1*y_turn
                        s[0][1] = k_origin1*y_turn
                        s[0][actor.s_dim-11] = (math.cos(rot_th)*uatn_origin1-math.sin(rot_th)*vatn_origin1)*y_turn 
                        s[0][actor.s_dim-10] = (math.sin(rot_th)*uatn_origin1+math.cos(rot_th)*vatn_origin1) 
                        s[0][actor.s_dim-9] = (math.cos(rot_th)*fx_body_origin1-math.sin(rot_th)*fy_body_origin1)*y_turn
                        s[0][actor.s_dim-8] = (math.sin(rot_th)*fx_body_origin1+math.cos(rot_th)*fy_body_origin1)
                        s[0][actor.s_dim-7] = (math.cos(rot_th)*(x_origin1-x0)-math.sin(rot_th)*(y_origin1-y0))*y_turn+x0+del_x
                        s[0][actor.s_dim-6] = (math.sin(rot_th)*(x_origin1-x0)+math.cos(rot_th)*(y_origin1-y0))+y0+del_y
                        s[0][actor.s_dim-5] = (math.cos(rot_th)*u_origin1-math.sin(rot_th)*v_origin1)*y_turn 
                        s[0][actor.s_dim-4] = (math.sin(rot_th)*u_origin1+math.cos(rot_th)*v_origin1) 
                        s[0][actor.s_dim-3] = math.sin((theta_origin1*math.pi+rot_th)*y_turn)
                        s[0][actor.s_dim-2] = math.cos((theta_origin1*math.pi+rot_th)*y_turn)
                        s[0][actor.s_dim-1] = omega_origin1*y_turn 

                        # action
                        a[0][0] = a_xl_origin
                        a[0][1] = a_alphal_origin
                        a[0][2] = f_origin

                        # next state
                        s2[0][0] = xl_origin2*y_turn
                        s2[0][1] = k_origin2*y_turn
                        s2[0][actor.s_dim-11] = (math.cos(rot_th)*uatn_origin2-math.sin(rot_th)*vatn_origin2)*y_turn 
                        s2[0][actor.s_dim-10] = (math.sin(rot_th)*uatn_origin2+math.cos(rot_th)*vatn_origin2) 
                        s2[0][actor.s_dim-9] = (math.cos(rot_th)*fx_body_origin2-math.sin(rot_th)*fy_body_origin2)*y_turn
                        s2[0][actor.s_dim-8] = (math.sin(rot_th)*fx_body_origin2+math.cos(rot_th)*fy_body_origin2)
                        s2[0][actor.s_dim-7] = (math.cos(rot_th)*(x_origin2-x0)-math.sin(rot_th)*(y_origin2-y0))*y_turn+x0+del_x
                        s2[0][actor.s_dim-6] = (math.sin(rot_th)*(x_origin2-x0)+math.cos(rot_th)*(y_origin2-y0))+y0+del_y
                        s2[0][actor.s_dim-5] = (math.cos(rot_th)*u_origin2-math.sin(rot_th)*v_origin2)*y_turn 
                        s2[0][actor.s_dim-4] = (math.sin(rot_th)*u_origin2+math.cos(rot_th)*v_origin2) 
                        s2[0][actor.s_dim-3] = math.sin((theta_origin2*math.pi+rot_th)*y_turn)
                        s2[0][actor.s_dim-2] = math.cos((theta_origin2*math.pi+rot_th)*y_turn)
                        s2[0][actor.s_dim-1] = omega_origin2*y_turn 

                        r[0] = - (s2[0][actor.s_dim-7]**2.0+s2[0][actor.s_dim-6]**2.0)**0.5 - s2[0][actor.s_dim-1]**2.0 

                        replay_buffer.append((deepcopy(np.reshape(s[0],(actor.s_dim,))), deepcopy(np.reshape(a[0],(actor.a_dim,))),
                                              r[0], deepcopy(np.reshape(s2[0],(actor.s_dim,)))))

    # save Net
    save_path = saver.save(sess,"./Netsaver.ckpt")

def main(args):

    with tf.Session() as sess:

        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))

        state_dim = int(13)
        action_dim = int(3)
        action_bound = np.zeros(action_dim)
        for i in range(action_dim):
            action_bound[i] = 1.0

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())

        critic1 = CriticNetwork(sess, state_dim, action_dim,
                                float(args['critic_lr']), float(args['tau']),
                                float(args['gamma']),
                                actor.get_num_trainable_vars()+critic.get_num_trainable_vars())
        
        train(sess, args, actor, critic, critic1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.0001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.0005)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=100)

    # run parameters
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=999999999)

    args = vars(parser.parse_args())

    main(args)
