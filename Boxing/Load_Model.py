#####################################################################################################################
#					Written by: Yih Kai Teh				     			    #
#														    #
#    This project is from one of my modules (Advanced Topic in Machine Learning) at UCL taught by Google DeepMind   #
#														    #
#				This file is only for evaluation, not for training				    #
#####################################################################################################################

# Due to the limited time when running this games, the input height and width are further reduced into 28

import gym
import numpy as np
import tensorflow as tf
import random

from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque
from tensorflow.contrib.layers import convolution2d, fully_connected

env = gym.make("Boxing-v3")				# atari games selection
n_steps = 1000000					# train for 1 millions steps (can increase for more if have better GPU)
batch_size = 64						# mini batch size used for optimization
discount_rate = 0.99					# discount rate (control the value of reward for near future or distant future)
test_episode = 100					# number of test episodes to run for evaluation as each episode is stostatic
input_height = input_width = 28				# size of the reduced image height and width 
input_channels = 4					# the number of frame to stack together in order to capture the motion
conv_kernel_output_channel = [16, 32]			# output channel of kernel/filter for CNN
conv_kernel_sizes = [(6,6), (4,4)]			# size of the kernel/filter for CNN
conv_strides = [2, 2]					# number of strides for the kernel/filter to slide across image
conv_paddings = ["SAME"] * 2				# padding choice 
conv_activation = [tf.nn.relu] * 2			# activation for CNN (RELU is used here)
n_hidden_inputs = input_height * input_width * 2	# size of the flatten layer
n_hidden = 256						# number of hidden layer
n_outputs = env.action_space.n				# number of possible output for the game (this is different for each game)

initializer = tf.random_normal_initializer(seed=600, stddev=0.01)	# initializer for weight
b_initializer = tf.constant_initializer(0.1)				# initializer for bias
learning_rate = 0.0001							# learning rate for the optimizater

# Q network architecture (conv(6x6x16) -> RELU -> conv(4x4x32) -> RELU -> flatten -> hidden layer (256 units) -> RELU -> output layer(number of actions))
def q_network(X_state, scope):
	prev_layer = X_state
	with tf.variable_scope(scope) as scope:
		for n_maps, kernel_size, stride, padding, activation in zip(conv_kernel_output_channel, conv_kernel_sizes, conv_strides, conv_paddings, conv_activation):
			prev_layer = convolution2d(prev_layer, num_outputs=n_maps, kernel_size=kernel_size, stride=stride, padding=padding, activation_fn=activation, weights_initializer=initializer, biases_initializer=b_initializer)
		last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_inputs])
		hidden = fully_connected(last_conv_layer_flat, n_hidden, activation_fn=tf.nn.relu, weights_initializer=initializer, biases_initializer=b_initializer)
		outputs = fully_connected(hidden, n_outputs, activation_fn=None, weights_initializer=initializer, biases_initializer=b_initializer)
	trainable_vars = {var.name[len(scope.name):]: var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)}
	return outputs, trainable_vars

X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels])	# placeholder for the state
target_Q_value, stationary_target_vars = q_network(X_state, scope="target_q_networks")		# the Q value from the stationary network
Q_value, current_target_vars = q_network(X_state, scope="q_networks")				# the Q value from the current network

# copy operation to update the statinary network by copying over from current network every 5k steps
copy = [stationary_target_var.assign(current_target_vars[var_name]) for var_name, stationary_target_var in stationary_target_vars.items()]
update_stationary_target = tf.group(*copy)

with tf.variable_scope("train"):
	X_action = tf.placeholder(tf.int32, shape=[None])					# placeholder for the action
	y = tf.placeholder(tf.float32, shape=[None])						# placeholder for the true Q value
	qvalue = tf.reduce_sum(tf.multiply(Q_value, tf.one_hot(X_action, n_outputs)),axis=1)	# calculate the Q value
	cost = tf.reduce_mean(tf.square(y - qvalue))						# define the cost function (differences between the true and estimated Q value)
	optimizer = tf.train.AdamOptimizer(learning_rate)					# optimizer used is ADAM, can be changed for other e.g. SGD
	training_op = optimizer.minimize(cost)							# minimize the cost function


# preprocess observation to convert RGB into greyscale, remove some extra useless blank on the side, and resize. (Mainly to speed up computation and improve performance)
def preprocess_observation(obs):
	img = resize(rgb2gray(obs[0:188,23:136,:]), (input_height, input_width),mode='constant')
	img = np.reshape(img, [input_height,input_width,1])
	return img


# Build up the initial 4 frames in each eipsodes
def initial_4_frames():
	for t in range(input_channels):
		if t == 0:	# for the first frame
			obs = env.reset()
			img = preprocess_observation(obs)
			state = img
		else:	# for the next 3 frames
			obs, _, _, _ = env.step(0)
			img = preprocess_observation(obs)
			state = np.dstack((state, img))
	return state


# run the evaluation at every 50k steps in order to monitor the performance during training 
def evaluation():
	# Reset all the return (real, discounted, clipped) to 0 for each time the evaluation is run
	real_return = clip_return = discounted_return = 0

	# run the evaluation for 100 episodes and take the average because each episodes is stochastics
	for e in range(test_episode):
		print("\rTest: {} ".format(e), end="")

		# Build up the initial 4 frames of observation
		evaluation_state = initial_4_frames()

		# reset the superscript of reward in each episodes 
		superscript = 0

		while True:
			env.render()
			# select action based on the maximum value of action-value function (Q function)
			selected_action = np.argmax(Q_value.eval(feed_dict={X_state: [evaluation_state]})[0])

			# Step through the environment with the selected action
			obs, reward, done, _ = env.step(selected_action)

			# Store the previous observation for later calculation and update the 4 frames with the latest observation
			old_obs = evaluation_state
			next_state = preprocess_observation(obs)
			evaluation_state = np.delete(evaluation_state, [0], axis=2)
			evaluation_state = np.dstack((evaluation_state, next_state))

			# Evaluate the real, clip and discounted rewards
			real_return +=  reward
			reward = np.clip(reward, -1, 1)
			clip_return += reward
			discounted_return += (discount_rate ** (superscript)) * reward

			# increase the superscript of the reward in each timestep 
			superscript += 1

			if done:
				break

	# calculate all the average rewards (real, clip and discounted) over 100 episodes as the indicators of the performance
	average_real_return = real_return/test_episode
	average_clip_return = clip_return/test_episode
	average_discounted_return = discounted_return/test_episode
	print ('Mean Real Return: %10f Mean Clip Return: %10f Mean Discounted Return: %10f' %(average_real_return, average_clip_return, average_discounted_return))

init = tf.global_variables_initializer()
with tf.Session() as sess:
	init.run()
	saver = tf.train.Saver()
	saver.restore(sess, "./model_Boxing/Boxing")
	evaluation()
