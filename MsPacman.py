import gym
import numpy as np
import tensorflow as tf
import random
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque
from tensorflow.contrib.layers import convolution2d, fully_connected

env = gym.make("MsPacman-v3")		# Atari Games Selection
max_score = 0				# Initialized the best score as 0, so that 
n_steps = 1000000			# Train for 1 millions steps (can increase for more if have better GPU)
batch_size = 64				# Mini batch size used for optimization
discount_rate = 0.99			# Discount rate (control the value of the reward for ner future or distant future)
done = True
test_episode = 100			# Number of test episodes to run for evaluation because each episode is stostatic
replay_memory_size = 250000		# Size of the Experience Replay to store the previous episodes for more stable gradient update
replay_memory = deque()			# To store of the details of episodes in Experience Replay
input_height = 60			# Size of the reduced image height (to conserve memory, can increased for more for better performance)
input_width = 60			# Size of the reduced image width  (to conserve memory, can increased for more for better performance)
input_channels = 4  			# The number of frame to stack together in order to capture the motion
conv_n_maps = [16, 32]			# Size of the kernel/filter for CNN
conv_kernel_sizes = [(6,6), (4,4)]	# 
conv_strides = [2, 2]			# Number of strides for the kernel/filter to slide across image
conv_paddings = ["SAME"] * 2		# Padding choice 
conv_activation = [tf.nn.relu] * 2	# Activation for CNN (RELU is used here)
n_hidden_inputs = 32 * 15 * 15
n_hidden = 256 				# Number of hidden layer
hidden_activation = tf.nn.relu 		# Activation for hidden layer (RELU is also used here)
n_outputs = env.action_space.n  	# Number of possible output for the game (this is different for each game)

initializer = tf.random_normal_initializer(seed=seed, stddev=0.01)	# initializer for weight
b_initializer = tf.constant_initializer(0.1)						# initializer for bias
learning_rate = 0.0001												# learning rate for the optimizater 
Total_Clip_Return_Across_TimeStep = np.array([])
Total_Real_Return_Across_TimeStep = np.array([])
Total_Discounted_Return_Across_TimeStep = np.array([])
Total_Loss_Across_TimeStep = np.array([])
Total_Episode_Across_TimeStep = np.array([])

def q_network(X_state, scope):
	prev_layer = X_state
	with tf.variable_scope(scope) as scope:
		for n_maps, kernel_size, stride, padding, activation in zip(conv_n_maps, conv_kernel_sizes, conv_strides, conv_paddings, conv_activation):
			prev_layer = convolution2d(prev_layer, num_outputs=n_maps, kernel_size=kernel_size, stride=stride, padding=padding, activation_fn=activation, weights_initializer=initializer, biases_initializer=b_initializer)
		last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_inputs])
		hidden = fully_connected(last_conv_layer_flat, n_hidden, activation_fn=hidden_activation, weights_initializer=initializer, biases_initializer=b_initializer)
		outputs = fully_connected(hidden, n_outputs, activation_fn=None, weights_initializer=initializer, biases_initializer=b_initializer)
	trainable_vars = {var.name[len(scope.name):]: var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)}
	return outputs, trainable_vars

X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels])
target_Q_value, actor_vars = q_network(X_state, scope="target_q_networks")    
Q_value, critic_vars = q_network(X_state, scope="q_networks") 

copy_ops = [actor_var.assign(critic_vars[var_name]) for var_name, actor_var in actor_vars.items()]
copy_critic_to_actor = tf.group(*copy_ops)

with tf.variable_scope("train"):
	X_action = tf.placeholder(tf.int32, shape=[None])
	y = tf.placeholder(tf.float32, shape=[None])
	qvalue = tf.reduce_sum(tf.multiply(Q_value, tf.one_hot(X_action, n_outputs)),axis=1)
	cost = tf.reduce_mean(tf.square(y - qvalue))
	optimizer = tf.train.AdamOptimizer(learning_rate)
	training_op = optimizer.minimize(cost)


# Randomly sample the mini batch of previous episdoes from experience replay for training
def sample_memories(batch_size):
	indices = np.random.permutation(len(replay_memory))[:batch_size]
	cols = [[], [], [], [], []] # state, action, reward, next_state, continue
	for idx in indices:
		memory = replay_memory[idx]
		for col, value in zip(cols, memory):
			col.append(value)
	cols = [np.array(col) for col in cols]
	return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)


# The epsilon greedy here is with a fix epsilon. Ideally we would want a decay epsilon to ensure the system can reach optimal policy. But it is difficult pick the decay rate.
def epsilon_greedy(q_values):
	if float(np.random.rand(1)) < 0.1:
		return env.action_space.sample() # random action
	else:
		return np.argmax(q_values) # optimal action


# Preprocess the observation to convert the RGB into greyscale, remove some extra useless blank on the side, and resize. (Mainly to speed up computation and improve performance)
def preprocess_observation(obs):
	img = resize(rgb2gray(obs[0:195,:,:]), (input_height, input_width),mode='constant')
	img = np.reshape(img, [input_height,input_width,1])
	return img


def evaluation():
	total_clip_return = 0
	total_real_return = 0
	total_discounted_return = 0
	for e in range(test_episode):
		print("\rTest: {} ".format(e), end="")
		for t in range(input_channels):
			if t == 0:
				obs = env.reset()
				img = preprocess_observation(obs)
				evaluation_state = img
			else:
				obs, reward, _, _ = env.step(0)
				img = preprocess_observation(obs)
				evaluation_state = np.dstack((evaluation_state, img))

		j = 0
		temp_loss = np.array([])

		while True:
			selected_action = np.argmax(Q_value.eval(feed_dict={X_state: [evaluation_state]})[0])

			obs, reward, done, _ = env.step(selected_action)

			old_obs = evaluation_state
			next_state = preprocess_observation(obs)
			evaluation_state = np.delete(evaluation_state, [0], axis=2)
			evaluation_state = np.dstack((evaluation_state, next_state))

			total_real_return +=  reward
			reward = np.clip(reward, -1, 1)
			total_clip_return += reward
			total_discounted_return += (discount_rate ** (j)) * reward

			j += 1

			if done:
				break

	average_discounted_return = total_discounted_return/test_episode
	average_real_return = total_real_return/test_episode
	average_clip_return = total_clip_return/test_episode

	return average_discounted_return, average_real_return, average_clip_return

cummulative_loss = np.array([])
total_cummulative_loss = np.array([])
total_steps_cummulative_loss = np.array([])
run_evaluation = False
iteration = 0
one_time = True

init = tf.global_variables_initializer()
with tf.Session() as sess:
	init.run()
	saver = tf.train.Saver()

	# Copy from real network into stationary network to ensure both have the same initial weight and bias
	copy_critic_to_actor.run()

	while True:
		print("\rIteration {}".format(iteration), end="")
		iteration += 1

		# Break the loop when it hit maximum 1 million steps and run one last evaluation
		if iteration >= n_steps and done:
			average_discounted_return, average_real_return, average_clip_return = evaluation()

			print ('       Steps: %4d Mean Real Return: %10f Mean Clip Return: %10f Mean Discounted Return: %10f Training Loss : %f' %(iteration + 1, average_real_return, average_clip_return, average_discounted_return, np.mean(cummulative_loss)))
			Total_Clip_Return_Across_TimeStep = np.append(Total_Clip_Return_Across_TimeStep, average_clip_return)
			Total_Discounted_Return_Across_TimeStep = np.append(Total_Discounted_Return_Across_TimeStep, average_discounted_return)
			Total_Loss_Across_TimeStep = np.append(Total_Loss_Across_TimeStep, np.mean(cummulative_loss))
			Total_Episode_Across_TimeStep = np.append(Total_Episode_Across_TimeStep, iteration + 1)
			cummulative_loss = np.array([])

			# Store the final model
			saver.save(sess, './model_Final_PacMan/MsPacman')

			break

		# Copy over the network
		if (iteration + 1) % 5000 == 0:
			copy_critic_to_actor.run()

		# Run the evaluation every 50k steps
		if (iteration + 1) % 50000 == 0:
			run_evaluation = True

		# Run the evaluation one time at 1k steps
		if (iteration + 1) % 1000 == 0 and one_time:
			run_evaluation = True
			one_time = False

		if done: 
			# Run the evaluation every 50k steps
			if run_evaluation == True:

				average_discounted_return, average_real_return, average_clip_return = evaluation()

				print ('       Steps: %4d Mean Real Return: %10f Mean Clip Return: %10f Mean Discounted Return: %10f Training Loss : %f' %(iteration + 1, average_real_return, average_clip_return, average_discounted_return, np.mean(cummulative_loss)))
				Total_Clip_Return_Across_TimeStep = np.append(Total_Clip_Return_Across_TimeStep, average_clip_return)
				Total_Discounted_Return_Across_TimeStep = np.append(Total_Discounted_Return_Across_TimeStep, average_discounted_return)
				Total_Loss_Across_TimeStep = np.append(Total_Loss_Across_TimeStep, np.mean(cummulative_loss))
				Total_Episode_Across_TimeStep = np.append(Total_Episode_Across_TimeStep, iteration + 1)
				cummulative_loss = np.array([])

				# Store the Best Model 
				if average_clip_return > max_score:
					saver.save(sess, './model_PacMan/MsPacman')
					max_score = average_clip_return

				run_evaluation = False


			# Build up the initial 4 frames of observation
			for i in range(input_channels):
				# for the first frame
				if i == 0:
					obs = env.reset()
					img = preprocess_observation(obs)
					state = img
				# for the next 3 frames
				else:
					obs, _, _, _ = env.step(0)
					img = preprocess_observation(obs)
					state = np.dstack((state, img))	

		# Evaluate the Q Value to select action
		q_values = Q_value.eval(feed_dict={X_state: [state]})
		action = epsilon_greedy(q_values)

		# Step through the environment
		obs, reward, done, info = env.step(action)

		# Run the Preprocess of the observation into 60 x 60
		next_state = preprocess_observation(obs)

		# Store the previous observation for later calculation and update the 4 frames with the latest observation
		old_state = state
		state = np.delete(state, [0], axis=2)
		state = np.dstack((state, next_state))

		# clip the reward (for more stable gradient update) and append all the information into experience replay 
		clipped_reward = np.clip(reward, -1, 1)
		replay_memory.append((old_state, action, clipped_reward, state, 1.0 - done))


		# clear the experience replay by one to allow for new memory (This only start after experience replay has stored over 1 millions episodes)
		if len(replay_memory) > replay_memory_size:
			replay_memory.popleft()

		# Start Training once the experience replay has enough for one batch size
		if len(replay_memory) > batch_size:
			X_state_val, X_action_val, rewards, X_next_state_val, continues = sample_memories(batch_size)
			next_q_values = target_Q_value.eval(feed_dict={X_state: X_next_state_val})
			y_val = rewards + discount_rate * continues * np.max(next_q_values, axis=1, keepdims=True)
			y_val = np.reshape(y_val, (64))
			loss, _ = sess.run([cost, training_op], feed_dict={X_state: X_state_val, X_action: X_action_val, y: y_val})
			cummulative_loss = np.append(cummulative_loss, loss)

	# Plot for the graph
	plt.figure(1)
	plt.plot(Total_Episode_Across_TimeStep, Total_Clip_Return_Across_TimeStep)
	plt.title('Performance (Clip Return) ')
	plt.xlabel('Training Steps')
	plt.ylabel('Performance (Clip Return)')
	plt.savefig('B_Plot of Performance (Clip Return) Over 1 million Steps.png')

	plt.figure(2)
	plt.plot(Total_Episode_Across_TimeStep, Total_Discounted_Return_Across_TimeStep)
	plt.title('Performance (Mean Discounted Return)')
	plt.xlabel('Training Steps')
	plt.ylabel('Performance (Mean Discounted Return)')
	plt.savefig('B_Plot of Performance Over 1 million Steps.png')

	plt.figure(3)
	plt.plot(Total_Episode_Across_TimeStep, Total_Loss_Across_TimeStep)
	plt.title('Training Loss (Averagining over 50k steps)')
	plt.xlabel('Training Steps')
	plt.ylabel('Training Loss')
	plt.savefig('B_Plot of Loss Over 1 million Steps.png')
