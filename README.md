# Deep_Reinforcement_Learning
Deep Q-Network (DQN) to play classic Atari Games

3 Atari games (MsPacman, Boxing and Pong) are being tested with the same architecture and achieved decent performance.
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")
<dl>
  <dt>Note:</dt>
  This is a smaller network with shorter training times than commonly used for accomodating the training with normal PC.
</dl>
<br><br/>
The key details of the architecture is as follow:

<dl>
  <dt>State Space:</dt>
  <ul>
  <li>Environment observation is converted to greyscale and reduced in size (60 x 60) to conserve memory.</li>
  <li>4 consecutive frames are stacked together (60 x 60 x 4) in order to capture the motion.</li>
  </ul>
  
   <dt>Agent:</dt>
  <ul>
  <li>Convolutional neural network (CNN) is used to approximate Q-function.</li>
  <li>input &rarr; conv (6 x 6 x 16) + stride 2 + RELU &rarr; conv (4 x 4 x 32) + stride 2 + RELU &rarr; flatten &rarr; hidden layer (256 units) + RELU &rarr; linear layer &rarr; state-action value function</li>
  </ul>

  <dt>Training:</dt>
  <ul>
  <li>1 million environmental steps is used as the duration of training (This can be increased for better performance).</li>
  <li>For more stable gradient update, several modifications are added as follow:</li>
  <ul>
  <li>Experience replay to store transitions.</li>
  <li>Separate stationary target network (updated every 5k steps).</li>
  <li>Rewards is clipped to be between -1 and 1.</li>
  </ul>
  </ul>
</dl>

<br><br/>

The required library:
* TensorFlow
* numpy
* matplotlib
* gym
* random
* skimage
