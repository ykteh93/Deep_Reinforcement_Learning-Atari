# Deep_Reinforcement_Learning
Deep Q-Network (DQN) to play classic Atari Games

3 Atari games (MsPacman, Boxing and Pong) are being tested with the same architecture of Q-learning

<br><br/>
The details of the architecture is as follow:

<dl>
  <dt>State Space:</dt>
  <ul>
  <li>Environment observation is converted to greyscale and reduced in size (60 x 60) to conserve memory.</li>
  <li>4 consecutive frames are stacked together (60 x 60 x 4) in order to capture the motion.</li>
  </ul>
  
   <dt>Agent:</dt>
  <ul>
  <li>Convolutional neural network (CNN) is used to approximate Q-function.</li>
  <li>input &rarr; conv(6 x 6 x 16) + stride 2 + RELU &rarr; conv(4 x 4 x 32) + stride 2 + RELU &rarr; flatten &rarr; hidden layer (256 units) + RELU &rarr; linear layer &rarr; action probabilities</li>
  </ul>
</dl>
