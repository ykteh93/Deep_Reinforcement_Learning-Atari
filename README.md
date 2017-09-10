# Deep Reinforcement Learning
Deep Q-Network (DQN) to play classic Atari Games

3 Atari games (MsPacman, Boxing and Pong) are being tested with the same architecture and achieved decent performance.
<br>
<br>
<p align="center"> 
<img src="https://github.com/ykteh93/Deep_Reinforcement_Learning-Atari/blob/master/MsPacman/Graphs_and_Figure/For_README.png">
</p>

<br>
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

<dl>
  <dt>Note:</dt>
  1. This project is from one of my modules (Advanced Topics in Machine Learning) at UCL, taught by Google DeepMind.<br>
  2. So, it is a smaller network with shorter training times than commonly used for accomodating the training with normal PC.<br>
  3. Due to the requirement of the module, the codes are separated for each of the game but there is only minor differences.<br>
  4. The saved model for each games after training are included <i>(Run the Load_Model.py file for each games to evaluate)</i>.
  <ul style="list-style-type:circle">
  <li>This performance is still far from optimal because it is only trained with 1 million environmental steps.</li>
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
