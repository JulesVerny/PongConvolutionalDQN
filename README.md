## Pong Game Reinforcement DQN Learning ##

Deep DQN Based Reinforcement Learning for simple Pong PyGame.  This python based RL Experiment plays a Py Pong Game (DQN control of Left Hand Paddle against a programmed RHS Paddle)

![alt text](https://github.com/JulesVerny/PongConvolutionalDQN/blob/master/ScreenImage.PNG "Game Play")

The Objective is simply measured as successfully returning of the Ball by the Left Paddle which is Trained and Controlled by a DQN Agent.  The programmed opponent player is a pretty hot player. So success as is simply the  ability to return ball served from Serena Williams. The Moving Average Score is calculated in the range from [-10, +10] from Complete failure to return the balls, to full success in returning the Ball. This experiment demonstrates DQN based Reinforcement Learning Agent, which improves from poor performance ~ -9.0 towards reasonably good  performance +9.9 in around 40,000 epochs.  

This is a Convolutional Network based RL implementation where it is based upon the Game Image state returned from the pyGame Game:
ScreenImage = pygame.surfarray.array3d(pygame.display.get_surface()) 
This makes Learning very slow at about 40,000 epochs to Train (About 5 hours on a Tensor Flow enbaled GPU) 
The Best Weights are then stored in BestPongModelWeights.h5, for use in Subsequent Agent Play 

This DQN code takes the 400x400 Screen image, and reduces it down to 40x40 greyscale image using skimage image processing, and stacks this up with previous 3 images into a 40x40x4 input into the Keras based Convolutional network.
The 'successful' network compromises of 3 convolutional layers and two dense layers to make an estimate of Q, for Three Actions (Stay, Up, Down)
![alt text](https://github.com/JulesVerny/PongConvolutionalDQN/blob/master/FinalPerfomance.png "Score growth")      

### Erratic Long Term Training ###
Note I capture and abort the DQN Training as soon as I see the Training Game performance approach and stay around +10.0 for the First time.  Regardless of any further Epsilon decay.  I have noticed that keeping the Training going, with further epsilon decay  will cause various erratic game declines and recovery growths. I cannot explain these erratic declines.  So its good to keep a watch on Training Performance and not waste days expecting the ultimate performance.
![alt text](https://github.com/JulesVerny/PongConvolutionalDQN/blob/master/Scoreat250000.png "Erratic Long Term Perfomance")

### Useage ###

* python TrainAgent.py   : To Train the Agent  up to the point where good perfomance is observed
* python PlayBestAgent.py  : To Play the Trained Agent (By loading the BestPongModelWeights.h5)
* python PlotProgress.py   : To check the Game Score Growth during the long hours of Training

The Experiment is based upon the following files:  
* MyPong.py   : The pygame based Pong Game based upon Siraj Raval's code
* MyAgent.py  : The Convolutional DQN based agent using Ben Laus Convolutional Flappy Bird DQN code as a source  

### Main Python Package Dependencies ###
pygame, keras [hence TensorFlow,Theano], numpy, matplotlib, skimage

### Acknowledgments: ###
* The  Pong Game Code is based upon Siraj Raval's inspiring videos on Machine learning and Reinforcement Learning 
https://github.com/llSourcell/pong_neural_network_live

* The DQN Agent Software is Based upon Ben Lau source code: 
https://github.com/yanpanlau/Keras-FlappyBird

* Daniel Slaters Blog & Examples:
http://www.danielslater.net/2016/03/deep-q-learning-pong-with-tensorflow.html?showComment=1502902115538

* WILDML Reinforcement Learning Summary (Examples):
http://www.wildml.com/2016/10/learning-reinforcement-learning/
