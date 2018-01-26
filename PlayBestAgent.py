#
# MyPong DQN Reinforcement Learning Experiment
# Play the Best Agent  as previously Trained
# 
#  requires pygame, numpy, matplotlib, keras [and hence Tensorflow Backend] 
# ==========================================================================================
import MyPong # My PyGame Pong Game 
import MyAgent # My DQN Based Agent
import numpy as np 
import random 
#
import pickle, os
import warnings
# Import some image processing
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
#
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
#
import matplotlib.pyplot as plt
# =======================================================================
#   DQN Algorith Paramaters 

TOTAL_GAMETIME = 5000 
#
#  Now the Processed Convolutional Image Array dimensions into the Agent  
IMGHEIGHT = 40
IMGWIDTH = 40
IMGHISTORY=4
SCORELENGTH = 3

# =======================================================================
# Process Reduce the 420x 400 Pong Game Screen Image
#
def ProcessGameImage(RawImage):
	GreyImage = skimage.color.rgb2gray(RawImage)
	# Get rid of bottom Score line 
	# Pygame seems to have turned the Image sideways so remove X direction
	CroppedImage = GreyImage[0:400,0:400]
	ReducedImage = skimage.transform.resize(CroppedImage,(IMGHEIGHT,IMGWIDTH), mode='reflect')
	ReducedImage = skimage.exposure.rescale_intensity(ReducedImage,out_range=(0,255))
	#Decide to Normalise
	ReducedImage = ReducedImage/128 # Normalise data to [0, 1] range
	
	return ReducedImage
# =====================================================================
# Main Experiment Method 
def PlayGame():
	GameTime = 0
    
	GameHistory = []
	
	#Create our PongGame instance
	TheGame = MyPong.PongGame()
    # Initialise Game
	TheGame.InitialDisplay()
	#
	#  Create our Agent (including DQN based Brain)
	TheAgent = MyAgent.Agent()
	
	# Now Now the Trained Model into the Agent
	TheAgent.LoadBestModel()
	
	# Initialise NextAction  Assume Action is scalar:  0:stay, 1:Up, 2:Down
	BestAction = 0
	
	# Get an Initial State
	[InitialScore,InitialScreenImage]= TheGame.PlayNextMove(BestAction)
	InitialGameImage = ProcessGameImage(InitialScreenImage);
	#
	# Now Initialise the Game State as the Stack of four x intial Images
	GameState = np.stack((InitialGameImage, InitialGameImage, InitialGameImage, InitialGameImage), axis=2)
	# Keras expects shape 1x40x40x4
	GameState = GameState.reshape(1, GameState.shape[0], GameState.shape[1], GameState.shape[2])
	
    # =================================================================
	#Main Experiment Loop 
	while (GameTime < TOTAL_GAMETIME):    
	
		# First just Update the Game Display
		if GameTime % 100 == 0:
			TheGame.UpdateGameDisplay(GameTime,TheAgent.epsilon)

		# Get the Best Action From the Agent
		BestAction = 0
		BestAction = TheAgent.ReturnBestAct(GameState)
		
		#  Now Apply the Recommended Action into the Game 	
		[ReturnScore,NewScreenImage]= TheGame.PlayNextMove(BestAction)
		
		# Need to process the returned Screen Image, 
		NewGameImage = ProcessGameImage(NewScreenImage);
		
		# Now reshape Keras expects shape 1x40x40x1
		NewGameImage = NewGameImage.reshape(1, NewGameImage.shape[0], NewGameImage.shape[1], 1)
		
		#Now Add the new Image into the Next GameState stack, using 3 previous capture game images 
		NextState = np.append(NewGameImage, GameState[:, :, :, :3], axis=3)
		
		# Move State On
		GameState = NextState
		
		# Move GameTime Click
		GameTime = GameTime+1

        #Save the model every 5000
		if GameTime % 5000 == 0:
            # Save the Keras Model
			TheAgent.SaveWeights()

		if GameTime % 25 == 0:
			print("Game Time: ", GameTime,"  Game Score: ", "{0:.2f}".format(TheGame.GScore), "   EPSILON: ", "{0:.4f}".format(TheAgent.epsilon))
			GameHistory.append((GameTime,TheGame.GScore,TheAgent.epsilon))
			
			#  Now write the Play progress to File
			GFile = open('PlayHistory.dat','wb')
			pickle.dump(GameHistory,GFile)
			GFile.close()
				
	# ===============================================
	
	#  Game Completed So Display the Final Scores Grapth
	GFile = open('PlayHistory.dat','rb')
	PlayHistory = pickle.load(GFile)
	GFile.close()	

	# Plot the Score vs Game Time profile
	x_val = [x[0] for x in PlayHistory]
	y_val = [x[1] for x in PlayHistory]

	plt.plot(x_val,y_val)
	plt.xlabel("Play Time")
	plt.ylabel("Score")
	plt.show()
	
	# =======================================================================
def main():
    #
	# Main Method Just Play our Game
	PlayGame()

	# =======================================================================
if __name__ == "__main__":
    main()
