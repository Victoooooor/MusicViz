# 3D Music Visualization

This project is an attempt to explore the various methods for **music feature extraction and music visualization**. The project includes 2 components, a string cluster to visualize music with 3D motion, and a generated styled video as an attempt to express mood of the music.

 



# Files

	./ckpt/					#Trained Models for Valence & Arousal
	./musicviz
		Cqt.py					#Constant Q-Transform (deprecated)
		Emo_CNN.py				#model definition for CNN used to extract Valence/Arousal
		feature.py				#Music Feature extraction
		music_cluster.py		#clustering for music features(deprecated)
		style_transfer.py		#keras layers using trained model for style transform with controllable intensity
		visual_gan.py			#biggan wrapper to generate video w/ music features
		visualize.py			#wrapper for all components using multiprocessing for stack reset
	./data 					#Data used for project
	./Demo.ipynb			#Colab notebook for generating videos
	./play.py				#Control motor through Arduino and play video w/ per frame synchronization
	./main.py				#local version of Demo
	./train.py				#training wrapper for music mood CNN

## Installation and Usage

	pip install git+https://github.com/Victoooooor/MusicViz.git
	from musicviz.visualize import visualize

## Algorithm

![Algorithm](https://github.com/Victoooooor/MusicViz/blob/main/data/image/diagram.png?raw=true)

## Generated Output


[![Youtube Vid](https://www.google.com/imgres?imgurl=https%3A%2F%2Fwww.openaccessgovernment.org%2Fwp-content%2Fuploads%2F2020%2F02%2Fyella-2.jpg&imgrefurl=https%3A%2F%2Fwww.openaccessgovernment.org%2Fbumble-bees-need-biodiversity%2F81996%2F&tbnid=uBK5Hhtgdr1t0M&vet=12ahUKEwiGmpmhuLz2AhVzqnIEHWRVCkkQMygBegUIARDgAQ..i&docid=rR3VGMoI1fFm_M&w=2000&h=1333&q=bumblebee&ved=2ahUKEwiGmpmhuLz2AhVzqnIEHWRVCkkQMygBegUIARDgAQ)](https://youtu.be/C5sjTF1ENas)

