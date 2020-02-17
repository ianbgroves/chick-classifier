# CNNclassifier
CNN image classifier for chick embryo stage classification

This repository contains the code required to train a CNN to classify images of different chick embryo stages (stages 10.1, 10.2, 10.3). The training images are avaliable at this link https://drive.google.com/open?id=1Y6eitKyd1gypDd2w81cvq2vAnmqYH637. Please ignore the 10_1 folder in the repository as some of the images uploaded to github but not all.

The dataset for training the CNN is contained within the 'training_data' folder, which is then split into the 3 stages. These images have already been processed so they are of uniform size and greyscale. Each image is also repeated 24 times at 15° rotational intervals to give a larger training set. 
The 'images' folder contains the images without the additional rotations, as well as the edge cases which were not used for training. (Nb. it would be interesting to see what the trained model classifies the edge cases as if there is time). If you need the original unprocessed images let me know and we'll work out a way to get them to you.

'Code' contains the programs used to prepare the data and to train the CNN, as well as two .ipynb files which contain the results of the model training I already did.

The most important program is 'cnn2.py' which is the program used to train the CNN. It should be fairly straighforward to see how it works from the code comments but if you have any questions just shoot me an email. 'Code' also contains the programs used to grayscale and produced the rotated versions of every image ('grayscale.py' and 'rotate_save.py).

The two .ipynb files contain the results of the two times I ran trained the CNN, first time using all 3 stages, which resulted in an accuracy of around 65% (CNNtest1), and second time using stages 10.1 and 10.3, which resulted in an accuracy of 80% (CNNtest2). In both cases, the validation accuracy (accuracy of the CNN on the unlabeled test samples) gets better each epoch - see the outputs in CNNtest1 and CNNtest2 (One epoch is when an entire dataset is passed both forward and backward through the neural network once). These results suggest to me that the training is working, as it is more accurate when only differentiating between the two stages, and should therefore be investigated further.


NEXT STAGES:
1. I've not been able to store the trained model anywhere, so it will have to be run again before anything else can be done. All this requires is running cnn2.py once you've downloaded the training_data images. On my machine (only 4GB) this took around 5 hours.
2. Once the CNN is trained, look into how to get predictions on a single image. I may be able to make some time to look into this myself.
3. Work out how to store the trained model so it doesn't have to be re-trained everytime.
4. Change parameters for training the model to see if a better validation accuracy can be achieved. 65% is okay but I'm sure with some tweaking it can do better.
