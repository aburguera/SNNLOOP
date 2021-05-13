# Siamese Neural Network for Visual Loop Detection

The repository contains some convenience objects and examples to build, train and evaluate a Siamese Neural Network aimed at quantifying the overlap between two input images. Thus, its main goal is to perform visual loop detection.

Please take into account that:
* The system has been designed to work with underwater images grabbed by a bottom-looking camera.
* The code is designed to work with UCAMGEN datasets. Please check UCAMGEN repository for more information.
* The Siamese branches are the encoder part of an autoencoder. For more information about the autoencoders we have used, please check the AUTOENCODER repository.
* The code is based on Keras + Tensorflow.

If you use this software, please cite the following paper:

Reference to be posted soon. Please contact us.

## Credits

* Antoni Burguera (antoni dot burguera at uib dot es)

## Understanding the system

The main class of our system is **LoopModel**, which implements the Siamese Loop Detector. It has methods to create, train, evaluate, save and load the model, as well as to print and plot some basic stats.

To help in using the LoopModels, the following classes are provided:

* **AutoModel** : Implements the Convolutional Autoencoder whose Encoder constitutes the feature extractor. Please check the AUTOENCODER repository for more information.
* **LoopGenerator** : A Data Generator inheriting from Sequence, ready to be used to train and evaluate the proposed Siamese Neural Network. The LoopGenerator cannot directly access the UCAMGEN datasets. To interface an UCAMGEN dataset and the LoopGenerator, the LoopReader is provided.
* **LoopReader** : Pre-processes an UCAMGEM dataset to the desired format. This means labelling pairs of images according the how much do they overlap, balancing classes, using a specific output format, etcetera.

There are some auxiliary classes/modules/scripts:

* **loopmodeltest** : Fully commented usage example of the LoopModel class.
* **loopreadertest** : Fully commented usage example of the LoopReader class.
* **loopgeneratortest** : Fully commented usage example of the LoopGenerator class.
* **utils** and **looptools** : They were suposed to contain general purpose tools and application specific tools respectively, though they actually are an unsorted mix of tools used all around the repository. Some of the provided functions need serious refactoring.
* **preparereaders** : Pre-computes and saves LoopReaders for different configurations.
* **prepareloopmodels** : Pre-trains, performs basic evaluation and saves different LoopModel configurations.
* **preparestats** : Computes and saves different stats for different LoopModel configurations.

Please note that the **prepare\*** files are meant for internal (i.e. myself) usage. They are not optimal, they need refactoring, they depend on each other through saved files, ... In an ideal world, you should NOT use them. In an ideal world you should NOT even look at them. In a real world, however, you can be tempted to look or use them. If you do so and the Universe collapses or you lose your sanity, do NOT blame me. You have been warned.

## Using the system

Creating, training and evaluating a Siamese Loop Detector involves the following steps:

* Prepare the train and test **LoopReaders**.
* Use the train LoopReader to create the train and the validation **LoopGenerators**.
* Use the test LoopReader to create the test **LoopGenerator**.
* **Instantiate** a LoopModel: theModel=LoopModel()
* **Create** the LoopModel with the desired parameters: theModel.create(...)
* **Train** the model: theModel.fit(x=trainGenerator, validation_data=valGenerator, ...)
* **Evaluate** the model: theModel.evaluate(testGenerator, ...)
* **Save** the model: theModel.save(fileName)

Then, you can print some stats:

* Model **summary**: theModel.summary()
* Plot **architecture**: theModel.plot()
* Plot **training history** (loss, metrics, ...): theModel.plot_training_history()
* Print **evaluation results**: theModel.print_evaluation()

All these steps are exemplified in **loopmodeltest.py**. You are advised to check this particular file and refer to **loopreadertest.py** and **loopgeneratortest.py** for more specific examples.

To execute these \*test.py files be sure to:
* Have properly created and saved UCAMGEN datasets.
* Change the paths to datasets and storage folders at your convenience.

Please note that if you do not have GPU support you should change set_gpu(True) to set_gpu(False) at the beginning of example.py and main.py. **Actually, if you do not have GPU suppport it is advisable to remove the call to set_gpu**. Even if you have GPU, this may not work. Read the **Troubleshooting** section.

## Requirements

To execute this software, you will need:

* Python 3
* Keras
* Tensorflow
* NumPy
* Matplotlib
* Pickle
* SciKit-Image

## Troubleshooting

The GPU activation/deactivation function (set_gpu) works on my computer (Ubuntu 20 and CUDA Toolkit 10.1) but may not work on other computers, even if they use Ubuntu 20 and CUDA 10.1. Installing CUDA on Ubuntu 20 and making it useable by Keras+Tensorflow is (at least in April 2021) a true nightmare. Really. I don't even know why it works today and didn't work yesterday. So, if it doesn't work do not blame me. **Just remove the set_gpu call and use your own or execute the code on CPU.**

## Disclaimer

The code is provided as it is. It may work in your computer, it may not work. It may even crash it or, eventually, create a time paradox, the result of which could cause a chain reaction that would unravel the very fabric of the space-time continuum and destroy the entire universe. Some users have reported inter-dimensional portals being created and letting demon-like creatures enter our reality after executing this code because it is not dead which can eternal lie, and with strange aeons even death may die. Some other users reported Rick Sánchez entering our reality after downloading this repo. Just be careful and try to understand everything before using it. If you have questions, please carefully read the code and the paper. If this doesn't help, contact us.