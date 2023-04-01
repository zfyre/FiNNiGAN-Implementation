# Frame interpolation using Convolutional NN and Generative Adversiral Network- Wasserstein Loss


* FinniGAN_1.0: https://colab.research.google.com/drive/1gltXtQ7LWMdwInzO29oedGHi3KhsLsCG#scrollTo=I2X5yppsmDpr
* WFinniGAN: https://colab.research.google.com/drive/1fdgDyCpNM2Zc2Vb9_Rw4udkw3GWyW4eF#scrollTo=--I4qBCGd9hd

* How to use:

  * Clone the FinniGan Repo using ‘git clone’
  * In the Directory ‘data’ create an empty directory ‘output’, all the videos to be processed and all the datapoints generated will be saved in the ‘Videos’ and ‘output’ directory respectively.
  * (Make sure to be in FinniGAN directory befire this) Run the ‘FrameExt.py’ script it uses OpenCV’s VidCapture method to extract the frame and save them in the output directory.
  * If training using the BCE loss,no further Changes to be made, else remove the ‘nn.Sigmoid’ activation from the Discriminator from ‘model.py’.
  * Finally run the ‘test.py’ to train the model.

* The Pre-Trained model can be used from the ‘logs’ directory, to test on your images:
  * Pass 2 consecutive frame tensors, stacked (each pixel value being the avg of two) upon each other having shape (1,3,256,256), the output you receive will be the middle generated frame.
  * You can also use the showImgTEST() method on a predefine dataset to test the results.
