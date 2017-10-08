## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time
import scipy.misc

from setup_inception import InceptionModel

from li_attack import CarliniLi
from colorama import Back, init

init()

VERBOSE=True
TARGETED=False

NUM_INCEPTION_CLASSES=1008


def load_images(input_dir, batch_shape=[2000,299,299,3]):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  
  filenames = []
  idx = 0
  filepaths=tf.gfile.Glob(os.path.join('./', '*.png'))
  print(len(filepaths))
  print(filepaths)
  batch_shape[0]=len(filepaths)
  batch_size = batch_shape[0]
  print(batch_shape)
  print("ZZZ")
  images = np.zeros(batch_shape, dtype=np.float32)
  
  for filepath in filepaths:
#    with tf.gfile.Open(filepath) as f:
#      image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
    image = np.array(scipy.misc.imresize(scipy.misc.imread(filepath),(299,299)),dtype=np.float32)/255
    
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image -0.5 #* 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      return filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    return filenames, images

def show(img):
    return 

'''

images betweeen [-0.5, 0.5]

inputs.shape (TARGETED)
(9, 28, 28, 1)

targets.shape (TARGETED)
(9, 10)

----------------

inputs.shape (TARGETED)
(1, 28, 28, 1)

targets.shape (TARGETED)
(1, 10)



'''
def generate_data(data, samples, targeted=True, start=0, inception=True):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    
    assert (targeted==True and start==0 and inception==True)
    
    
    inputs = []
    targets = []
    
    '''
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            for j in seq:
                if (j == np.argmax(data.test_labels[start+i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start+i])
            targets.append(data.test_labels[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)
    '''

    return inputs, targets


def get_labels(inputs, model, filenames=None):
  print (inputs.shape)
  labels = np.zeros((inputs.shape[0], NUM_INCEPTION_CLASSES), dtype=np.float32)
  for i in range(inputs.shape[0]):
    model_output=model.predict(inputs[i]).eval()
    print(model_output.shape)
    label=np.argmax(model_output)
    labels[i,label]=1
    if filenames is not None:
      print("model_output for {} is {}".format(filenames[i], model_output[i]))
      print("label for {} is {} ({})".format(filenames[i], label, model_output[i, label]))
      
  print (labels)
  print (labels.shape)
  return labels
  

if __name__ == "__main__":
    with tf.Session() as sess:
        data=['sample_image_00.png'] # true label for inception-V3: 133
        model=InceptionModel(sess)
        
        attack = CarliniLi(sess, model, targeted=TARGETED, abort_early=True)
        
        
        filenames, inputs = load_images('./')
        targets=get_labels(inputs, model, filenames=filenames)
        
        
        
        '''
        inputs, targets = generate_data(data, samples=1, targeted=False,
                                        start=0, inception=False)
        '''
        timestart = time.time()
        adv = attack.attack(inputs, targets)
        timeend = time.time()
        
        print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")

        for i in range(len(adv)):
            
            
            classification=model.predict(adv[i:i+1])
            classified_as=np.argmax(classification[0])
            print(Back.RED if classified_as==7 else Back.GREEN)
            
            if VERBOSE:
              print("Valid:")
              show(inputs[i])
              print("Adversarial:")
              show(adv[i])
              print("Classification:", classification)
            
            
            print( "classified as :", classified_as, Back.BLACK)

            print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)
            
            print("Input between [", np.min(inputs[i]), ", ", np.max(inputs[i]),"]")
            print("Inf. norm:", np.max(adv[i]-inputs[i]))
