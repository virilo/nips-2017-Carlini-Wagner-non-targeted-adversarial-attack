# -*- coding: utf-8 -*-
## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf

def inspecciona(hito=""):
  try:
    var=[v for v in tf.trainable_variables() if v.name == "InceptionV3/Conv2d_1a_3x3/weights:0"][0]
#    z=str(var.eval(sess))
    z=str(sess.run(var))
    print(hito + " ---------1234qwerty")
    print(z[:21])
  except Exception as ex:
    print(hito + " " + type(ex).__name__)
    


from tensorflow.contrib.slim.nets import inception
from tensorflow.core.framework.graph_pb2 import GraphDef
from google.protobuf import text_format
import numpy as np
from PIL import Image
import os
import time
import scipy.misc

#from setup_inception import InceptionModel

from li_attack import CarliniLi
from colorama import Back, init

init()

VERBOSE=True
TARGETED=False


NUM_INCEPTION_CLASSES=1001





INCEPTION_CKPT_FILE='inception_v3.ckpt'

# Get the current directory
dir_path = os.path.dirname(os.path.realpath(__file__))
checkpoint_dir=dir_path + "/"
checkpoint_path=checkpoint_dir + INCEPTION_CKPT_FILE 
checkpoint_path_meta=checkpoint_path+ '.meta'




slim = tf.contrib.slim

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
                seq = random.sample(range(1,1000), 10)
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

def get_labels_KKKKKK(inputs, model, filenames=None):
  print (inputs.shape)
  labels = np.zeros((inputs.shape[0], NUM_INCEPTION_CLASSES), dtype=np.float32)
  for i in range(inputs.shape[0]):
    labels[i,1]=1
      
  print (labels)
  print (labels.shape)
  return labels





def get_batch_shape(input_dir,batch_size, image_height, image_width, num_channels):
  input_files=tf.gfile.Glob(os.path.join(input_dir, '*.png'))
  batch_shape = [np.min([len(input_files),batch_size]), image_height, image_width, num_channels]
  print("batch_shape: ", batch_shape)
  
  return batch_shape 

InceptionSlimV3_created_graph = False
class InceptionSlimV3(object):
  image_size = 299
  num_labels = NUM_INCEPTION_CLASSES
  num_channels = 3
  def __init__(self, sess):
    
    self.sess = sess
    self.num_classes = NUM_INCEPTION_CLASSES
    self.built = False
  
  def predict(self, img, eval=True):
    global InceptionSlimV3_created_graph
    
    scaled = (0.5+tf.reshape(img,((-1,299,299,3))))*255
    
#    if not InceptionSlimV3_created_graph:
     = tf.import_graph_def(
      self.sess.graph.as_graph_def(),
#      input_map={'Cast:0': scaled},
#      return_elements=['softmax/logits:0'])
      input_map={'Placeholder:0': scaled},
      return_elements=['InceptionV3/Logits/SpatialSqueeze:0'], name="")
#      return_elements=['InceptionV3/Predictions/Softmax:0'])


        
    
    print("logits_tensor:", logits_tensor.__class__.__name__)
    print(logits_tensor)
    
    saver.restore(sess,checkpoint_path)
    
    
    
#    , feed_dict={'Placeholder:0': scaled}
    
    output_tensor= logits_tensor[0]
  
#    model_output=output_tensor.eval()
#    print(model_output.shape)
#    label=np.argmax(model_output)
#    print("label is ", label)
    InceptionSlimV3_created_graph = True
    
    return output_tensor


 


if __name__ == "__main__":
  with tf.Graph().as_default():
    '''  Esto ¿debería? ser equivalente a tf.train.import_meta_graph del fichero .meta
    pero usando el meta, da una salida con más información
    '''
    
#    from tensorflow.contrib.slim.nets import inception
    slim = tf.contrib.slim
    
    x_input = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))
    
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, end_points = inception.inception_v3(
            x_input, num_classes=NUM_INCEPTION_CLASSES , is_training=False,
            reuse=False)
    
    variables_to_restore = slim.get_model_variables()
    
    saver = tf.train.Saver(variables_to_restore)
    
    
    def predict(img):
      # Images for inception classifier are normalized to be in [-1, 1] interval.
      scaled = ((2.0 * tf.reshape(img,((-1,299,299,3))))/255) -1
      logits_tensor = tf.import_graph_def(
        sess.graph.as_graph_def(),

        input_map={'Placeholder:0': scaled},
        return_elements=['InceptionV3/Logits/SpatialSqueeze:0']) #, name="")
    
      print("logits_tensor:", logits_tensor.__class__.__name__)
      print(logits_tensor)
      
#      saver.restore(sess,checkpoint_path)
      
      return logits_tensor[0]

    with tf.Session() as sess:
      
      saver.restore(sess,checkpoint_path)
  
#      tf.logging.set_verbosity(tf.logging.INFO)
      
#     sample_image_00.png      actual label: 133
      image = np.array(scipy.misc.imresize(scipy.misc.imread('sample_image_00.png'),(299,299)),dtype=np.float32)
    
      model_output=predict(image).eval()
      print(model_output.shape)
      label=np.argmax(model_output)
      print("label is ", label)
      
      
      print("*"*400)
      
      
      
      #attack = CarliniL2(sess, model, batch_size=9, max_iterations=1000, confidence=0)
      attack = CarliniLi(sess, model, targeted=TARGETED, abort_early=True)
      
      
      
      
#      sess.run(tf.global_variables_initializer())
      targets=get_labels_KKKKKK(inputs, model, filenames=filenames)
      '''
      inputs, targets = generate_data(data, samples=1, targeted=False,
                                      start=0, inception=False)
      '''
#      sess.run(tf.global_variables_initializer())
      
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
