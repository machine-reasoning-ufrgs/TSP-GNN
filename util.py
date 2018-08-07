import time, sys, os, random
import numpy as np
import tensorflow as tf

def load_weights(sess,path,scope=None):
  if os.path.exists(path):
    # Restore saved weights
    print("Restoring saved model ... ")
    # Create model saver
    if scope is None:
      saver = tf.train.Saver()
    else:
      saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
    #end
    saver.restore(sess, "%s/model.ckpt" % path)
  else:
    raise Exception('Path does not exist!')
  #end if
#end

def save_weights(sess,path,scope=None):
  # Create /tmp/ directory to save weights
  if not os.path.exists(path):
    os.makedirs(path)
  #end if
  # Create model saver
  if scope is None:
    saver = tf.train.Saver()
  else:
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
  #end
  saver.save(sess, "%s/model.ckpt" % path)
  print( "MODEL SAVED IN PATH: {path}\n".format( path=path ) )
#end
