
import sys, os, time, shutil, random, argparse
import tensorflow as tf
import numpy as np
from itertools import islice

from model import build_network
from instance_loader import InstanceLoader
from util import load_weights
from train import run_batch, summarize_epoch
from dataset import create_dataset

import matplotlib as mpl
mpl.use( "Agg" )
from matplotlib import pyplot as plt
import seaborn

def get_accuracy(sess, model, batch, time_steps):

    EV, W, C, edges_mask, route_exists, n_vertices, n_edges = batch

    # Define feed dict
    feed_dict = {
        model['EV']: EV,
        model['W']: W,
        model['C']: C,
        model['time_steps']: time_steps,
        model['route_exists']: route_exists,
        model['n_vertices']: n_vertices,
        model['n_edges']: n_edges
    }

    accuracies = sess.run(model['acc'], feed_dict=feed_dict)

    return np.mean(accuracies)
#end

if __name__ == '__main__':

    # Define argument parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d', default=64, type=int, help='Embedding size for vertices and edges')
    parser.add_argument('-dev', default=0.02, type=float, help='Target cost deviation')
    parser.add_argument('-instances', default='test', help='Path for the test instances')
    parser.add_argument('-checkpoint', default='training-0.02-small/checkpoints/epoch=2000', help='Path for the checkpoint of the trained model')

    # Parse arguments from command line
    args = parser.parse_args()

    # Setup parameters
    d                       = vars(args)['d']
    dev                     = vars(args)['dev']

    # Create instance loader
    loader = InstanceLoader(vars(args)['instances'])

    # Build model
    print('Building model ...', flush=True)
    GNN = build_network(d)

    # Disallow GPU use
    config = tf.ConfigProto( device_count = {'GPU':0})
    with tf.Session(config=config) as sess:

        # Initialize global variables
        print("Initializing global variables ... ", flush=True)
        sess.run( tf.global_variables_initializer() )

        # Restore saved weights
        load_weights(sess,vars(args)['checkpoint']);

        """with open('test_varying_timesteps.dat','a') as out:
            for t in range(0,40,5):

                # Create instance loader
                loader = InstanceLoader('test')

                accuracy = 0
                batch_size = 16
                for i,batch in enumerate(islice(loader.get_batches_diff(batch_size,dev),64)):
                    accuracy += get_accuracy(sess,GNN,batch,t)
                    if i % 4 == 0:
                        print('t={} {}% Complete...'.format(t,round(100*i/64)))
                    #end
                #end
                accuracy /= 64

                print('t={} acc={}'.format(t,accuracy))
                out.write('{} {}\n'.format(t,accuracy))
                out.flush()
            #end
        #end"""

        with open('test_varying_timesteps.dat') as f:
            data = np.array([ [float(x) for x in line.split()] for line in f.readlines()])
            
            figure, axis = plt.subplots(figsize=(4,4))

            axis.set_xlabel('Timesteps')
            axis.set_ylabel('Accuracy (%)')

            for y in range(50,80+1,5):
                axis.axhline(y=y, linewidth=0.75, color='gray', zorder=2)
            #end

            axis.plot(data[:,0],100*data[:,1], marker='o', mfc='white', mec='blue')
        #end

        plt.savefig('figures/test_varying_timesteps.eps', format='eps')

    #end

#end