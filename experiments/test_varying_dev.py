
import sys
sys.path.insert(0, '..')

import os, time, shutil, random, argparse
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

    EV, W, C, route_exists, n_vertices, n_edges = batch

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

    accuracies = sess.run(model['TP'], feed_dict=feed_dict)

    return np.mean(accuracies / len(n_vertices))
#end

if __name__ == '__main__':

    # Define argument parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d', default=64, type=int, help='Embedding size for vertices and edges')
    parser.add_argument('-time_steps', default=32, type=int, help='# Timesteps')
    parser.add_argument('-instances', default='../instances/test', help='Path for the test instances')
    parser.add_argument('-checkpoint', default='../training/dev=0.02/checkpoints/epoch=2000', help='Path for the checkpoint of the trained model')

    # Create folders
    if not os.path.exists('results'): os.makedirs('results');
    if not os.path.exists('figures'): os.makedirs('figures');

    # Parse arguments from command line
    args = parser.parse_args()

    # Setup parameters
    d                       = vars(args)['d']
    time_steps              = vars(args)['time_steps']

    # Create instance loader
    loader = InstanceLoader(vars(args)['instances'])

    # Build model
    print('Building model ...', flush=True)
    GNN = build_network(d)

    # Disallow GPU use
    config = tf.ConfigProto( device_count = {'GPU':0})
    with tf.Session() as sess:

        # Initialize global variables
        print("Initializing global variables ... ", flush=True)
        sess.run( tf.global_variables_initializer() )

        # Restore saved weights
        load_weights(sess,vars(args)['checkpoint']);

        # Run tests
        with open('results/test_varying_dev_TP.dat','w') as out:
            for dev in np.linspace(0,0.1,11)[1:]:
                loader.reset()
                accuracy = 0
                batch_size = 16
                for i,batch in enumerate(islice(loader.get_batches(batch_size,dev),64)):
                    accuracy += get_accuracy(sess,GNN,batch,time_steps)
                    if i % 4 == 0:
                        print('dev={} {}% Complete...'.format(dev,round(100*i/64)))
                    #end
                #end
                accuracy /= 64

                print('dev={} acc={}'.format(dev,accuracy))
                out.write('{} {}\n'.format(dev,accuracy))
                out.flush()
            #end
        #end

        # Create figure
        figure, axis = plt.subplots(figsize=(4,3))

        # Set axes' labels
        axis.set_xlabel('Deviation (%)')
        axis.set_ylabel('Accuracy (%)')

        # Set x-axis ticks
        axis.set_xticks(np.arange(1,10+1))

        # Draw guide lines
        for y in range(65,100+1,5):
            axis.axhline(y=y, linewidth=0.75, color='gray', zorder=2)
        #end

        # Draw vertical line indicating the deviation at which the model was trained
        axis.axvline(x=2, linewidth=1.5, linestyle='--', color='red', zorder=2)

        with open('results/test_varying_dev.dat') as f:
            data = np.array([ [float(x) for x in line.split()] for line in f.readlines()])
            axis.plot(100*data[:,0],100*data[:,1], marker='o', mfc='white')
        #end

        axis.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig('figures/test_varying_dev.eps', format='eps')
    #end

#end