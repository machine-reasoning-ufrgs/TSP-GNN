
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

    accuracies = sess.run(model['acc'], feed_dict=feed_dict)

    return np.mean(accuracies)
#end

if __name__ == '__main__':

    # Define argument parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d', default=64, type=int, help='Embedding size for vertices and edges')
    parser.add_argument('-time_steps', default=32, type=int, help='# Timesteps')
    parser.add_argument('-dev', default=0.02, type=float, help='Target cost deviation')
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

        # Run tests
        for dev in [0.01,0.02,0.05,0.1]:
            with open('results/test_varying_sizes-{dev}.dat'.format(dev=dev),'w') as out:
                for n in range(20,80+1,5):
                    if not os.path.isdir('test_acceptance/n={}'.format(n)):
                        # Create dataset
                        create_dataset(
                            'test_acceptance/n={}'.format(n),
                            n,n,
                            1,1,
                            samples=1024,
                            distances='euc_2D'
                        )
                    #end

                    # Create instance loader
                    loader = InstanceLoader('test_acceptance/n={}'.format(n))

                    accuracy = 0
                    batch_size = 16
                    for i,batch in enumerate(islice(loader.get_batches(batch_size,dev),64)):
                        accuracy += get_accuracy(sess,GNN,batch,time_steps)
                        if i % 4 == 0:
                            print('n={} {}% Complete...'.format(n,round(100*i/64)))
                        #end
                    #end
                    accuracy /= 64

                    print('n={} acc={}'.format(n,accuracy))
                    out.write('{} {}\n'.format(n,accuracy))
                    out.flush()
                #end
            #end
        #end

        # Create figure
        figure, axis = plt.subplots(figsize=(4,4))

        # Set axes' labels
        axis.set_xlabel('n')
        axis.set_ylabel('Accuracy (%)')

        # Draw guide lines
        for y in range(50,100+1,5):
            axis.axhline(y=y, linewidth=0.75, color='gray', zorder=2)
        #end

        # Paint between n=20 and n=40 which is the region comprehending the instance sizes with which the model was trained
        axis.axvspan(20, 40, alpha=0.1, color='#f7c9c0')

        colors = ['red','green','blue','black']
        for i,dev in enumerate([0.01,0.02,0.05,0.1]):
            with open('results/test_varying_sizes-{dev}.dat'.format(dev=dev)) as f:
                data = np.array([ [float(x) for x in line.split()] for line in f.readlines()])
                axis.plot(data[:,0],100*data[:,1], marker='o', mfc='white', mec=colors[i], color=colors[i], label='{}% Dev.'.format(round(100*dev)))
            #end
        #end

        axis.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig('figures/test_varying_sizes.eps', format='eps')
    #end

#end