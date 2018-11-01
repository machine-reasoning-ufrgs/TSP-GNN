
import sys
sys.path.insert(0, '..')

import os, time, shutil, random, argparse
import tensorflow as tf
import numpy as np
from itertools import islice

from train import run_batch
from model import build_network
from util import load_weights
from dataset import create_dataset
from dataset import create_graph_euc_2D
from instance_loader import InstanceLoader
import matplotlib as mpl
mpl.use( "Agg" )
from matplotlib import pyplot as plt
import seaborn

def get_predictions(sess, model, batch, time_steps):

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

    predictions = sess.run(model['predictions'], feed_dict=feed_dict)

    return predictions
#end

def process_instances():

    nmin,nmax = 20,40

    d = 64
    time_steps = 32
    num_deviations = 32

    # Create folders
    if not os.path.exists('results'): os.makedirs('results');
    if not os.path.exists('figures'): os.makedirs('figures');

    # Build model
    print('Building model ...')
    model = build_network(d)

    # Disallow GPU use
    config = tf.ConfigProto( device_count = {'GPU':0})
    with tf.Session(config=config) as sess:
        
        # Initialize global variables
        print('Initializing global variables ...')
        sess.run( tf.global_variables_initializer() )

        # Restore saved weights
        load_weights(sess,'training/dev=0.02/checkpoints/epoch=2000')

        deviations = np.linspace( -1, 1, num_deviations, endpoint = True )

        for n in range(20,20+1,5):

            # Create dataset
            if not os.path.isdir('test_acceptance/n={}'.format(n)):
                create_dataset(
                    n,n,
                    1,1,
                    bins=10**6,
                    samples=1024,
                    path='test_acceptance/n={}'.format(n),
                    dataset_type='euc_2D'
                )
            #end

            # Create instance loader
            loader = InstanceLoader('test_acceptance/n={}'.format(n))

            predictions = np.zeros(num_deviations)
            with open('acceptance_n={}-full.dat'.format(n),'w') as out:
                batch_size = 16
                for i,dev in enumerate(deviations):
                    loader.reset()
                    aux = []
                    for batch in islice(loader.get_batches(batch_size,dev),64):
                        aux.append(get_predictions(sess,model,batch,time_steps))
                    #end
                    predictions[i] = np.mean(aux)
                    print("Dev {} Pred {}".format(dev,predictions[i]))
                    out.write('{} {}\n'.format(dev,predictions[i]))
                    out.flush()
                #end
            #end
        #end

        exit()

        # Create figure
        figure, axis = plt.subplots(figsize=(4,4))

        axis.set_xlim(-0.1,0.1)
        #axis.set_ylim(0,100)
        #axis.set_xlabel('Deviation')
        #axis.set_ylabel(r'$\frac{∂}{∂x} Prediction (\%)$')
        axis.set_ylabel('Prediction (%)')
        for x in np.linspace(-0.15,0.15,7):
            axis.axvline(x=x, linewidth=0.75, color='gray', zorder=2)
        #end

        for y in range(0,100,10):
            axis.axhline(y=y, linewidth=0.75, color='gray', zorder=2)
        #end

        plt.tight_layout()

        # Paint between -2% and +2% which is the region comprehending the deviations with which the model was trained
        axis.axvspan(-0.02, 0.02, alpha=0.1, color='#ff9d8c')

        plt.tick_params(
        axis='x',          
        which='both',      
        bottom=False,      
        top=False,         
        labelbottom=False) 

        critical_points = []

        for i,n in enumerate(range(20,40+1,5)):
            # Read results
            with open('acceptance_n={}.dat'.format(n)) as f:
                data = np.array([ [float(x) for x in line.split()] for line in f.readlines()])
                dx = data[1,0]-data[0,0]
                dydx = np.array(np.gradient(100*data[:,1],dx))

                critical_points.append(data[dydx.argsort()[-1],0])

                #axis.plot(data[::2,0],dydx[::2], linestyle='-', linewidth=1, marker='', markersize=4, markeredgewidth=0.5, label='n={}'.format(n))
                axis.plot(data[::2,0],100*data[::2,1], linestyle='-', linewidth=1, marker='', markersize=4, markeredgewidth=0.5, label='n={}'.format(n))
                axis.legend()
            #end
        #end

        plt.savefig('acceptance-curves.eps', format='eps')

        with open('critical_points.dat'.format(n),'w') as f:
            for n,cp in zip(range(20,40+1,5),critical_points):
                f.write('{} {}\n'.format(n,cp))
            #end
        #end
    #end
#end

if __name__ == '__main__':
    process_instances()
#end
