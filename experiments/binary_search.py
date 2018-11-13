
import sys
sys.path.insert(0, '..')

import os, time, shutil, random, argparse
import tensorflow as tf
import numpy as np

from model import build_network
from util import load_weights
from instance_loader import InstanceLoader

def get_cost(sess, model, instance, time_steps, threshold = 0.5, stopping_delta = 0.01):
    
    # Extract information from instance
    Ma, Mw, route = instance
    edges = list(zip(np.nonzero(Ma)[0],np.nonzero(Ma)[1]))
    n = Ma.shape[0]
    m = len(edges)

    # Get loose minimum and maximum
    # wmin is the sum of the n lightest edges
    # wmax is the sum of the n heaviest edges
    wmin = np.minimum(
      np.sum( np.sort( np.triu( Mw ).flatten() )[:n] ),
      np.sum( np.sort( np.tril( Mw ).flatten() )[:n] )
    )
    wmax = np.maximum(
      np.sum( np.sort( np.triu( Mw ).flatten() )[-n:] ),
      np.sum( np.sort( np.tril( Mw ).flatten() )[-n:] )
    )
    # Normalize wmin and wmax
    wmin /= n
    wmax /= n

    # Start at the middle
    wpred = (wmin+wmax)/2
    
    # Create batch of size 1 with the given instance
    route_cost = sum([ Mw[min(i,j),max(i,j)] for (i,j) in zip(route,route[1:]+route[:1]) ]) / n
    batch = InstanceLoader.create_batch([(Ma,Mw,route)], target_cost=wpred)
    EV, W, _, route_exists, n_vertices, n_edges = batch
    C = np.ones((m,1))

    # Define feed dict
    feed_dict = {
        model['EV']           : EV,
        model['W']            : W,
        model['C']            : None,
        model['time_steps']   : time_steps,
        model['route_exists'] : route_exists,
        model['n_vertices']   : n_vertices,
        model['n_edges']      : n_edges
    }
    
    # Run binary search
    iterations = 0
    while wmin < wpred * (1-stopping_delta) or wpred * (1+stopping_delta) < wmax:

        # Update feed dict
        feed_dict[model['C']] = C*wpred

        # Get predictions from trained model
        pred = sess.run(model['predictions'], feed_dict=feed_dict)

        # Update binary search limits
        if pred < threshold:
            wmin = wpred
        else:
            wmax = wpred
        #end
        wpred = ( wmax + wmin ) / 2
        
        # Increment iterations
        iterations += 1
    #end
    return wpred, pred, route_cost, iterations
#end

if __name__ == '__main__':

    d = 64
    time_steps = 32
    num_instances = 512

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
        load_weights(sess,'../training/dev=0.02/checkpoints/epoch=100')

        # Init instance loader
        loader = InstanceLoader('../instances/test')

        avg_deviation = 0
        
        with open('results/binary-search.dat','w') as out:
            # Get instances from instance loader
            for instance in loader.get_instances(len(loader.filenames)):

                # Get number of cities
                n = instance[0].shape[0]

                # Compute cost with binary search
                pred_cost, pred_prob, real_cost, iterations = get_cost(sess,model,instance,time_steps)
                deviation = (pred_cost-real_cost)/real_cost

                avg_deviation += abs(deviation)
              
                # Write to file
                out.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(n,pred_cost,pred_prob[0],real_cost,deviation,iterations))
                out.flush()

                # Print
                print(
                    "{n} {pred_cost:6.5f}\t{pred_prob:6.5f}\t{real_cost:6.5f}\t{deviation:+05.4f}\t{iterations: 6d}".format(
                        n = n,
                        pred_cost  = float( pred_cost ),
                        pred_prob  = float( pred_prob ),
                        real_cost  = float( real_cost ),
                        deviation  = 100*deviation,
                        iterations = iterations
                    )
                )
            #end
        #end

    #end
#end
