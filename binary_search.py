import tensorflow as tf
import numpy as np

from model import build_network
from util import load_weights
from dataset import create_graph_euc_2D
from instance_loader import InstanceLoader
import matplotlib as mpl
mpl.use( "Agg" )
from matplotlib import pyplot as plt

def get_cost_from_binary_search( sess, model, instance, time_steps, threshold = 0.5, stopping_delta = 0.01 ):
    # Extract list of edges from instance
    Ma, Mw, route, nodes = instance
    edges = list(zip(np.nonzero(Ma)[0],np.nonzero(Ma)[1]))
    n = Ma.shape[0]
    m = len(edges)
    # Get loose minimum and maximum
    wmin = np.minimum(
      np.sum( np.sort( np.triu( Mw ).flatten() )[:n] ),
      np.sum( np.sort( np.tril( Mw ).flatten() )[:n] )
    )
    wmax = np.maximum(
      np.sum( np.sort( np.triu( Mw ).flatten() )[-n:] ),
      np.sum( np.sort( np.tril( Mw ).flatten() )[-n:] )
    )
    # Normalize
    wmin /= n
    wmax /= n
    # Start at the middle
    wpred = (wmin+wmax)/2
    
    # Create batch of size 1
    route_cost = sum([ Mw[min(i,j),max(i,j)] for (i,j) in zip(route,route[1:]+route[:1]) ]) / n
    batch = InstanceLoader.create_batch([(Ma,Mw,route)], target_cost=wpred)
    EV, W, _, edges_mask, route_exists, n_vertices, n_edges = batch
    C = np.ones((m,1))
    
    iterations = 0
    
    while wmin < wpred * (1-stopping_delta) or wpred * (1+stopping_delta) < wmax:
        # Define feed dict
        feed_dict = {
            model['EV']           : EV,
            model['W']            : W,
            model['C']            : C * wpred,
            model['time_steps']   : time_steps,
            model['route_exists'] : route_exists,
            model['n_vertices']   : n_vertices,
            model['n_edges']      : n_edges
        }

        pred = sess.run(model['predictions'], feed_dict=feed_dict)
        if pred < threshold:
            wmin = wpred
        else:
            wmax = wpred
        #end if
        wpred = ( wmax + wmin ) / 2
        iterations += 1
    #end while
    return wpred, pred, route_cost, iterations
#end get_cost_from_binary_search

def process_instances():

    n = 20
    bins = 10**6
    connectivity = 1

    d = 64
    time_steps = 25
    num_instances = 512

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
        load_weights(sess,'./TSP-checkpoints-0.025/epoch=400.0')
        
        print(
            "pcost\tpprob\trcost\t%error\titer".format(
                pred_cost = "pc",
                pred_prob = "pp",
                real_cost = "rc",
                deviation = "%d",
                iterations = "it"
            )
        )
        
        for ins in range( num_instances ):          
          # Create instance
          instance = create_graph_euc_2D(n,bins,connectivity)
          pred_cost, pred_prob, real_cost, iterations = get_cost_from_binary_search(
            sess,
            model,
            instance,
            time_steps
          )
          print(
              "{pred_cost:6.5f}\t{pred_prob:6.5f}\t{real_cost:6.5f}\t{deviation:+05.4f}\t{iterations: 6d}".format(
                  pred_cost  = float( pred_cost ),
                  pred_prob  = float( pred_prob ),
                  real_cost  = float( real_cost ),
                  deviation  = 100.0 * float( (pred_cost - real_cost) / real_cost ),
                  iterations = iterations
              )
          )
        #end for
    #end session
#end process_instances

if __name__ == '__main__':
    process_instances()
#end
