import tensorflow as tf
import numpy as np

from model import build_network
from util import load_weights
from dataset import create_graph_euc_2D
from instance_loader import InstanceLoader
import matplotlib as mpl
mpl.use( "Agg" )
from matplotlib import pyplot as plt

def get_embeddings_and_preds(sess, model, instance, time_steps, target_cost_dev=0.025):

    # Extract list of edges from instance
    Ma,Mw,route,nodes = instance
    edges = list(zip(np.nonzero(Ma)[0],np.nonzero(Ma)[1]))
    n = Ma.shape[0]
    m = len(edges)

    # Create batch of size 1
    route_cost = sum([ Mw[min(i,j),max(i,j)] for (i,j) in zip(route,route[1:]+route[:1]) ]) / n
    batch = InstanceLoader.create_batch([(Ma,Mw,route)], target_cost_dev=target_cost_dev)
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

    embeddings, predictions = sess.run([model['last_states'], model['predictions']], feed_dict=feed_dict)
    edge_embeddings = embeddings['E'].h
    vertex_embeddings = embeddings['V'].h

    return vertex_embeddings, edge_embeddings, predictions
#end

def process_instances():

    n = 20
    bins = 10**6
    connectivity = 1

    d = 64
    time_steps = 25
    
    num_deviations = 256
    num_instances = 512

    # Build model
    print('Building model ...')
    model = build_network(d)

    # Disallow GPU use
    config = tf.ConfigProto( device_count = {'GPU':0})
    with tf.Session(config=config) as sess:
        deviations = np.linspace( -1, 1, num_deviations, endpoint = True )
        probabilities = np.zeros( [ num_deviations, num_instances ] )
        
        # Initialize global variables
        print('Initializing global variables ...')
        sess.run( tf.global_variables_initializer() )

        # Restore saved weights
        load_weights(sess,'./TSP-checkpoints-0.025/epoch=400.0')

        for ins in range( num_instances ):
          print( "Instance {}".format( ins ) )
          # Create instance
          instance = create_graph_euc_2D(n,bins,connectivity)
          for dev in range( num_deviations ):
            _, _, p = get_embeddings_and_preds(
              sess,
              model,
              instance,
              time_steps,
              deviations[dev]
            )
            probabilities[dev,ins] = p
          #end for
        #end for
        
        plt_avg = np.mean( probabilities, axis = 1, keepdims = True )
        plt_std = np.std( probabilities, axis = 1, keepdims = True )
        plt_max = np.max( probabilities, axis = 1, keepdims = True )
        plt_min = np.min( probabilities, axis = 1, keepdims = True )


        plt_lower_std = np.max( np.concatenate( [plt_avg - plt_std, plt_min], axis = 1 ), axis = 1 )        
        plt_upper_std = np.min( np.concatenate( [plt_avg + plt_std, plt_max], axis = 1 ), axis = 1 )
        plt_avg = np.squeeze( plt_avg )
        plt_min = np.squeeze( plt_min )
        plt_max = np.squeeze( plt_max )
        
        figure, axis = plt.subplots()

        axis.plot( deviations, plt_max )
        axis.plot( deviations, plt_upper_std )
        axis.plot( deviations, plt_avg )
        axis.plot( deviations, plt_lower_std )
        axis.plot( deviations, plt_min )
        figure.savefig( "acceptance.eps" )
    #end
#end

if __name__ == '__main__':
    process_instances()
#end
