
import tensorflow as tf
import numpy as np

from model import build_network
from util import load_weights
from dataset import create_graph_euc_2D
from instance_loader import InstanceLoader

from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from scipy.interpolate import interp1d

def get_k_cluster(embeddings, k):

    # Perform k clustering
    k_clustering = KMeans(n_clusters=k).fit(embeddings)

    # Organize into list of (k) clusters
    clusters = [ [ j for j,x in enumerate(k_clustering.labels_) if x==i] for i in range(k) ]
    centers = k_clustering.cluster_centers_

    clusters_and_centers = sorted(zip(clusters,centers), key=lambda x:len(x[0]))
    clusters = [ x[0] for x in clusters_and_centers]
    centers = [ x[1] for x in clusters_and_centers]

    return clusters, centers
#end

def get_projections(embeddings, k):
    # Given a list of n-dimensional vertex embeddings, project them into k-dimensional space

    # Standarize dataset onto the unit scale (mean = 0, variance = 1)
    embeddings = StandardScaler().fit_transform(embeddings)

    # Get principal components
    principal_components = PCA(n_components=k).fit_transform(embeddings)

    return principal_components
#end

def extract_solution_clusters(n,embeddings):

    # Perform 2-clustering
    clusters, cluster_centers = get_k_cluster(embeddings,1)

    # Get list of predicted edges' indices
    predicted_edges_indices = clusters[0]

    # Select the 'n' points closest to their cluster center
    distances = [ np.linalg.norm(embeddings[e]-cluster_centers[0]) for e in predicted_edges_indices ]
    predicted_edges_indices = [ e for e,d in sorted(zip(predicted_edges_indices,distances), key=lambda x: -x[1])[:n]]

    return predicted_edges_indices
#end

def extract_solution_outliers(n,embeddings):

    LOF = LocalOutlierFactor(n_neighbors=1).fit(embeddings).negative_outlier_factor_

    #outliers = [ e for e,x in enumerate(LocalOutlierFactor(n_neighbors=1).fit_predict(embeddings)) if x == -1 ]

    n_outliers = [ e for e,x in sorted(enumerate(LOF),key=lambda x: x[1]) ][:n]

    return n_outliers
#end

def get_embeddings_and_preds(sess, model, instance, time_steps, target_cost_dev=0.025):

    # Extract list of edges from instance
    Ma,Mw,route,nodes = instance
    edges = list(zip(np.nonzero(Ma)[0],np.nonzero(Ma)[1]))
    n = Ma.shape[0]
    m = len(edges)

    # Create batch of size 1
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

def plot_projections_through_time(n=20, target_cost_dev=0.02):

    bins = 10**6
    connectivity = 1
    d = 64

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
        load_weights(sess,'./decision-checkpoints-0.05/epoch=100.0')

        # Create instance
        instance = create_graph_euc_2D(n,bins,connectivity)
        Ma,Mw,route,nodes = instance
        edges = list(zip(np.nonzero(Ma)[0],np.nonzero(Ma)[1]))
        label_edges = [ (min(i,j),max(i,j)) for i,j in zip(route,route[1:]) ]

        # Get embeddings
        vertex_embeddings, edge_embeddings, predictions = get_embeddings_and_preds(sess,model,instance,25,target_cost_dev)

        # Sequence of # of timesteps with which to evaluate the network
        time_steps = np.arange(5,32+1,5)

        # Init figure
        f, axes = plt.subplots(1, len(time_steps), dpi=200, sharex=True, sharey=True)

        # Iterate over timesteps
        for i,(t,ax) in enumerate(zip(time_steps,axes)):

            # Obtain embeddings
            vertex_embeddings, edge_embeddings, predictions = get_embeddings_and_preds(sess,model,instance,t,target_cost_dev)

            # Get 2 clustering
            clusters, centers = get_k_cluster(edge_embeddings, 2)

            # Compute embeddings 2D PCA projection
            edge_embeddings_pca = get_projections(edge_embeddings, 2)

            data = [ (edge_embeddings_pca[e], Mw[i,j], (i,j) in label_edges) for e,(i,j) in enumerate(edges) ]

            ax.scatter( [p[0] for p,w,r in data], [p[1] for p,w,r in data], c=[w for p,w,r in data], cmap='jet', edgecolors=['black' if r else 'none' for p,w,r in data] )

            # Set subplot title
            ax.set_title('{t} steps\npred:{pred:.0f}%'.format(t=t,pred=100*predictions[0]))

        #end

        plt.show()

    #end
#end

def process_instances():

    n = 30
    bins = 10**6
    connectivity = 1

    d = 64
    time_steps = 32

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
        load_weights(sess,'./training-0.02-small/checkpoints/epoch=1700')

        # Create instance
        instance = create_graph_euc_2D(n,bins,connectivity)
        Ma,Mw,route,nodes = instance
        edges = list(zip(np.nonzero(Ma)[0],np.nonzero(Ma)[1]))
        route_edges = [ (min(i,j),max(i,j)) for i,j in zip(route,route[1:]) ]
        route_edges_indices = [ e for e,(i,j) in enumerate(edges) if (i,j) in route_edges ]

        # Get embeddings
        vertex_embeddings, edge_embeddings, predictions = get_embeddings_and_preds(sess,model,instance,time_steps, 0.02)

        print('Prediction: {}%'.format(round(100*predictions[0],1)))

        # Compute predicted route
        predicted_route = [ edges[e] for e in extract_solution_clusters(n,edge_embeddings) ]
        print(len(predicted_route))

        #for i,j in predicted_route:
        #    plt.plot(nodes[[i,j],0],nodes[[i,j],1], c='black', zorder=1)
        ##end
        #plt.scatter(nodes[:,0],nodes[:,1], edgecolors='black', c='white', zorder=2)

        # Compute embeddings 2D PCA projection
        edge_embeddings_pca = get_projections(edge_embeddings, 2)
        edge_embeddings_pca1 = get_projections(edge_embeddings, 1)

        # Scatterplot projections
        
        bla = [ (p[0], p[1], Mw[edges[e]], e in route_edges_indices ) for e, p in enumerate( edge_embeddings_pca ) ]
        ble = [ (p, Mw[edges[e]], e in route_edges_indices ) for e, p in enumerate( edge_embeddings_pca1 ) ]

        #plt.scatter( [p for p,w,r in ble], [w for p,w,r in ble],  edgecolors=['black' if r else 'white' for p,w,r in ble], c= [w for p,w,r in ble], cmap='jet')
        plt.scatter( [p for p,_,_,_ in bla], [p for _,p,_,_ in bla], edgecolors=['black' if r else 'none' for _,_,_,r in bla], c= [w for _,_,w,_ in bla], cmap='jet')
        plt.show()

    #end
#end

def draw_solutions():

    # Init figure
    f, axes = plt.subplots(2,2, dpi=200, sharex=True, sharey=True, figsize=(4,4))

    sizes = [40,40,40,40]
    colors = ['red','green','blue','black']

    f.patch.set_visible(False)

    for k,ax in enumerate(axes.flatten()):
        n = sizes[k]
        # Create instance
        instance = create_graph_euc_2D(n,10**6,1)
        Ma,Mw,route,nodes = instance
        edges = list(zip(np.nonzero(Ma)[0],np.nonzero(Ma)[1]))

        ax.tick_params(
            axis='both',
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False
        )
        ax.axis('off')
        ax.axis('equal')
        ax.set(adjustable='box-forced', aspect='equal')

        ax.plot(nodes[route+route[:1],0], nodes[route+route[:1],1], c=colors[k], linewidth=0.25, zorder=1)

        ax.scatter(nodes[:,0],nodes[:,1], s=20, c='white', edgecolors='black', zorder=2, linewidth=0.25)

    #end

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig('figures/route-examples.eps', format='eps')
#end

def plot_training():

    data = open('training-0.02-small/log.dat').readlines()
    data = np.array([ [float(x) for x in line.split()] for line in data ])

    epoch = data[:,0]

    loss_train = data[:,1]
    acc_train = 100*data[:,2]

    loss_test = data[:,9]
    acc_test = 100*data[:,10]

    fig, ax1 = plt.subplots(dpi=100)

    color = 'tab:red'
    ax1.set_xlabel('Epoch', fontsize=16)
    ax1.set_ylabel('Loss', fontsize=16)
    ax1.plot(epoch[:2000], loss_train[:2000], color=color)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('Acc (%)', fontsize=16)
    ax2.plot(epoch[:2000], acc_train[:2000], color=color)
    ax2.tick_params(axis='y')
    ax2.grid(linestyle='-', axis='y')

    plt.savefig('figures/training-decision.eps',format='eps')
#end

if __name__ == '__main__':
    process_instances()
    #plot_projections_through_time(target_cost_dev=0.05)
    #draw_solutions()
    #plot_training()
#end