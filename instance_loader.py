
import os, sys
import random
import numpy as np
from functools import reduce

class InstanceLoader(object):

    def __init__(self,path):
        self.path = path
        self.filenames = [ path + '/' + x for x in os.listdir(path) ]
        random.shuffle(self.filenames)
        self.reset()
    #end

    def get_instances(self, n_instances):
        for i in range(n_instances):
            Ma,Mw,route = read_graph(self.filenames[self.index])
            yield Ma,Mw,route
            self.index += 1
        #end
    #end

    def create_batch(instances, target_cost_dev=None, target_cost=None):

        # n_instances: number of instances
        n_instances = len(instances)
        
        # n_vertices[i]: number of vertices in the i-th instance
        n_vertices  = np.array([ x[0].shape[0] for x in instances ])
        # n_edges[i]: number of edges in the i-th instance
        n_edges     = np.array([ len(np.nonzero(x[0])[0]) for x in instances ])
        # total_vertices: total number of vertices among all instances
        total_vertices  = sum(n_vertices)
        # total_edges: total number of edges among all instances
        total_edges     = sum(n_edges)

        # Compute matrices M, W, CV, CE
        # and vectors edges_mask and route_exists
        EV              = np.zeros((total_edges,total_vertices))
        W               = np.zeros((total_edges,1))
        C               = np.zeros((total_edges,1))
        edges_mask      = np.zeros(total_edges)
        route_exists    = np.zeros(n_instances)
        for (i,(Ma,Mw,route)) in enumerate(instances):
            
            # Get the number of vertices (n) and edges (m) in this graph
            n, m = n_vertices[i], n_edges[i]
            # Get the number of vertices (n_acc) and edges (m_acc) up until the i-th graph
            n_acc = sum(n_vertices[0:i])
            m_acc = sum(n_edges[0:i])

            # Get the list of edges in this graph
            edges = list(zip(np.nonzero(Ma)[0], np.nonzero(Ma)[1]))

            # Get the list of edges in the optimal TSP route for this graph
            route_edges = [ (min(x,y),max(x,y)) for (x,y) in zip(route,route[1:]) ]

            # Compute the optimal (normalized) TSP cost for this graph
            cost = sum([ Mw[x,y] for (x,y) in route_edges ]) / n

            # Choose a target cost and fill CV and CE with it
            if target_cost is None:
                delta = target_cost_dev*cost
                C[m_acc:m_acc+m,0] = cost + delta if i%2==0 else cost - delta
                route_exists[i] = 1 if i%2==0 else 0
            else:
                C[m_acc:m_acc+m,0] = target_cost
                route_exists[i] = 1 if target_cost >= cost else 0
            #end

            # Populate EV, W and edges_mask
            for e,(x,y) in enumerate(edges):
                EV[m_acc+e,n_acc+x] = 1
                EV[m_acc+e,n_acc+y] = 1
                W[m_acc+e] = Mw[x,y]
                if (x,y) in route_edges:
                    edges_mask[m_acc+e] = 1
                #end
            #end
        #end

        return EV, W, C, edges_mask, route_exists, n_vertices, n_edges
    #end

    def get_batches(self, batch_size, target_cost_dev):
        for i in range( len(self.filenames) // batch_size ):
            instances = list(self.get_instances(batch_size))
            instances = reduce(lambda x,y: x+y, zip(instances,instances))
            yield InstanceLoader.create_batch(instances, target_cost_dev)
        #end
    #end

    def reset(self):
        random.shuffle(self.filenames)
        self.index = 0
    #end
#end

def read_graph(filepath):
    with open(filepath,"r") as f:

        line = ''

        while 'DIMENSION' not in line: line = f.readline();

        n = int(line.split()[1])
        Ma = np.zeros((n,n),dtype=int)
        Mw = np.zeros((n,n),dtype=float)

        while 'EDGE_DATA_SECTION' not in line: line = f.readline();
        line = f.readline()
        
        while '-1' not in line:
            i,j = [ int(x) for x in line.split() ]
            Ma[i,j] = 1
            line = f.readline()
        #end
        line = f.readline()

        for i in range(n):
            Mw[i,:] = [ float(x) for x in f.readline().split() ]
        #end
        line = f.readline()

        route = [ int(x) for x in f.readline().split() ]

    #end
    return Ma,Mw,route
#end