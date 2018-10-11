
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
            # Read graph from file
            Ma,Mw,route,target_cost,diff_edge = read_graph(self.filenames[self.index])

            Ma1 = Ma
            Ma2 = Ma.copy()

            if diff_edge is not None:
                # Create a (UNSAT/SAT) pair of instances with one edge of difference
                # The second instance has one edge more (diff_edge) which renders it SAT
                Ma2[diff_edge[0],diff_edge[1]] = 1
            #end

            # Yield both instances
            yield Ma1,Mw,route,target_cost
            yield Ma2,Mw,route,target_cost

            self.index += 1
        #end
    #end

    def create_batch(instances, dev=0.02, training_mode='deviation'):

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

        # Even index instances are UNSAT, odd are SAT
        route_exists = np.array([ i%2 for i in range(n_instances) ])

        for (i,(Ma,Mw,route,target_cost)) in enumerate(instances):
            # Get the number of vertices (n) and edges (m) in this graph
            n, m = n_vertices[i], n_edges[i]
            # Get the number of vertices (n_acc) and edges (m_acc) up until the i-th graph
            n_acc = sum(n_vertices[0:i])
            m_acc = sum(n_edges[0:i])

            # Get the list of edges in this graph
            edges = list(zip(np.nonzero(Ma)[0], np.nonzero(Ma)[1]))

            # Populate EV, W and edges_mask
            for e,(x,y) in enumerate(edges):
                EV[m_acc+e,n_acc+x] = 1
                EV[m_acc+e,n_acc+y] = 1
                W[m_acc+e] = Mw[x,y]
            #end

            # Compute the cost of the optimal route
            cost = sum([ Mw[min(x,y),max(x,y)] for (x,y) in zip(route,route[1:]+route[1:]) ])

            if training_mode == 'deviation':
                C[m_acc:m_acc+m,0] = (1+dev)*cost if i%2 == 0 else (1-dev)*cost
            elif training_mode == 'relational':
                C[m_acc:m_acc+m,0] = target_cost / n
            else:
                raise Exception('Unknown training mode!')
            #end
        #end

        return EV, W, C, edges_mask, route_exists, n_vertices, n_edges
    #end

    """
    def create_batch_diff(instances, target_cost_dev=None, target_cost=None):

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
            route_edges = [ (min(x,y),max(x,y)) for (x,y) in zip(route,route[1:]+route[:1]) ]

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
    
    def __create_batch_with_devs(instances, target_cost_devs=[]):
        # n_devs: number of deviations that are being repeated with every instance
        n_devs = len(target_cost_devs)

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
            route_edges = [ (min(x,y),max(x,y)) for (x,y) in zip(route,route[1:]+route[:1]) ]

            # Compute the optimal (normalized) TSP cost for this graph
            cost = sum([ Mw[x,y] for (x,y) in route_edges ]) / n

            # Choose a target cost and fill CV and CE with it
            delta = target_cost_devs[i%n_devs]*cost
            C[m_acc:m_acc+m,0] = cost + delta
            route_exists[i] = cost + delta >= cost

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
            route_edges = [ (min(x,y),max(x,y)) for (x,y) in zip(route,route[1:]+route[:1]) ]

            # Compute the optimal (normalized) TSP cost for this graph
            cost = sum([ Mw[x,y] for (x,y) in route_edges ]) / n

            if target_cost is not None:
                C[m_acc:m_acc+m,0] = target_cost
                route_exists[i] = 1 if target_cost > cost else 0
            else:
                # Choose a target cost and fill CV and CE with it
                delta = target_cost_dev*cost
                C[m_acc:m_acc+m,0] = cost + delta
                route_exists[i] = 1 if delta > 0 else 0
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

    def get_batches_with_devs(self, batch_size, deviations=[]):
        for i in range( len(self.filenames) // batch_size ):
            instances = list(self.get_instances(batch_size))
            instances = reduce(lambda x,y: x+y, zip(*[instances for _ in deviations]))
            yield InstanceLoader.__create_batch_with_devs(instances, deviations)
        #end
    #end
    """

    def get_batches(self, batch_size, dev, training_mode='deviation'):
        for i in range( len(self.filenames) // batch_size ):
            instances = list(self.get_instances(batch_size))
            yield InstanceLoader.create_batch(instances, dev=dev)
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

        # Parse number of vertices
        while 'DIMENSION' not in line: line = f.readline();
        n = int(line.split()[1])
        Ma = np.zeros((n,n),dtype=int)
        Mw = np.zeros((n,n),dtype=float)

        # Parse edges
        while 'EDGE_DATA_SECTION' not in line: line = f.readline();
        line = f.readline()
        while '-1' not in line:
            i,j = [ int(x) for x in line.split() ]
            Ma[i,j] = 1
            line = f.readline()
        #end

        # Parse edge weights
        while 'EDGE_WEIGHT_SECTION' not in line: line = f.readline();
        for i in range(n):
            Mw[i,:] = [ float(x) for x in f.readline().split() ]
        #end

        # Parse tour
        while 'TOUR_SECTION' not in line: line = f.readline();
        route = [ int(x) for x in f.readline().split() ]

        # Parse diff edge
        while 'DIFF_EDGE' not in line: line = f.readline();
        diff_edge = [ int(x) for x in f.readline().split() ]

        # Parse target cost
        while 'TARGET_COST' not in line: line = f.readline();
        target_cost = float(f.readline().strip())

    #end
    return Ma,Mw,route,target_cost,diff_edge
#end