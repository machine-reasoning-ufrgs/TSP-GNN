
import sys, os, argparse
import numpy as np
import random
import networkx as nx
from concorde.tsp import TSPSolver
from redirector import Redirector

def solve(Ma, Mw):
    """
        Invokes Concorde to solve a TSP instance

        Uses Python's Redirector library to prevent Concorde from printing
        unsufferably verbose messages
    """

    STDOUT = 1
    STDERR = 2

    write_graph(Ma,Mw,'tmp',int_weights=True)
    redirector_stdout = Redirector(fd=STDOUT)
    redirector_stderr = Redirector(fd=STDERR)
    redirector_stderr.start()
    redirector_stdout.start()
    solver = TSPSolver.from_tspfile('tmp')
    solution = solver.solve(verbose=False)
    redirector_stderr.stop()
    redirector_stdout.stop()

    return list(solution.tour) if solution.found_tour else []
#end

def create_graph_erdos_renyi(n, target_cost):

    # Select 'n' 2D points in the unit square
    nodes = np.random.rand(n,2)

    # Initialize graph with n nodes (initially disconnected)
    Ma = np.zeros((n,n))
    # Build a weight matrix
    Mw = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            # Multiply by 1/√2 to normalize
            Mw[i,j] = (1.0/np.sqrt(2)) * np.sqrt((nodes[i,0]-nodes[j,0])**2+(nodes[i,1]-nodes[j,1])**2)
        #end
    #end

    # To ensure that G admits at least one Hamiltonian path, sample a random
    # permutation of [0..n] and add the corresponding edges to G
    permutation = list(np.random.permutation(range(n)))
    edges = zip(permutation, permutation[1:]+permutation[:1])
    for i,j in edges:
        Ma[i,j] = 1
        Ma[j,i] = 1
    #end

    # Init list of inexistent edges
    not_edges = [ (i,j) for i in range(n) for j in range(i+1,n) if Ma[i,j] == 0 ]

    # While Lopt(G) > target_cost, add edges to G
    route = solve(Ma,Mw)
    Lopt = np.sum([ Mw[i,j] for (i,j) in zip(route,route[1:]+route[:1]) ])
    diff_edge = None
    while Lopt > target_cost:
        if len(not_edges) == 0:
            raise Exception("Could not create graph with Lopt < {}. Current Lopt={}".format(target_cost,Lopt))
        #end
        # Add a random edge to G
        (i,j) = not_edges.pop(random.randrange(len(not_edges)))
        Ma[i,j] = 1
        Ma[j,i] = 1
        diff_edge = (i,j)
        # Solve TSP on G again
        route = solve(Ma,Mw)
        Lopt = np.sum([ Mw[i,j] for (i,j) in zip(route,route[1:]+route[:1]) ])
    #end

    return np.triu(Ma), Mw, diff_edge, route
#end

def create_graph_euc_2D(n, connectivity):
    """
        Creates an euclidean 2D graph with 'n' vertices and the given
        connectivity

        Concretely, we sample 'n' points in the unit square and link every
        pair with an edge whose weight is given by their euclidean distance.
        Then some edges are erased to match the required connectivity

        the 'bins' parameter determines in how many bins the edge weights are
        going to be quantized (this is required as Concorde deals with
        integer-valued weights)
    """
    
    # Select 'n' 2D points in the unit square
    nodes = np.random.rand(n,2)

    # Build an adjacency matrix with given connectivity
    Ma = (np.random.rand(n,n) < connectivity).astype(int)
    for i in range(n):
        Ma[i,i] = 0
        for j in range(i+1,n):
            Ma[i,j] = Ma[j,i]
        #end
    #end
    
    # Build a weight matrix
    Mw = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            # Multiply by 1/√2 to normalize
            Mw[i,j] = (1.0/np.sqrt(2)) * np.sqrt((nodes[i,0]-nodes[j,0])**2+(nodes[i,1]-nodes[j,1])**2)
        #end
    #end

    # Add huge costs to inexistent edges to simulate a disconnected instance
    for i in range(n):
        for j in range(n):
            if Ma[i,j] == 0:
                Mw[i,j] = n+1

    # Solve
    route = solve(Ma,Mw)
    if route == []: print('Unsolvable');

    # Check if route contains edges which are not in the graph and add them
    for (i,j) in [ (i,j) for (i,j) in zip(route,route[1:]) if Ma[i,j] == 0 ]:
        Ma[i,j] = Ma[j,i] = 1
        Mw[i,j] = Mw[j,i] = 1
    #end

    # Remove huge costs from inexistent edges to simulate a disconnected instance
    for i in range(n):
        for j in range(n):
            if Ma[i,j] == 0:
                Mw[i,j] = 0

    return np.triu(Ma), Mw, ([] if route is None else route), nodes
#end

def shortest_paths(n, Ma, Mw):

    D = [float('inf')]*n
    visited = np.zeros(n,dtype=bool)

    while len(visited) > 0:
        i = visited.pop()
    #end
#end

def create_graph_random_metric(n, connectivity):
    """
        Creates a graph 'n' vertices and the given connectivity. Edge weights
        are initialized randomly.
    """

    # Build an adjacency matrix with given connectivity
    Ma = (np.random.rand(n,n) < connectivity).astype(int)
    for i in range(n):
        Ma[i,i] = 0
        for j in range(i+1,n):
            Ma[i,j] = Ma[j,i]
        #end
    #end
    
    # Build a weight matrix
    Mw = np.random.rand(n,n)

    # Create networkx graph G
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from([ (i,j,{'weight':Mw[i,j]}) for i in range(n) for j in range(n) ])

    # Enforce metric property
    for i in range(n):
        for j in range(n):
            if i != j:
                Mw[i,j] = nx.shortest_path_length(G,source=i,target=j,weight='weight')
            else:
                Mw[i,j] = 0
            #end
        #end
    #end

    # Check for metric property
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if Mw[i,j] + Mw[j,k] < Mw[i,k]:
                    print("Graph not metric!")

    # Enforce symmetry (but it does not matter because only edges (i,j) with i < j are added to the instance)
    for i in range(n):
        for j in range(i+1,n):
            Mw[j,i] = Mw[i,j]

    # Add huge costs to inexistent edges to simulate a disconnected instance
    for i in range(n):
        for j in range(n):
            if Ma[i,j] == 0:
                Mw[i,j] = n+1

    # Solve
    route = solve(Ma,Mw)
    if route == []: print('Unsolvable');

    # Check if route contains edges which are not in the graph and add them
    for (i,j) in [ (i,j) for (i,j) in zip(route,route[1:]) if Ma[i,j] == 0 ]:
        Ma[i,j] = Ma[j,i] = 1
        Mw[i,j] = Mw[j,i] = 1
    #end

    # Remove huge costs from inexistent edges to simulate a disconnected instance
    for i in range(n):
        for j in range(n):
            if Ma[i,j] == 0:
                Mw[i,j] = 0

    return np.triu(Ma), Mw, ([] if route is None else route), []
#end

def create_graph_random(n, connectivity):
    """
        Creates a graph 'n' vertices and the given connectivity. Edge weights
        are initialized randomly.
    """

    # Build an adjacency matrix with given connectivity
    Ma = (np.random.rand(n,n) < connectivity).astype(int)
    for i in range(n):
        Ma[i,i] = 0
        for j in range(i+1,n):
            Ma[i,j] = Ma[j,i]
        #end
    #end
    
    # Build a weight matrix
    Mw = np.random.rand(n,n)

    # Create networkx graph G
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from([ (i,j,{'weight':Mw[i,j]}) for i in range(n) for j in range(n) ])

    # Enforce symmetry (but it does not matter because only edges (i,j) with i < j are added to the instance)
    for i in range(n):
        for j in range(i+1,n):
            Mw[j,i] = Mw[i,j]

    # Add huge costs to inexistent edges to simulate a disconnected instance
    for i in range(n):
        for j in range(n):
            if Ma[i,j] == 0:
                Mw[i,j] = n+1

    # Solve
    route = solve(Ma,Mw)
    if route == []: print('Unsolvable');

    # Check if route contains edges which are not in the graph and add them
    for (i,j) in [ (i,j) for (i,j) in zip(route,route[1:]) if Ma[i,j] == 0 ]:
        Ma[i,j] = Ma[j,i] = 1
        Mw[i,j] = Mw[j,i] = 1
    #end

    # Remove huge costs from inexistent edges to simulate a disconnected instance
    for i in range(n):
        for j in range(n):
            if Ma[i,j] == 0:
                Mw[i,j] = 0

    return np.triu(Ma), Mw, ([] if route is None else route), []
#end

def create_dataset(nmin, nmax, conn_min, conn_max, path, connectivity=1, samples=1000, dataset_type='euc_2D'):

    if not os.path.exists(path):
        os.makedirs(path)
    #end if

    for i in range(samples):

        # Sample different instances until we find one which admits a Hamiltonian route
        route = []
        while route == []:
            n = np.random.randint(nmin,nmax+1)
            connectivity = np.random.uniform(conn_min,conn_max)
            if dataset_type == 'euc_2D':
                Ma,Mw,route,nodes = create_graph_euc_2D(n,connectivity)
            elif dataset_type == 'random':
                Ma,Mw,route,nodes = create_graph_random(n,connectivity)
            elif dataset_type == 'random_metric':
                Ma,Mw,route,nodes = create_graph_random_metric(n,connectivity)
            elif dataset_type == 'erdos_renyi':
                target_cost = 0.7080*np.sqrt(n) + 0.522
                Ma,Mw,diff_edge,route = create_graph_erdos_renyi(n, target_cost)
            else:
                raise Exception('Unknown dataset type')
            #end
        #end

        # Write graph to file
        if dataset_type == 'erdos_renyi':
            write_graph(Ma,Mw,"{}/{}.graph".format(path,i), diff_edge=diff_edge,)
        else:
            write_graph(Ma,Mw,route,"{}/{}.graph".format(path,i))
        #end

        if (i-1) % (samples//10) == 0:
             print('{}% Complete'.format(np.round(100*i/samples)), flush=True)
        #end

    #end
#end

def write_graph(Ma, Mw, filepath, route=None, diff_edge=None, int_weights=False, bins=10**6):
    with open(filepath,"w") as out:

        n, m = Ma.shape[0], len(np.nonzero(Ma)[0])
        
        out.write('TYPE : TSP\n')

        out.write('DIMENSION: {n}\n'.format(n = n))

        out.write('EDGE_DATA_FORMAT: EDGE_LIST\n')
        out.write('EDGE_WEIGHT_TYPE: EXPLICIT\n')
        out.write('EDGE_WEIGHT_FORMAT: FULL_MATRIX \n')
        
        # List edges in the (generally not complete) graph
        out.write('EDGE_DATA_SECTION:\n')
        for (i,j) in zip(list(np.nonzero(Ma))[0], list(np.nonzero(Ma))[1]):
            out.write("{} {}\n".format(i,j))
        #end
        out.write('-1\n')

        # Write edge weights as a complete matrix
        out.write('EDGE_WEIGHT_SECTION:\n')
        for i in range(n):
            if int_weights:
                out.write('\t'.join([ str(int(bins*Mw[i,j])) if Ma[i,j] == 1 else str(2*bins) for j in range(n)]))
            else:
                out.write('\t'.join([ str(float(Mw[i,j])) for j in range(n)]))
            #end
            out.write('\n')
        #end

        if not route is None:
            # Write route
            out.write('TOUR_SECTION:\n')
            out.write('{}\n'.format(' '.join([str(x) for x in route])))
        #end

        if not diff_edge is None:
            # Write diff edge (for erdos renyi distr. only)
            out.write('DIFF_EDGE:\n')
            out.write('{}\n'.format(' '.join([str(x) for x in diff_edge])))
        #end

        out.write('EOF\n')
    #end
#end

if __name__ == '__main__':

    # Define argument parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-seed', type=int, default=42, help='RNG seed for Python, Numpy and Tensorflow')
    parser.add_argument('-type', default='euc_2D', help='Which type of dataset? (euc_2D, random, random_metric, erdos_renyi)')
    parser.add_argument('-samples', default=2**10, type=int, help='How many samples?')
    parser.add_argument('-path', help='Save path', required=True)
    parser.add_argument('-nmin', default=20, type=int, help='Min. number of vertices')
    parser.add_argument('-nmax', default=40, type=int, help='Max. number of vertices')
    parser.add_argument('-cmin', default=1, help='Min. connectivity')
    parser.add_argument('-cmax', default=1, help='Max. connectivity')
    parser.add_argument('-bins', default=10**6, help='Quantize edge weights in how many bins?')

    # Parse arguments from command line
    args = parser.parse_args()

    print('Creating {} instances'.format(vars(args)['samples']), flush=True)
    create_dataset(
        vars(args)['nmin'], vars(args)['nmax'],
        vars(args)['cmin'], vars(args)['cmax'],
        samples=vars(args)['samples'],
        path=vars(args)['path'],
        dataset_type=vars(args)['type']
    )
#end