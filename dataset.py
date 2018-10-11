
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
    redirector_stdout = Redirector(fd=STDOUT)
    redirector_stderr = Redirector(fd=STDERR)

    # Write graph on a temporary file
    write_graph(Ma,Mw,filepath='tmp',int_weights=True)
    redirector_stderr.start()
    redirector_stdout.start()
    # Solve TSP on graph
    solver = TSPSolver.from_tspfile('tmp')
    # Get solution
    solution = solver.solve(verbose=False)
    redirector_stderr.stop()
    redirector_stdout.stop()

    """
        Concorde solves only symmetric TSP instances. To circumvent this, we
        fill inexistent edges with large weights. Concretely, an inexistent
        edge is assigned with the strictest upper limit to the cost of an
        optimal tour, which is n (the number of nodes) times 1 (the maximum weight).

        If one of these edges is used in the optimal solution, we can deduce
        that no valid tour exists (as just one of these edges costs more than
        all the others combined).

        OBS. in this case the maximum weight 1 is multiplied by 'bins' because
        we are converting from floating point to integers
    """
    if any([ Ma[i,j] == 0 for (i,j) in zip(list(solution.tour),list(solution.tour)[1:]+list(solution.tour)[:1]) ]):
        return None
    else:
        return list(solution.tour)
    #end
#end

def create_graph_euc_2D(n, connectivity):
    """
        Creates an euclidean 2D graph with 'n' vertices and the given
        connectivity
    """
    
    # Select 'n' 2D points in the unit square
    nodes = np.random.rand(n,2)
    # Multiply by 1/√2 such that all edge weights lie ∈ [0,1]
    nodes = nodes / np.sqrt(2)

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
        for j in range(i+1,n):
            Mw[i,j] = Mw[j,i] = np.sqrt(np.sum((nodes[i,:]-nodes[j,:])**2))
        #end
    #end

    # Add huge costs to inexistent edges to simulate a disconnected instance
    for i in range(n):
        for j in range(n):
            if Ma[i,j] == 0:
                Mw[i,j] = n+1
            #end
        #end
    #end

    # Solve
    route = solve(Ma,Mw)
    if route is None:
        raise Exception('Unsolvable')
    #end

    return np.triu(Ma), Mw, route, nodes
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

    # Enforce symmetry (but it does not matter because only edges (i,j) with i < j are added to the instance)
    for i in range(n):
        for j in range(i+1,n):
            Mw[j,i] = Mw[i,j]
        #end
    #end

    # Add huge costs to inexistent edges to simulate a disconnected instance
    for i in range(n):
        for j in range(n):
            if Ma[i,j] == 0:
                Mw[i,j] = n+1
            #end
        #end
    #end

    # Solve
    route = solve(Ma,Mw)
    if route is None:
        raise Exception('Unsolvable')
    #end

    # Check if route contains edges which are not in the graph and add them
    for (i,j) in [ (i,j) for (i,j) in zip(route,route[1:]) if Ma[i,j] == 0 ]:
        Ma[i,j] = Ma[j,i] = 1
    #end

    return np.triu(Ma), Mw, route, []
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
                    raise Exception("Graph not metric!")
                #end
            #end
        #end
    #end

    # Add huge costs to inexistent edges to simulate a disconnected instance
    for i in range(n):
        for j in range(n):
            if Ma[i,j] == 0:
                Mw[i,j] = n+1
            #end
        #end
    #end

    # Solve
    route = solve(Ma,Mw)
    if route is None:
        raise Exception('Unsolvable')
    #end

    # Check if route contains edges which are not in the graph and add them
    for (i,j) in [ (i,j) for (i,j) in zip(route,route[1:]) if Ma[i,j] == 0 ]:
        Ma[i,j] = Ma[j,i] = 1
    #end

    return np.triu(Ma), Mw, route, []
#end

def create_graph_diff_edge(n, dev=0.02):

    # Select 'n' 2D points in the unit square
    nodes = np.random.rand(n,2)

    # Choose connectivity at random
    connectivity = np.random.uniform(0.9,1)

    # Initialize graph with n nodes (initially disconnected)
    Ma = (np.random.rand(n,n) < connectivity).astype(float)
    for i in range(n):
        Ma[i,i] = 0
        for j in range(n):
            Ma[i,j] = Ma[j,i]
        #end
    #end
    # Build a weight matrix
    Mw = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            # Multiply by 1/√2 to normalize
            Mw[i,j] = (1.0/np.sqrt(2)) * np.sqrt(sum((nodes[i,:]-nodes[j,:])**2))
        #end
    #end

    # Connect a random sequence of nodes in order to guarantee the existence of a Hamiltonian tour
    permutation = list(np.random.permutation(n))
    for (i,j) in zip(permutation,permutation[1:]+permutation[:1]):
        Ma[i,j] = Ma[j,i] = 1
    #end

    # Init list of inexistent edges
    not_edges = [ (i,j) for i in range(n) for j in range(i+1,n) if Ma[i,j] == 0 ]
    random.shuffle(not_edges)

    # Solve instance
    route = solve(Ma,Mw)
    if route is None:
        raise Exception("Unsolvable Instance")
    #end

    # Compute cost
    Lopt0 = np.sum([ Mw[i,j] for (i,j) in zip(route,route[1:]+route[:1]) ])

    diff_edge = None
    for (i,j) in not_edges:
        # Add edge
        Ma[i,j] = Ma[j,i] = 1
        diff_edge = (i,j)
        # Solve TSP on G again
        route = solve(Ma,Mw)
        # Compute cost
        Lopt1 = np.sum([ Mw[i,j] for (i,j) in zip(route,route[1:]+route[:1]) ])
        # Deviation from Lopt0
        if (Lopt1-Lopt0)/Lopt0 > -dev:
            break
        else:
            # Backtrack
            Ma[i,j] = Ma[j,i] = 0
        #end
    #end

    if diff_edge is None:
        raise Exception("Could not create diff_edge instance!")
    #end

    # Erase diff edge
    (i,j) = diff_edge
    Ma[i,j] = Ma[j,i] = 0

    # Compute target cost as the mean value between Lopt0 and Lopt1
    target_cost = (Lopt1+Lopt0)/2

    return np.triu(Ma), Mw, target_cost, diff_edge, nodes
#end

def create_dataset(nmin, nmax, path, conn_min=1, conn_max=1, samples=1000, distribution='euc_2D'):

    if not os.path.exists(path):
        os.makedirs(path)
    #end if

    for i in range(samples):

        n = random.randint(nmin,nmax)
        conn = np.random.uniform(conn_min,conn_max)

        route = target_cost = diff_edge = None

        # Create graph
        if distribution == 'euc_2D':
            Ma,Mw,route,nodes = create_graph_euc_2D(n,conn)
        elif distribution == 'random':
            Ma,Mw,route,nodes = create_graph_random(n,conn)
        elif distribution == 'random_metric':
            Ma,Mw,route,nodes = create_graph_random_metric(n,conn)
        elif distribution == 'diff_edge':
            creation_successful = False
            while not creation_successful:
                try:
                    Ma,Mw,target_cost,diff_edge,nodes = create_graph_diff_edge(n)
                    creation_successful = True
                except:
                    creation_successful = False
                #end
            #end
        else:
            raise Exception('Unknown dataset type')
        #end

        # Write graph to file
        write_graph(Ma,Mw, filepath="{}/{}.graph".format(path,i), route=route, diff_edge=diff_edge, target_cost=target_cost)

        # Report progress
        if (i-1) % (samples//20) == 0:
             print('Dataset creation {}% Complete'.format(int(100*i/samples)), flush=True)
        #end

    #end
#end

def write_graph(Ma, Mw, filepath, route=None, diff_edge=None, target_cost=None, int_weights=False, bins=10**6):
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
            for j in range(n):
                if Ma[i,j] == 1:
                    out.write(str( int(bins*Mw[i,j]) if int_weights else Mw[i,j]))
                else:
                    out.write(str(n*bins+1 if int_weights else 0))
                #end
                out.write(' ')
            #end
            out.write('\n')
        #end

        if route is None: route = [-1]*n;
        if diff_edge is None: diff_edge = (-1,-1);
        if target_cost is None: target_cost = -1;

        # Write route
        out.write('TOUR_SECTION:\n')
        out.write('{}\n'.format(' '.join([str(x) for x in route])))

        # Write diff edge
        out.write('DIFF_EDGE:\n')
        out.write('{}\n'.format(' '.join([str(x) for x in diff_edge])))

        # Write target cost
        out.write('TARGET_COST:\n')
        out.write('{}\n'.format(str(target_cost)))

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