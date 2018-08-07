
import sys, os
import numpy as np
import random
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

    write_graph(Ma,Mw,[],'tmp',int_weights=True)
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

def create_graph_euc_2D(n, bins, connectivity):
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

    # Rescale and round weights, quantizing them into 'bins' integer bins
    Mw = np.round(bins * Mw)

    # Solve
    route = solve(Ma,Mw)
    if route == []: print('Unsolvable');

    # Check if route contains edges which are not in the graph and add them
    for (i,j) in [ (i,j) for (i,j) in zip(route,route[1:]+route[0:1]) if Ma[i,j] == 0 ]:
        Ma[i,j] = Ma[j,i] = 1
        Mw[i,j] = Mw[j,i] = 1
    #end

    # Remove huge costs from inexistent edges to simulate a disconnected instance
    for i in range(n):
        for j in range(n):
            if Ma[i,j] == 0:
                Mw[i,j] = 0

    # Rescale weights such that they are all ∈ [0,1]
    Mw = Mw / bins

    return np.triu(Ma), Mw, ([] if route is None else route), nodes
#end

def create_dataset(nmin, nmax, conn_min, conn_max, path, bins=10**6, connectivity=1, samples=1000):

    if not os.path.exists(path):
        os.makedirs(path)
    #end if

    for i in range(samples):

        # Sample different instances until we find one which admits a Hamiltonian route
        route = []
        while route == []:
            n = np.random.randint(nmin,nmax+1)
            connectivity = np.random.uniform(conn_min,conn_max)
            Ma,Mw,route,nodes = create_graph_euc_2D(n,bins,connectivity)
        #end

        # Write graph to file
        write_graph(Ma,Mw,route,"{}/{}.graph".format(path,i))
        if (i-1) % (samples//10) == 0:
            print('{}% Complete'.format(np.round(100*i/samples)), flush=True)
        #end
    #end
#end

def write_graph(Ma, Mw, route, filepath, int_weights=False):
    with open(filepath,"w") as out:

        n, m = Ma.shape[0], len(np.nonzero(Ma)[0])
        
        out.write('TYPE : TSP\n')

        out.write('DIMENSION: {n}\n'.format(n = n))

        out.write('EDGE_DATA_FORMAT: EDGE_LIST\n')
        out.write('EDGE_WEIGHT_TYPE: EXPLICIT\n')
        out.write('EDGE_WEIGHT_FORMAT: FULL_MATRIX \n')
        
        # List edges in the (generally not complete) graph
        out.write('EDGE_DATA_SECTION\n')
        for (i,j) in zip(list(np.nonzero(Ma))[0], list(np.nonzero(Ma))[1]):
            out.write("{} {}\n".format(i,j))
        #end
        out.write('-1\n')

        # Write edge weights as a complete matrix
        out.write('EDGE_WEIGHT_SECTION\n')
        for i in range(n):
            if int_weights:
                out.write('\t'.join([ str(int(Mw[i,j])) for j in range(n)]))
            else:
                out.write('\t'.join([ str(float(Mw[i,j])) for j in range(n)]))
            #end
            out.write('\n')
        #end

        # Write route as a concorde commentary
        out.write('TOUR_SECTION\n')
        out.write('{}\n'.format(' '.join([str(x) for x in route])))

        out.write('EOF\n')
    #end
#end