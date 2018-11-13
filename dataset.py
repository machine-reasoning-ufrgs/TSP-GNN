
import sys, os, argparse, time, datetime
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

def create_graph(n, connectivity, distances='euc_2D', metric=True):

    # Init adjacency and weight matrices
    Ma = np.zeros((n,n))
    Mw = np.zeros((n,n))

    # Define adjacencies
    for i in range(n):
        Ma[i,i] = 0
        for j in range(i+1,n):
            Ma[i,j] = Ma[j,i] = int(np.random.rand() < connectivity)
        #end
    #end

    # Define weights
    nodes = None
    if distances == 'euc_2D':
        # Select 'n' points in the √2/2 × √2/2 square uniformly at random
        nodes = np.random.rand(n,2)
        for i in range(n):
            for j in range(i+1,n):
                Mw[i,j] = Mw[j,i] = np.sqrt(sum((nodes[i,:]-nodes[j,:])**2))
            #end
        #end
    elif distances == 'random':
        # Init all weights uniformly at random
        for i in range(n):
            for j in range(i+1,n):
                Mw[j,i] = Mw[i,j] = np.random.rand()
            #end
        #end
    #end

    # Enforce metric property, if requested
    if metric and distances != 'euc_2D':
        # Create networkx graph G
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from([ (i,j,{'weight':Mw[i,j]}) for i in range(n) for j in range(n) ])

        for i in range(n):
            for j in range(n):
                if i != j:
                    Mw[i,j] = nx.shortest_path_length(G,source=i,target=j,weight='weight')
                else:
                    Mw[i,j] = 0
                #end
            #end
        #end
    #end

    # Connect a random sequence of nodes in order to guarantee the existence of a Hamiltonian tour
    permutation = list(np.random.permutation(n))
    for (i,j) in zip(permutation,permutation[1:]+permutation[:1]):
        Ma[i,j] = Ma[j,i] = 1
    #end

    # Solve
    route = solve(Ma,Mw)
    if route is None:
        raise Exception('Unsolvable')
    #end

    return np.triu(Ma), Mw, route, nodes
#end

def create_dataset(path, nmin, nmax, conn_min=1, conn_max=1, samples=1000, distances='euc_2D', metric=True):

    if not os.path.exists(path):
        os.makedirs(path)
    #end if

    start_time = time.time()

    for i in range(samples):

        n = random.randint(nmin,nmax)

        # Create graph
        Ma,Mw,route,nodes = create_graph(n, np.random.uniform(conn_min,conn_max), distances=distances, metric=metric)

        # Write graph to file
        write_graph(Ma,Mw, filepath="{}/{}.graph".format(path,i), route=route)

        # Report progress
        if (i-1) % (samples//20) == 0:
            elapsed_time = time.time() - start_time
            remaining_time = (samples-i)*elapsed_time/(i+1)
            print('Dataset creation {}% Complete. Remaining time at this rate: {}'.format(int(100*i/samples), str(datetime.timedelta(seconds=remaining_time))), flush=True)
        #end
    #end
#end

def write_graph(Ma, Mw, filepath, route=None, int_weights=False, bins=10**6):
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

        if route is not None:
            # Write route
            out.write('TOUR_SECTION:\n')
            out.write('{}\n'.format(' '.join([str(x) for x in route])))
        #end

        out.write('EOF\n')
    #end
#end

if __name__ == '__main__':

    # Define argument parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-seed', type=int, default=42, help='RNG seed for Python, Numpy and Tensorflow')
    parser.add_argument('-distances', default='euc_2D', help='What type of distances? (euc_2D or random)')
    parser.add_argument('--metric', const=False, default=True, action='store_const', help='Create metric instances?')
    parser.add_argument('-samples', default=2**10, type=int, help='How many samples?')
    parser.add_argument('-path', help='Save path', required=True)
    parser.add_argument('-nmin', default=20, type=int, help='Min. number of vertices')
    parser.add_argument('-nmax', default=40, type=int, help='Max. number of vertices')
    parser.add_argument('-cmin', default=1, type=float, help='Min. connectivity')
    parser.add_argument('-cmax', default=1, type=float, help='Max. connectivity')
    parser.add_argument('-bins', default=10**6, help='Quantize edge weights in how many bins?')

    # Parse arguments from command line
    args = parser.parse_args()

    print('Creating {} instances'.format(vars(args)['samples']), flush=True)
    create_dataset(
        vars(args)['path'],
        vars(args)['nmin'], vars(args)['nmax'],
        vars(args)['cmin'], vars(args)['cmax'],
        samples=vars(args)['samples'],
        distribution=vars(args)['distribution']
    )
#end
