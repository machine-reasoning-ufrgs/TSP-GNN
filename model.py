
import sys, os
import tensorflow as tf

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from graphnn import GraphNN
from mlp import Mlp

def build_network(d):

    # Define hyperparameters
    d = d
    learning_rate = 2e-5
    l2norm_scaling = 1e-10
    global_norm_gradient_clipping_ratio = 0.65

    # Placeholder for answers to the decision problems (one per problem)
    route_exists = tf.placeholder( tf.float32, shape = (None,), name = 'route_exists' )
    # Placeholders for the list of number of vertices and edges per instance
    n_vertices  = tf.placeholder( tf.int32, shape = (None,), name = 'n_vertices')
    n_edges     = tf.placeholder( tf.int32, shape = (None,), name = 'edges')
    # Placeholder for the adjacency matrix connecting each edge to its source and target vertices
    EV_matrix   = tf.placeholder( tf.float32, shape = (None,None), name = "EV" )
    # Placeholder for the column matrix of edge weights
    edge_weight = tf.placeholder( tf.float32, shape = (None,1), name = "edge_weight" )
    # Placeholder for route target costs (one per problem)
    target_cost = tf.placeholder( tf.float32, shape = (None,1), name = "target_cost" )
    # Placeholder for the number of timesteps the GNN is to run for
    time_steps  = tf.placeholder( tf.int32, shape = (), name = "time_steps" )
    
    # Define a MLP to compute an initial embedding for each edge, given its
    # weight and the target cost of the corresponding instance
    edge_init_MLP = Mlp(
        layer_sizes = [ d/8, d/4, d/2 ],
        activations = [ tf.nn.relu for _ in range(3) ],
        output_size = d,
        name = 'E_init_MLP',
        name_internal_layers = True,
        kernel_initializer = tf.contrib.layers.xavier_initializer(),
        bias_initializer = tf.zeros_initializer()
    )
    # Compute initial embeddings for edges
    edge_initial_embeddings = edge_init_MLP(tf.concat([ edge_weight, target_cost ], axis = 1))
    
    # All vertex embeddings are initialized with the same value, which is a trained parameter learned by the network
    total_n = tf.shape(EV_matrix)[1]
    v_init = tf.get_variable(initializer=tf.random_normal((1,d)), dtype=tf.float32, name='V_init')
    vertex_initial_embeddings = tf.tile(
        tf.div(v_init, tf.sqrt(tf.cast(d, tf.float32))),
        [total_n, 1]
    )

    # Define GNN dictionary
    GNN = {}

    # Define Graph neural network
    gnn = GraphNN(
        {
            # V is the set of vertex embeddings
            'V': d,
            # E is the set of edge embeddings
            'E': d
        },
        {
            # M is a E×V adjacency matrix connecting each edge to the vertices it is connected to
            'EV': ('E','V')
        },
        {
            # V_msg_E is a MLP which computes messages from vertex embeddings to edge embeddings
            'V_msg_E': ('V','E'),
            # E_msg_V is a MLP which computes messages from edge embeddings to vertex embeddings
            'E_msg_V': ('E','V')
        },
        {
            # V(t+1) ← Vu( EVᵀ × E_msg_V(E(t)) )
            'V': [
                {
                    'mat': 'EV',
                    'msg': 'E_msg_V',
                    'transpose?': True,
                    'var': 'E'
                }
            ],
            # E(t+1) ← Eu( EV × V_msg_E(V(t)) )
            'E': [
                {
                    'mat': 'EV',
                    'msg': 'V_msg_E',
                    'var': 'V'
                }
            ]
        },
        name='TSP'
    )

    # Populate GNN dictionary
    GNN['gnn']          = gnn
    GNN['route_exists'] = route_exists
    GNN['n_vertices']   = n_vertices
    GNN['n_edges']      = n_edges
    GNN['EV']           = EV_matrix
    GNN['W']            = edge_weight
    GNN['C']            = target_cost
    GNN['time_steps']   = time_steps

    # Define E_vote, which will compute one logit for each edge
    E_vote_MLP = Mlp(
        layer_sizes = [ d for _ in range(3) ],
        activations = [ tf.nn.relu for _ in range(3) ],
        output_size = 1,
        name = 'E_vote',
        name_internal_layers = True,
        kernel_initializer = tf.contrib.layers.xavier_initializer(),
        bias_initializer = tf.zeros_initializer()
        )
    
    # Get the last embeddings
    last_states = gnn(
      { "EV": EV_matrix, 'W': edge_weight, 'C': target_cost },
      { "V": vertex_initial_embeddings, "E": edge_initial_embeddings },
      time_steps = time_steps
    )
    GNN["last_states"] = last_states
    E_n = last_states['E'].h

    # Compute a vote for each embedding
    #E_vote = tf.reshape(E_vote_MLP( tf.concat([E_n,target_cost],axis=1) ), [-1])
    E_vote = tf.reshape(E_vote_MLP(E_n), [-1])

    # Compute the number of problems in the batch
    num_problems = tf.shape(n_vertices)[0]

    # Compute a logit probability for each problem
    pred_logits = tf.while_loop(
        lambda i, pred_logits: tf.less(i, num_problems),
        lambda i, pred_logits:
            (
                (i+1),
                pred_logits.write(
                    i,
                    tf.reduce_mean(E_vote[tf.reduce_sum(n_edges[0:i]):tf.reduce_sum(n_edges[0:i])+n_edges[i]])
                )
            ),
        [0, tf.TensorArray(size=num_problems, dtype=tf.float32)]
        )[1].stack()
    # Convert logits into probabilities
    GNN['predictions'] = tf.sigmoid(pred_logits)

    # Compute True Positives, False Positives, True Negatives, False Negatives, accuracy
    GNN['TP'] = tf.reduce_sum(tf.multiply(route_exists, tf.cast(tf.equal(route_exists, tf.round(GNN['predictions'])), tf.float32)))
    GNN['FP'] = tf.reduce_sum(tf.multiply(route_exists, tf.cast(tf.not_equal(route_exists, tf.round(GNN['predictions'])), tf.float32)))
    GNN['TN'] = tf.reduce_sum(tf.multiply(tf.ones_like(route_exists)-route_exists, tf.cast(tf.equal(route_exists, tf.round(GNN['predictions'])), tf.float32)))
    GNN['FN'] = tf.reduce_sum(tf.multiply(tf.ones_like(route_exists)-route_exists, tf.cast(tf.not_equal(route_exists, tf.round(GNN['predictions'])), tf.float32)))
    GNN['acc'] = tf.reduce_mean(tf.cast(tf.equal(route_exists, tf.round(GNN['predictions'])), tf.float32))

    # Define loss
    GNN['loss'] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=route_exists, logits=pred_logits))

    # Define optimizer
    optimizer = tf.train.AdamOptimizer(name='Adam', learning_rate=learning_rate)

    # Compute cost relative to L2 normalization
    vars_cost = tf.add_n([ tf.nn.l2_loss(var) for var in tf.trainable_variables() ])

    # Define gradients and train step
    grads, _ = tf.clip_by_global_norm(tf.gradients(GNN['loss'] + tf.multiply(vars_cost, l2norm_scaling),tf.trainable_variables()),global_norm_gradient_clipping_ratio)
    GNN['train_step'] = optimizer.apply_gradients(zip(grads, tf.trainable_variables()))
    
    # Return GNN dictionary
    return GNN
#end
