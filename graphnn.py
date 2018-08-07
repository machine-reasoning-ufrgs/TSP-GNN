import tensorflow as tf
from mlp import Mlp

class GraphNN(object):
  def __init__(
    self,
    var,
    mat,
    msg,
    loop,
    MLP_depth = 3,
    MLP_weight_initializer = tf.contrib.layers.xavier_initializer,
    MLP_bias_initializer = tf.zeros_initializer,
    RNN_cell = tf.contrib.rnn.LayerNormBasicLSTMCell,
    Cell_activation = tf.nn.relu,
    Msg_activation = tf.nn.relu,
    Msg_last_activation = None,
    float_dtype = tf.float32,
    name = 'GraphNN'
  ):
    """
    Receives three dictionaries: var, mat and msg.

    ○ var is a dictionary from variable names to embedding sizes.
      That is: an entry var["V1"] = 10 means that the variable "V1" will have an embedding size of 10.
    
    ○ mat is a dictionary from matrix names to variable pairs.
      That is: an entry mat["M"] = ("V1","V2") means that the matrix "M" can be used to mask messages from "V1" to "V2".
    
    ○ msg is a dictionary from function names to variable pairs.
      That is: an entry msg["cast"] = ("V1","V2") means that one can apply "cast" to convert messages from "V1" to "V2".
    
    ○ loop is a dictionary from variable names to lists of dictionaries:
      {
        "mat": the matrix name which will be used,
        "transpose?": if true then the matrix M will be transposed,
        "fun": transfer function (python function built using tensorflow operations,
        "msg": message name,
        "var": variable name
      }
      If "mat" is None, it will be the identity matrix,
      If "transpose?" is None, it will default to false,
      if "fun" is None, no function will be applied,
      If "msg" is false, no message conversion function will be applied,
      If "var" is false, then [1] will be supplied as a surrogate.
      
      That is: an entry loop["V2"] = [ {"mat":None,"fun":f,"var":"V2"}, {"mat":"M","transpose?":true,"msg":"cast","var":"V1"} ] enforces the following update rule for every timestep:
        V2 ← tf.append( [ f(V2), Mᵀ × cast(V1) ] )
    """
    self.var, self.mat, self.msg, self.loop, self.name = var, mat, msg, loop, name

    self.MLP_depth = MLP_depth
    self.MLP_weight_initializer = MLP_weight_initializer
    self.MLP_bias_initializer = MLP_bias_initializer
    self.RNN_cell = RNN_cell
    self.Cell_activation = Cell_activation
    self.Msg_activation = Msg_activation
    self.Msg_last_activation  = Msg_last_activation 
    self.float_dtype = float_dtype
    
    # Check model for inconsistencies
    self.check_model()
    
    # Initialize the parameters
    with tf.variable_scope(self.name):
      with tf.variable_scope('parameters'):
        self._init_parameters()
      #end parameter scope
    #end GraphNN scope
  #end __init__

  def check_model(self):
    # Procedure to check model for inconsistencies
    for v in self.var:
      if v not in self.loop:
        raise Warning('Variable {v} is not updated anywhere! Consider removing it from the model'.format(v=v))
      #end if
    #end for

    for v in self.loop:
      if v not in self.var:
        raise Exception('Updating variable {v}, which has not been declared!'.format(v=v))
      #end if
    #end for

    for mat, (v1,v2) in self.mat.items():
      if v1 not in self.var:
        raise Exception('Matrix {mat} definition depends on undeclared variable {v}'.format(mat=mat, v=v1))
      #end if
      if v2 not in self.var and type(v2) is not int:
        raise Exception('Matrix {mat} definition depends on undeclared variable {v}'.format(mat=mat, v=v2))
      #end if
    #end for

    for msg, (v1,v2) in self.msg.items():
      if v1 not in self.var:
        raise Exception('Message {msg} maps from undeclared variable {v}'.format(msg=msg, v=v1))
      #end if
      if v2 not in self.var:
        raise Exception('Message {msg} maps to undeclared variable {v}'.format(msg=msg, v=v2))
      #end if
    #end for
  #end check_model

  def _init_parameters(self):
    # Init LSTM cells
    self._RNN_cells = {
      v: self.RNN_cell(
       d,
       activation = self.Cell_activation
      ) for (v,d) in self.var.items()
    }
    # Init message-computing MLPs
    self._msg_MLPs = {
      msg: Mlp(
        layer_sizes          = [ self.var[vin] for _ in range( self.MLP_depth ) ],
        output_size          = self.var[vout],
        activations          = [ self.Msg_activation for _ in range( self.MLP_depth ) ],
        output_activation    = self.Msg_last_activation,
        kernel_initializer   = self.MLP_weight_initializer(),
        bias_initializer     = self.MLP_weight_initializer(),
        name                 = msg,
        name_internal_layers = True
      ) for msg, (vin,vout) in self.msg.items()
    }
  #end _init_parameters

  def __call__( self, adjacency_matrices, initial_embeddings, time_steps, LSTM_initial_states = {} ):
    with tf.variable_scope(self.name):
      with tf.variable_scope( "assertions" ):
        assertions = self.check_run( adjacency_matrices, initial_embeddings, time_steps, LSTM_initial_states )
      #end assertion variable scope
      with tf.control_dependencies( assertions ):
        states = {}
        for v, init in initial_embeddings.items():
          h0 = init
          c0 = tf.zeros_like(h0, dtype=self.float_dtype) if v not in LSTM_initial_states else LSTM_initial_states[v]
          states[v] = tf.contrib.rnn.LSTMStateTuple(h=h0, c=c0)
        #end
        
        # Build while loop body function
        def while_body( t, states ):
          new_states = {}
          for v in self.var:
            inputs = []
            for update in self.loop[v]:
              if 'var' in update:
                y = states[update['var']].h
                if 'fun' in update:
                  y = update['fun'](y)
                #end if
                if 'msg' in update:
                  y = self._msg_MLPs[update['msg']](y)
                #end if
                if 'mat' in update:
                  y = tf.matmul(
                    adjacency_matrices[update['mat']],
                    y,
                    adjoint_a = update['transpose?'] if 'transpose?' in update else False
                  )
                #end if
                inputs.append( y )
              else:
                inputs.append( adjacency_matrices[update['mat']] )
              #end if var in update
            #end for update in loop
            inputs = tf.concat( inputs, axis = 1 )
            with tf.variable_scope( '{v}_cell'.format( v = v ) ):
              _, new_states[v] = self._RNN_cells[v]( inputs = inputs, state = states[v] )
            #end cell scope
          #end for v in var
          return (t+1), new_states
        #end while_body
        
        _, last_states = tf.while_loop(
          lambda t, states: tf.less( t, time_steps ),
          while_body,
          [0,states]
        )
      #end assertions
    #end Graph scope
    return last_states
  #end __call__

  def check_run( self, adjacency_matrices, initial_embeddings, time_steps, LSTM_initial_states ):
    assertions = []
    # Procedure to check model for inconsistencies
    num_vars = {}
    for v, d in self.var.items():
      init_shape = tf.shape( initial_embeddings[v] )
      num_vars[v] = init_shape[0]
      assertions.append(
        tf.assert_equal(
          init_shape[1],
          d,
          data = [ init_shape[1] ],
          message = "Initial embedding of variable {v} doesn't have the same dimensionality {d} as declared".format(
            v = v,
            d = d
          )
        )
      )
      if v in LSTM_initial_states:
        lstm_init_shape = tf.shape( LSTM_initial_states[v] )
        assertions.append(
          tf.assert_equal(
            lstm_init_shape[1],
            d,
            data = [ lstm_init_shape[1] ],
            message = "Initial hidden state of variable {v}'s LSTM doesn't have the same dimensionality {d} as declared".format(
              v = v,
              d = d
            )
          )
        )
          
        assertions.append(
          tf.assert_equal(
            lstm_init_shape,
            init_shape,
            data = [ init_shape, lstm_init_shape ],
            message = "Initial embeddings of variable {v} don't have the same shape as the its LSTM's initial hidden state".format(
              v = v,
              d = d
            )
          )
        )
      #end if
    #end for v

    for mat, (v1,v2) in self.mat.items():
      mat_shape = tf.shape( adjacency_matrices[mat] )
      assertions.append(
        tf.assert_equal(
          mat_shape[0],
          num_vars[v1],
          data = [ mat_shape[0], num_vars[v1] ],
          message = "Matrix {m} doesn't have the same number of nodes as the initial embeddings of its variable {v}".format(
            v = v1,
            m = mat
          )
        )
      )
      if type(v2) is int:
        assertions.append(
          tf.assert_equal(
            mat_shape[1],
            v2,
            data = [ mat_shape[1], v2 ],
            message = "Matrix {m} doesn't have the same dimensionality {d} on the second variable as declared".format(
              m = mat,
              d = v2
            )
          )
        )
      else:
        assertions.append(
          tf.assert_equal(
            mat_shape[1],
            num_vars[v2],
            data = [ mat_shape[1], num_vars[v2] ],
            message = "Matrix {m} doesn't have the same number of nodes as the initial embeddings of its variable {v}".format(
              v = v2,
              m = mat
            )
          )
        )
      #end if-else
    #end for mat, (v1,v2)
    return assertions
  #end check_run
#end GraphNN
