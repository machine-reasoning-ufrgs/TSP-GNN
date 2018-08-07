import tensorflow as tf

class Mlp(object):
  def __init__(
    self,
    layer_sizes,
    output_size = None,
    activations = None,
    output_activation = None,
    use_bias = True,
    kernel_initializer = None,
    bias_initializer = tf.zeros_initializer(),
    kernel_regularizer = None,
    bias_regularizer = None,
    activity_regularizer = None,
    kernel_constraint = None,
    bias_constraint = None,
    trainable = True,
    name = None,
    name_internal_layers = True
  ):
    """Stacks len(layer_sizes) dense layers on top of each other, with an additional layer with output_size neurons, if specified."""
    self.layers = []
    internal_name = None
    # If object isn't a list, assume it is a single value that will be repeated for all values
    if not isinstance( activations, list ):
      activations = [ activations for _ in layer_sizes ]
    #end if
    # If there is one specifically for the output, add it to the list of layers to be built
    if output_size is not None:
      layer_sizes = layer_sizes + [output_size]
      activations = activations + [output_activation]
    #end if
    for i, params in enumerate( zip( layer_sizes, activations ) ):
      size, activation = params
      if name_internal_layers:
        internal_name = name + "_MLP_layer_{}".format( i + 1 )
      #end if
      new_layer = tf.layers.Dense(
        size,
        activation = activation,
        use_bias = use_bias,
        kernel_initializer = kernel_initializer,
        bias_initializer = bias_initializer,
        kernel_regularizer = kernel_regularizer,
        bias_regularizer = bias_regularizer,
        activity_regularizer = activity_regularizer,
        kernel_constraint = kernel_constraint,
        bias_constraint = bias_constraint,
        trainable = trainable,
        name = internal_name
      )
      self.layers.append( new_layer )
    #end for
  #end __init__
  
  def __call__( self, inputs, *args, **kwargs ):
    outputs = [ inputs ]
    for layer in self.layers:
      outputs.append( layer( outputs[-1] ) )
    #end for
    return outputs[-1]
  #end __call__
#end Mlp
