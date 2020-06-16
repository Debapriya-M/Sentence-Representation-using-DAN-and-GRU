# std lib imports
from typing import Dict

# external libs
import tensorflow as tf
from tensorflow.keras import layers, models


class SequenceToVector(models.Model):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``tf.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``tf.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : tf.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : tf.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2):
        super(DanSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        
        self.input_dim = input_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.hlayers = []
        self.hlayers.append(tf.keras.layers.Dense(self.input_dim, activation='relu', use_bias=True, input_shape = (self.input_dim,)))

        for number_of_layers in range(1, self.num_layers):
            self.hlayers.append(tf.keras.layers.Dense(self.input_dim, activation='relu', use_bias=True))


        # TODO(students): end



    def call(self,
             vector_sequence: tf.Tensor, #(64*209*50)
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:

        # TODO(students): start
        
        presentwords = vector_sequence * tf.expand_dims(sequence_mask, 2)
        probability = tf.random.uniform(
                        (vector_sequence.shape[0],vector_sequence.shape[1]),
                        minval=0,
                        maxval=1,
                        dtype=tf.float32,
                        seed=None,
                        name=None
                    )
        
        if training == True :
            prob_mask = tf.cast(probability > self.dropout, tf.float32)
            tokens = presentwords * tf.expand_dims(prob_mask, 2)
        else :
            tokens = presentwords

        averagetokens = tf.reduce_mean(tokens, axis = 1) #(64, 50)
        layer_representations = []
        layer_representations.append(self.hlayers[0](averagetokens))


        for i in range(1, self.num_layers):
            layer_representations.append(self.hlayers[i](layer_representations[i-1]))
            
        combined_vector = layer_representations[i]
        layer_representations = tf.transpose(layer_representations, [1, 0, 2])

        # TODO(students): end

        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}


class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int):
        super(GruSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        self.input_dim = input_dim
        self.num_layers = num_layers

        self.hiddenlayers = []
        self.hiddenlayers.append(tf.keras.layers.GRU(self.input_dim, activation='tanh', use_bias=True, input_shape = (self.input_dim,), 
            return_sequences=True, return_state = True, recurrent_activation = 'sigmoid'))


        for number_of_layers in range(1, self.num_layers):
            self.hiddenlayers.append(tf.keras.layers.GRU(self.input_dim, activation='tanh', use_bias=True, 
            return_sequences=True, return_state = True, recurrent_activation = 'sigmoid'))
        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start
        
        layer_representations = []
        op1, op2 = self.hiddenlayers[0](vector_sequence, mask = sequence_mask)
        layer_representations.append(op2)
        # print(layer_representations)
        for i in range(1, self.num_layers):   
            # print("Entering loop ... ")
            op1, op2 = self.hiddenlayers[i](op1, mask = sequence_mask)
            layer_representations.append(op2)
            # print(layer_representations)

        combined_vector = layer_representations[len(layer_representations) - 1]

        layer_representations = tf.transpose(layer_representations, [1, 0, 2])

        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}
