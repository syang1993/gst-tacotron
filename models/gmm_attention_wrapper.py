import tensorflow as tf
from collections import namedtuple
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.rnn import RNNCell
from .rnn_wrappers import DecoderPrenetWrapper

GMMAttentionWrapperState = namedtuple(
    'GMMAttentionWrapperState',
    ['cell_out', 'cell_state', 'attention', 'time', 'alpha', 'beta', 'kappa', 'window', 'phi', 'alignment_history']
)
_zero_state_tensors = rnn_cell_impl._zero_state_tensors

class GMMAttentionWrapper(RNNCell):

  def __init__(self, 
               cell,
               window_size,
               num_attn_mixture,
               memory,
               memory_sequence_length=None,
               name="GMMAttention"):
    """Construct the GMM-based Attention mechanism wrapper.

    """
    self._cell = cell
    self._num_attn_mixture = num_attn_mixture
    self._memory = memory
    self._memory_sequence_length = memory_sequence_length
    self._window_size = window_size
    self._char_len = tf.shape(self._memory)[1]
    self._batch_size = tf.shape(self._memory)[0]

  @property
  def state_size(self):
    return GMMAttentionWrapperState(
        self._cell.state_size,
        self._cell.state_size,
        self._window_size,
        tf.TensorShape([]),
        self._num_attn_mixture,
        self._num_attn_mixture,
        self._num_attn_mixture,
        self._window_size,
        self._char_len,
        self._char_len
    )

  @property
  def output_size(self):
    return self._cell.state_size

  def zero_state(self, batch_size, dtype):
    initial_alignments = _zero_state_tensors(self._char_len, batch_size, dtype)
    return GMMAttentionWrapperState(
        tf.zeros([batch_size, self._cell.state_size]),
        tf.zeros([batch_size, self._cell.state_size]),
        tf.zeros([batch_size, self._window_size]),
        tf.zeros([], dtype=tf.int32),
        tf.zeros([batch_size, self._num_attn_mixture]),
        tf.zeros([batch_size, self._num_attn_mixture]),
        tf.zeros([batch_size, self._num_attn_mixture]),
        tf.zeros([batch_size, self._window_size]),
        tf.zeros([batch_size, self._char_len]),
        tf.TensorArray(
          dtype,
          size=0,
          dynamic_size=True, 
          element_shape=initial_alignments.shape)
    )
    
  def _get_params(self, cell_out, prev_kappa):
    """Compute window parameters
    
    In GMM-based attention, the location parameters kappa are defined 
    as offsets from the previous locations, and that the size of the 
    offset is constrained to be greater than zero. Then we get:

    alpha: the importance of the window within the mixture.
    beta: the width of the window.
    kappa: the location of the window.
    """
    window_params = tf.layers.dense(cell_out, units=3*self._num_attn_mixture)
    alpha, beta, kappa = tf.split(tf.exp(window_params), 3, axis=1)
    kappa = kappa + prev_kappa
    return alpha, beta, kappa

  def _gmm_score(self, alpha, beta, kappa):
    """Compute the window weights phi(t,u) of c_u at time t
    """

    u = tf.tile(
        tf.reshape(tf.range(self._char_len), (1, 1, self._char_len)), 
        (self._batch_size, self._num_attn_mixture, 1))
    u = tf.cast(u, tf.float32)
    phi = tf.reduce_sum(alpha * tf.exp(-tf.square(kappa - u) / beta), axis=1)

    return phi

  def _compute_attention(self, alpha, beta, kappa):
    """Compute the attention
    """
    phi_flat = self._gmm_score(alpha, beta, kappa)
    phi = tf.expand_dims(phi_flat, 2)

    sequence_mask = tf.cast(tf.sequence_mask(self._memory_sequence_length, maxlen=self._char_len), tf.float32)
    sequence_mask = tf.expand_dims(sequence_mask, 2)
    window = tf.reduce_sum(phi * self._memory * sequence_mask, axis=1)

    return phi_flat, window

  def __call__(self, query, state, scope="gmm_attention"):
    """Score the query based on the keys and values.
    """
    with tf.variable_scope(scope):
      # I concat the GMM windows and query as the inputs of the prenet (Alex Graves's paper)
      cur_inputs = tf.concat([query, state.window], axis=-1)
      cell_out, cell_state = self._cell(cur_inputs, state.cell_state)

      previous_alignment_history = [state.alignment_history]
      maybe_all_histories = []

      # GMM attention
      alpha_flat, beta_flat, kappa_flat = self._get_params(cell_out, state.kappa)
      alpha, beta, kappa = tf.expand_dims(alpha_flat, 2), tf.expand_dims(beta_flat, 2), tf.expand_dims(kappa_flat, 2)

      phi_flat, window_flat = self._compute_attention(alpha, beta, kappa)
      attention = window_flat

      alignment_history = previous_alignment_history[0].write(state.time, phi_flat)
      maybe_all_histories.append(alignment_history)

      new_state = GMMAttentionWrapperState(
          cell_out,
          cell_state,
          attention,
          state.time + 1,
          alpha_flat,
          beta_flat,
          kappa_flat,
          window_flat,
          phi_flat,
          maybe_all_histories[0]
      )

      #return tf.concat([query, cell_out], axis=-1), new_state
      return cell_out, new_state
