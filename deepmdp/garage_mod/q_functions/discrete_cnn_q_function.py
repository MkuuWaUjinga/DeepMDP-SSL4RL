import torch
from torch import nn
from garage.torch.modules.mlp_module import MLPModule

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DiscreteCNNQFunction(nn.Module):
    """Q function based on a CNN for discrete action space.
        This class implements a Q value network to predict Q based on the
        input state and action. It uses an CNN and a MLP to fit the function
        of Q(s, a).
        Args:
            env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
            filter_dims (tuple[int]): Dimension of the filters. For example,
                (3, 5) means there are two convolutional layers. The filter for
                first layer is of dimension (3 x 3) and the second one is of
                dimension (5 x 5).
            num_filters (tuple[int]): Number of filters. For example, (3, 32) means
                there are two convolutional layers. The filter for the first layer
                has 3 channels and the second one with 32 channels.
            strides (tuple[int]): The stride of the sliding window. For example,
                (1, 2) means there are two convolutional layers. The stride of the
                filter for first layer is 1 and that of the second layer is 2.
            dense_sizes (tuple[int]): Output dimension of dense layer(s).
                For example, (32, 32) means the MLP of this q-function consists of
                two hidden layers, each with 32 hidden units.
            name (str): Variable scope of the cnn.
            padding (str): The type of padding algorithm to use,
                either 'SAME' or 'VALID'.
            max_pooling (bool): Boolean for using max pooling layer or not.
            cnn_hidden_nonlinearity (callable): Activation function for
                intermediate dense layer(s) in the CNN. It should return a
                tf.Tensor. Set it to None to maintain a linear activation.
            hidden_nonlinearity (callable): Activation function for intermediate
                dense layer(s) in the MLP. It should return a tf.Tensor. Set it to
                None to maintain a linear activation.
            hidden_w_init (callable): Initializer function for the weight
                of intermediate dense layer(s) in the MLP. The function should
                return a tf.Tensor.
            hidden_b_init (callable): Initializer function for the bias
                of intermediate dense layer(s) in the MLP. The function should
                return a tf.Tensor.
            output_nonlinearity (callable): Activation function for output dense
                layer in the MLP. It should return a tf.Tensor. Set it to None
                to maintain a linear activation.
            output_w_init (callable): Initializer function for the weight
                of output dense layer(s) in the MLP. The function should return
                a tf.Tensor.
            output_b_init (callable): Initializer function for the bias
                of output dense layer(s) in the MLP. The function should return
                a tf.Tensor.
            layer_normalization (bool): Bool for using layer normalization or not.
        """
    def __init__(self,
                 env_spec,
                 filter_dims=None,
                 num_filters=None,
                 strides=None,
                 dense_sizes=None,
                 input_shape=None,
                 padding_mode='same',
                 cnn_hidden_nonlinearity=nn.ReLU,
                 hidden_nonlinearity=nn.ReLU,
                 hidden_w_init=torch.nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=torch.nn.init.xavier_normal_,
                 output_b_init=torch.nn.init.zeros_):
        self._env_spec = env_spec
        self._action_dim = env_spec.action_space.n
        self._filter_dims = tuple(filter_dims)
        self._num_filters = tuple(num_filters)
        self._strides = tuple(strides)
        self._dense_sizes = tuple(dense_sizes)
        self._padding_mode = padding_mode
        self._cnn_hidden_nonlinearity = cnn_hidden_nonlinearity
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._input_shape = tuple(input_shape)

        super(DiscreteCNNQFunction, self).__init__()

        self.obs_dim = self._env_spec.observation_space.shape
        action_dim = self._env_spec.action_space.flat_dim

        # Init Cnn
        self.cnn = nn.Sequential(*list(self.cnn_module_generator()))

        # Init Mlp
        flattened_input_size = self._get_conv_output(self._input_shape)
        self.mlp = MLPModule(input_dim=flattened_input_size,
                             output_dim=action_dim,
                             hidden_sizes=list(self._dense_sizes),
                             hidden_nonlinearity=self._hidden_nonlinearity,
                             hidden_w_init=self._hidden_w_init,
                             hidden_b_init=self._hidden_b_init,
                             output_nonlinearity=self._output_nonlinearity,
                             output_w_init=self._output_w_init,
                             output_b_init=self._output_b_init)

    # Infer shape of tensor passed to mlp
    def _get_conv_output(self, shape):
        input = torch.autograd.Variable(torch.rand(1, *shape))
        output_feat = self.cnn(input)
        n_size = output_feat.data.view(1, -1).size(1)
        return n_size

    # pylint: disable=arguments-differ
    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.FloatTensor(x)
        x = x.to(device)
        if len(x.size()) == 4:
            x = x.permute(0, 3, 2, 1)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x.cpu()

    def cnn_module_generator(self):
        for input_dim, output_dim, filter_dim, stride in zip(
                (self.obs_dim[-1],) + self._num_filters, self._num_filters, self._filter_dims, self._strides):
            conv_layer = nn.Conv2d(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=filter_dim,
                stride=stride,
                padding_mode=self._padding_mode
            )
            self._hidden_w_init(conv_layer.weight)
            self._hidden_b_init(conv_layer.bias)
            yield conv_layer
            yield self._cnn_hidden_nonlinearity()

    def clone(self):
        return self.__class__(env_spec=self._env_spec,
                              input_shape=self._input_shape,
                              filter_dims=self._filter_dims,
                              num_filters=self._num_filters,
                              strides=self._strides,
                              dense_sizes=self._dense_sizes,
                              padding_mode=self._padding_mode,
                              hidden_nonlinearity=self._hidden_nonlinearity,
                              hidden_w_init=self._hidden_w_init,
                              hidden_b_init=self._hidden_b_init,
                              output_nonlinearity=self._output_nonlinearity,
                              output_w_init=self._output_w_init,
                              output_b_init=self._output_b_init)