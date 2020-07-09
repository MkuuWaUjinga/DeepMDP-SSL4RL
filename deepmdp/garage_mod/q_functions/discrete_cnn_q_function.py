import torch
from torch import nn
from dowel import logger
from deepmdp.garage_mod.modules.mlp_module import MLPModule

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
                 encoder=None,
                 head=None,
                 hidden_nonlinearity=nn.ReLU,
                 hidden_w_init=torch.nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=torch.nn.init.xavier_normal_,
                 output_b_init=torch.nn.init.zeros_,
                 layer_norm=False):
        self._encoder_config = encoder
        self._head_config = head
        self._env_spec = env_spec
        self._action_dim = env_spec.action_space.n
        self._cnn_hidden_nonlinearity = hidden_nonlinearity
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self.obs_dim = self._env_spec.observation_space.shape
        self._layer_norm = layer_norm

        super(DiscreteCNNQFunction, self).__init__()

        action_dim = self._env_spec.action_space.flat_dim

        # Init Encoder
        self._input_shape = self._encoder_config["input_shape"]
        if "filter_dims" in self._encoder_config:
            self._filter_dims = tuple(self._encoder_config["filter_dims"])
            self._num_filters = tuple(self._encoder_config["num_filters"])
            self._strides = tuple(self._encoder_config["strides"])
            self.encoder = nn.Sequential(*list(self.cnn_module_generator()))
            output_dim_conv = self._get_conv_output(self._input_shape)
            if "dense_sizes" in self._encoder_config:
                self.embedding_size = self._encoder_config["dense_sizes"][-1]
                fc_module = MLPModule(input_dim=output_dim_conv,
                                      output_dim=self.embedding_size,
                                      hidden_sizes=list(self._encoder_config["dense_sizes"][:-1]),
                                      hidden_nonlinearity=self._hidden_nonlinearity,
                                      hidden_w_init=self._hidden_w_init,
                                      hidden_b_init=self._hidden_b_init,
                                      output_nonlinearity=self._output_nonlinearity,
                                      output_w_init=self._output_w_init,
                                      output_b_init=self._output_b_init,
                                      layer_normalization=self._layer_norm,
                                      output_normalization=True)
                self.encoder.add_module("flatten", nn.Flatten())
                self.encoder.add_module("fc", fc_module)
            else:
                self.embedding_size = output_dim_conv
        elif "dense_sizes" in self._encoder_config:
            if self._encoder_config["dense_sizes"]:
                self.embedding_size = self._encoder_config["dense_sizes"][-1]
                self.encoder = MLPModule(
                    input_dim=self._input_shape[0],
                    output_dim=self.embedding_size,
                    hidden_sizes=list(self._encoder_config["dense_sizes"][:-1]),
                    hidden_nonlinearity=self._hidden_nonlinearity,
                    hidden_w_init=self._hidden_w_init,
                    hidden_b_init=self._hidden_b_init,
                    output_nonlinearity=self._output_nonlinearity,
                    output_w_init=self._output_w_init,
                    output_b_init=self._output_b_init,
                    layer_normalization=self._layer_norm,
                    output_normalization=self._layer_norm
                )
            else:
                self.embedding_size = self._input_shape[0]
                self.encoder = nn.Sequential()

        # Init Mlp
        self.head = MLPModule(input_dim=self.embedding_size,
                              output_dim=action_dim,
                              hidden_sizes=list(tuple(self._head_config["dense_sizes"])),
                              hidden_nonlinearity=self._hidden_nonlinearity,
                              hidden_w_init=self._hidden_w_init,
                              hidden_b_init=self._hidden_b_init,
                              output_nonlinearity=self._output_nonlinearity,
                              output_w_init=self._output_w_init,
                              output_b_init=self._output_b_init,
                              layer_normalization=self._layer_norm,
                              output_normalization=False)
        logger.log(f"Encoder is {self.encoder}")
        logger.log(f"Head is {self.head}")


    # Infer shape of tensor passed to mlp
    def _get_conv_output(self, shape):
        self.eval()
        with torch.no_grad():
            input = torch.autograd.Variable(torch.rand(1, *shape))
            output_feat = self.encoder(input)
            n_size = output_feat.data.view(1, -1).size(1)
        self.train()
        return n_size

    # pylint: disable=arguments-differ
    def forward(self, x, return_embedding=False):
        if not torch.is_tensor(x):
            x = torch.FloatTensor(x)
        x = x.to(device)
        if len(x.size()) == 4:
            x = x.permute(0, 3, 2, 1)
        embedding = self.encoder(x)
        x = embedding.view(embedding.size(0), -1)
        preds = self.head(x)
        if return_embedding:
            return preds, embedding
        return preds

    def cnn_module_generator(self):
        for input_dim, output_dim, filter_dim, stride in zip(
                (self.obs_dim[-1],) + self._num_filters, self._num_filters, self._filter_dims, self._strides):
            conv_layer = nn.Conv2d(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=filter_dim,
                stride=stride,
                padding=filter_dim // 2  # Maintain spatial resolution if stride 1
            )
            if self._layer_norm == "batch":
                yield nn.BatchNorm2d(input_dim)
            elif self._layer_norm == "layer":
                yield nn.LayerNorm(input_dim)
            self._hidden_w_init(conv_layer.weight)
            self._hidden_b_init(conv_layer.bias)
            yield conv_layer
            yield self._cnn_hidden_nonlinearity()

    def clone(self):
        return self.__class__(env_spec=self._env_spec,
                              encoder=self._encoder_config,
                              head=self._head_config,
                              hidden_nonlinearity=self._hidden_nonlinearity,
                              hidden_w_init=self._hidden_w_init,
                              hidden_b_init=self._hidden_b_init,
                              output_nonlinearity=self._output_nonlinearity,
                              output_w_init=self._output_w_init,
                              output_b_init=self._output_b_init,
                              layer_norm=self._layer_norm)
