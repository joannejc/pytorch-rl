import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Softmax_Policy(nn.Module):
    """A discrete softmax policy, updated via REINFORCE with a value
    function baseline
    """
    def __init__(self, hidden_shape, input_state_dim, action_dim, baseline=True):
        """initialize the softmax policy. Build the layers and initialize weights
        and biases. Default using Tanh nonlinearity, xavier_normal weight init and zero bias init.

        Args:
            hidden_shape - tuple representing the layers, i.e (30,20) for a two
                            hidden layer network
            input_state_dim - the input state dimension, or the state space dimension
            action_dim - the output dimension of the network, representing # of
                        discrete possible actions
            baseline - boolean indicating whether we use a parameterized baseline
        """
        super(Softmax_Policy,self).__init__()
        self._input_dim = input_state_dim
        self._output_dim = action_dim
        self._hidden_shape = hidden_shape
        self._network_shape = tuple([input_state_dim]+list(hidden_shape)+[action_dim])

        # change these if you'd like
        self._hidden_act = nn.Tanh()

        # build the layers and initialize weights
        for i in range(len(self._network_shape)-1):
            # set the layer weight
            setattr(self,'affine'+str(i), nn.Linear(self._network_shape[i],self._network_shape[i+1]))
            # initialze the weight
            nn.init.xavier_normal((getattr(self,'affine'+str(i)).weight))
            # initialize bias as zeros
            nn.init.constant((getattr(self,'affine'+str(i))).bias,0.0)

        # build baseline if needed, change hidden layer size below if needed, or if need be the # of layers below
        baseline_hidden_size = 10

        if baseline:
            self.baseline = nn.Sequential(nn.Linear(input_state_dim+1, baseline_hidden_size),
                                          nn.ReLU(),
                                          nn.Linear(baseline_hidden_size, action_dim))
        else:
            self.baseline = None

    def forward(self, x):
        """forward pass through our policy network, applying the nonlinear transform

        Args:
            x - input to our policy network, this will be the state observations
        """
        for k in range(len(self._network_shape)-1):
            x = self._hidden_act((getattr(self,'affine'+str(k)))(x))
            print(x)

    def get_action(self, state_obs):
        """Given a state observation, we utilize the forward function to compute our softmax action selection, as well
        as value from our value function baseline
        """
