import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from DRRL.attention_module import AttentionModule
from collections import OrderedDict

from helpers.a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from helpers.a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError
        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        # self.main = nn.Sequential(
        #     init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
        #     init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
        #     init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
        #     init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())
        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 16, 2, stride=1)), nn.ReLU(),
            init_(nn.Conv2d(16, 16, 2, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(16*5*5, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class DRRLBase(NNBase):
    """Adaptation of DRRLnet class more easily usable with pytorch-a2c-ppo-acktr-gail implementation"""
    def __init__(self, num_inputs, recurrent=False, hidden_size=512, w=12, h=12, n_f_conv1 = 12, n_f_conv2 = 24, pad=True,
                 att_emb_size=64, n_heads=2, n_att_stack=2, n_fc_layers=4,
                 baseline_mode=False, n_baseMods=3):
        """
        Args:
            num_inputs: num input channels (usually 3 RGB channels)
            recurrent: Not implemented, hidden state can be put in and are returned
            hidden_size: Size of output layer of base, i.e. before action layer projects into action space inside Policy

            w: width of input image (including black boundary)
            h: hight of input image (including black boundary)
            n_f_conv1: #filters in first conv layer
            n_f_conv2: #filters in second conv layer
            pad: whether input images are padded through convolution layers so size is maintained

            att_emb_size: #attentional filters inside each head
            n_heads: #head in parallel inside attentional module
            n_att_stack: #times attentional module is stacked

            n_fc_layers: #fully connected output layers on top of attentional module

            baseline: use residual-convolutional baseline core instead of attentional module
            n_baseline: #residual-convolutional blocks inside baseline core"""

        if recurrent:
            raise NotImplementedError("Currently no recurrent version of DRRL architecture implemented.")
        #internal action replay buffer for simple training algorithms
        self.baseline_mode = baseline_mode
        self.saved_actions = []
        self.rewards = []

        self.pad = pad
        self.n_baseMods = n_baseMods
        super(DRRLBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.conv1 = init_(nn.Conv2d(3, n_f_conv1, kernel_size=2, stride=1))
        #possibly batch or layer norm, neither was mentioned in the paper though
        # self.ln1 = nn.LayerNorm([n_f_conv1,conv1w,conv1h])
        # self.bn1 = nn.BatchNorm2d(n_f_conv1)
        self.conv2 = init_(nn.Conv2d(n_f_conv1, n_f_conv2, kernel_size=2, stride=1))
        # self.ln2 = nn.LayerNorm([n_f_conv2,conv2w,conv2h])
        # self.bn2 = nn.BatchNorm2d(n_f_conv2)

        # calculate size of convolution module output
        def conv2d_size_out(size, kernel_size=2, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        if self.pad:
            conv1w = conv2w = w
            conv1h = conv2h = h
        else:
            conv1w = conv2d_size_out(w)
            conv1h = conv2d_size_out(h)
            conv2w = conv2d_size_out(conv1w)
            conv2h = conv2d_size_out(conv1h)

        # create x,y coordinate matrices to append to convolution output
        xmap = np.linspace(-np.ones(conv2h), np.ones(conv2h), num=conv2w, endpoint=True, axis=0)
        xmap = torch.tensor(np.expand_dims(np.expand_dims(xmap,0),0), dtype=torch.float32, requires_grad=False)
        ymap = np.linspace(-np.ones(conv2w), np.ones(conv2w), num=conv2h, endpoint=True, axis=1)
        ymap = torch.tensor(np.expand_dims(np.expand_dims(ymap,0),0), dtype=torch.float32, requires_grad=False)
        self.register_buffer("xymap", torch.cat((xmap,ymap),dim=1)) # shape (1, 2, conv2w, conv2h)

        # an "attendable" entity has 24 CNN channels + 2 coordinate channels = 26 features. this is also the default
        # number of baseline module conv layer filter number
        att_elem_size = n_f_conv2 + 2
        if not self.baseline_mode:
            # create attention module with n_heads heads and remember how many times to stack it
            self.n_att_stack = n_att_stack #how many times the attentional module is to be stacked (weight-sharing -> reuse)
            self.attMod = AttentionModule(conv2w*conv2h, att_elem_size, att_emb_size, n_heads)
        else:            # create baseline module of several residual-convolutional layers
            base_dict = {}
            for i in range(self.n_baseMods):
                base_dict[f"baseline_identity_{i}"] = nn.Identity()
                base_dict[f"baseline_conv_{i}_0"] = init_(nn.Conv2d(att_elem_size, att_elem_size, kernel_size=3,
                                                                   stride=1))
                base_dict[f"baseline_batchnorm_{i}_0"] = nn.BatchNorm2d(att_elem_size)
                base_dict[f"baseline_conv_{i}_1"] = init_(nn.Conv2d(att_elem_size, att_elem_size, kernel_size=3,
                                                                   stride=1))
                base_dict[f"baseline_batchnorm_{i}_1"] = nn.BatchNorm2d(att_elem_size)

            self.baseMod = nn.ModuleDict(base_dict)
        #max pooling
        # print(f"attnl element size:{att_elem_size}")
        # self.maxpool = nn.MaxPool1d(kernel_size=att_emb_size,return_indices=False) #don't know why maxpool reduces
        # kernel_size by 1

        # FC256 layers, 4 is default
        if n_fc_layers < 1:
            raise ValueError("At least 1 linear readout layer is required.")
        fc_dict = OrderedDict([('fc1', nn.Linear(att_elem_size, hidden_size)),
                               ('relu1', nn.ReLU())]) #first one has different inpuz size
        for i in range(n_fc_layers-1):
            fc_dict[f"fc{i+2}"] = nn.Linear(hidden_size, hidden_size)
            fc_dict[f"relu{i+2}"] = nn.ReLU()
        self.fc_seq = nn.Sequential(fc_dict) #sequential container from ordered dict

        self.critic_linear = nn.Linear(hidden_size, 1)

        self.train()

    def forward(self, inputs, rnn_hxs=None, masks=None):
        """hidden states rnn_hxs and masks are not implemented because there currently is no recurrent version of the
        attentional architecture.
        """
        x = inputs / 255.0

        #convolutional module
        if self.pad:
            x = F.pad(x, (1,0,1,0)) #zero padding so state size stays constant
        c = F.relu(self.conv1(x))
        if self.pad:
            c = F.pad(c, (1,0,1,0))
        c = F.relu(self.conv2(c))
        #append x,y coordinates to every sample in batch
        batchsize = c.size(0)
        # Filewriter complains about the this way of repeating the xymap, hope repeat is just as fine
        # batch_maps = torch.cat(batchsize*[self.xymap])
        batch_maps = self.xymap.repeat(batchsize,1,1,1,)
        c = torch.cat((c,batch_maps),1)
        if not self.baseline_mode:
            #attentional module
            #careful: we are flattening out x,y dimensions into 1 dimension, so shape changes from (batchsize, #filters,
            # #conv2w, conv2h) to (batchsize, conv2w*conv2h, #filters), because downstream linear layers take last
            # dimension to be input features
            a = c.view(c.size(0),c.size(1), -1).transpose(1,2)
            # n_att_mod passes through attentional module -> n_att_mod stacked modules with weight sharing
            for i_att in range(self.n_att_stack):
                a = self.attMod(a)
        else:
            #baseline module
            for i in range(self.n_baseMods):
                inp = self.baseMod[f"baseline_identity_{i}"](c)         #save input for residual connection
                #todo: make padding adaptive to kernel size and stride
                c = F.pad(c, (1, 1, 1, 1))                              #padding so input maintains size
                c = self.baseMod[f"baseline_conv_{i}_0"](c)             #conv1
                c = self.baseMod[f"baseline_batchnorm_{i}_0"](c)        #batch-norm
                c = F.relu(c)                                           #relu
                c = F.pad(c, (1, 1, 1, 1))                              #padding so input maintains size
                c = self.baseMod[f"baseline_conv_{i}_1"](c)             #conv2
                c = c + inp                                             #residual connecton
                c = self.baseMod[f"baseline_batchnorm_{i}_1"](c)        #batch-norm
                c = F.relu(c)                                           #relu
            a = c.view(c.size(0),c.size(1), -1).transpose(1,2)          #flatten (transpose not necessary but we do
                                                                        # it for consistency w/ attentional module

        #max pooling over "space", i.e. max scalar within each feature map m x n x f -> f
        # pool over entity dimension #isn't this a problem with gradients?
        # todo: try pooling over feature dimension
        kernelsize = a.shape[1] #but during forward passes called by SummaryWriter, a.shape[1] returns a tensor instead
        # of an int. if this causes any trouble it can be replaced by w*h
        if type(kernelsize) == torch.Tensor:
            kernelsize = kernelsize.item()
        pooled = F.max_pool1d(a.transpose(1,2), kernel_size=kernelsize) #pool out entity dimension
        #policy module: 4xFC256, then project to logits and value
        p = self.fc_seq(pooled.view(pooled.size(0),pooled.size(1)))

        return self.critic_linear(p), p, rnn_hxs

    def get_attention_weights(self, inputs, rnn_hxs=None, masks=None):
        """ Forward pass through the architecture but only to the point where attention weights are calculated.
        Identical up to that point to forward()
        """

        if self.baseline_mode:
            raise Exception("Baseline mode set to True. No attention.")
        x = inputs / 255.0
        #convolutional module
        if self.pad:
            x = F.pad(x, (1,0,1,0)) #zero padding so state size stays constant
        c = F.relu(self.conv1(x))
        if self.pad:
            c = F.pad(c, (1,0,1,0))
        c = F.relu(self.conv2(c))
        #append x,y coordinates to every sample in batch
        batchsize = c.size(0)
        # Filewriter complains about the this way of repeating the xymap, hope repeat is just as fine
        # batch_maps = torch.cat(batchsize*[self.xymap])
        batch_maps = self.xymap.repeat(batchsize,1,1,1,)
        c = torch.cat((c,batch_maps),1)
        #attentional module
        #careful: we are flattening out x,y dimensions into 1 dimension, so shape changes from (batchsize, #filters,
        # #conv2w, conv2h) to (batchsize, conv2w*conv2h, #filters), because downstream linear layers take last
        # dimension to be input features
        a = c.view(c.size(0),c.size(1), -1).transpose(1,2)
        # n_att_mod passes through attentional module -> n_att_mod stacked modules with weight sharing
        att_weights = []
        for i_att in range(self.n_att_stack):
            a, weights = self.attMod.get_att_weights(a)
            att_weights.append(weights)
        return att_weights

    # def get_body_output(self, x):
    #     #convolutional module
    #     if self.pad:
    #         x = F.pad(x, (1,0,1,0)) #zero padding so state size stays constant
    #     c = F.relu(self.conv1(x))
    #     if self.pad:
    #         c = F.pad(c, (1,0,1,0))
    #     c = F.relu(self.conv2(c))
    #     #append x,y coordinates to every sample in batch
    #     batchsize = c.size(0)
    #     # Filewriter complains about the this way of repeating the xymap, hope repeat is just as fine
    #     # batch_maps = torch.cat(batchsize*[self.xymap])
    #     batch_maps = self.xymap.repeat(batchsize,1,1,1,)
    #     c = torch.cat((c,batch_maps),1)
    #     if not self.baseline_mode:
    #         #attentional module
    #         #careful: we are flattening out x,y dimensions into 1 dimension, so shape changes from (batchsize, #filters,
    #         # #conv2w, conv2h) to (batchsize, conv2w*conv2h, #filters), because downstream linear layers take last
    #         # dimension to be input features
    #         a = c.view(c.size(0),c.size(1), -1).transpose(1,2)
    #         # n_att_mod passes through attentional module -> n_att_mod stacked modules with weight sharing
    #         for i_att in range(self.n_att_stack):
    #             a = self.attMod(a)
    #     else:
    #         #baseline module
    #         for i in range(self.n_baseMods):
    #             inp = self.baseMod[f"baseline_identity_{i}"](c)         #save input for residual connection
    #             #todo: make padding adaptive to kernel size and stride
    #             c = F.pad(c, (1, 1, 1, 1))                              #padding so input maintains size
    #             c = self.baseMod[f"baseline_conv_{i}_0"](c)             #conv1
    #             c = self.baseMod[f"baseline_batchnorm_{i}_0"](c)        #batch-norm
    #             c = F.relu(c)                                           #relu
    #             c = F.pad(c, (1, 1, 1, 1))                              #padding so input maintains size
    #             c = self.baseMod[f"baseline_conv_{i}_1"](c)             #conv2
    #             c = c + inp                                             #residual connecton
    #             c = self.baseMod[f"baseline_batchnorm_{i}_1"](c)        #batch-norm
    #             c = F.relu(c)                                           #relu
    #         a = c.view(c.size(0),c.size(1), -1).transpose(1,2)          #flatten (transpose not necessary but we do
    #                                                                     # it for consistency w/ attentional module
    #
    #     #max pooling over "space", i.e. max scalar within each feature map m x n x f -> f
    #     # pool over entity dimension #isn't this a problem with gradients?
    #     # todo: try pooling over feature dimension
    #     kernelsize = a.shape[1] #but during forward passes called by SummaryWriter, a.shape[1] returns a tensor instead
    #     # of an int. if this causes any trouble it can be replaced by w*h
    #     if type(kernelsize) == torch.Tensor:
    #         kernelsize = kernelsize.item()
    #     pooled = F.max_pool1d(a.transpose(1,2), kernel_size=kernelsize) #pool out entity dimension
    #     #policy module: 4xFC256, then project to logits and value
    #     p = self.fc_seq(pooled.view(pooled.size(0),pooled.size(1)))
    #     return p
    #
    # def predict(self, state):
    #     body_output = self.get_body_output(state)
    #     pi = F.softmax(self.logits(body_output), dim=1)
    #     return pi, self.value(body_output)
    #
    # def get_action(self, state):
    #     probs = self.predict(state)[0].detach().squeeze().numpy()
    #     action = np.random.choice(4, p=probs)
    #     return action
    #
    # def get_log_probs(self, state):
    #     body_output = self.get_body_output(state)
    #     logprobs = F.log_softmax(self.logits(body_output), dim=1)
    #     return logprobs
