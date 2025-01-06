import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from math import sqrt
from src.utils.AdamGnT import AdamGnT
import sys
from .algo_type import AlgoType

class GnT(object):
    """
    Generate-and-Test algorithm for LSTM-based networks, based on maturity-threshold based replacement
    """
    def __init__(
            self,
            net,
            hidden_activation,
            opt,
            decay_rate=0.99,
            replacement_rate=1e-4,
            init='kaiming',
            device="cpu",
            maturity_threshold=20,
            util_type='contribution',
            loss_func=F.mse_loss,
            accumulate=False,
    ):
        super(GnT, self).__init__()
        self.device = device
        self.net = net  # List of layers, LSTM and FC layers
        self.loss_func = loss_func
        self.accumulate = accumulate
        self.opt = opt
        self.opt_type = 'sgd' if not isinstance(self.opt, AdamGnT) else 'adam'
        self.replacement_rate = replacement_rate
        self.decay_rate = decay_rate
        self.maturity_threshold = maturity_threshold
        self.util_type = util_type

        # Set up utility arrays
        self.util = [torch.zeros(self.net[-1].in_features).to(self.device)]  # FC layer output size
        self.bias_corrected_util = [torch.zeros(self.net[-1].in_features).to(self.device)]
        self.ages = [torch.zeros(self.net[-1].in_features).to(self.device)]
        self.m = torch.nn.Softmax(dim=1)
        self.mean_feature_act = [torch.zeros(self.net[-1].in_features).to(self.device)]
        self.accumulated_num_features_to_replace = [0]

        # Compute bounds for initialization
        self.bounds = self.compute_bounds(hidden_activation=hidden_activation, init=init)

    def compute_bounds(self, hidden_activation, init='kaiming'):
        """Compute bounds for feature initialization"""
        bounds = [sqrt(3 / self.net[0].input_size)]  # LSTM input size for initialization
        return bounds

    def update_utility(self, layer_idx=0, features=None):
        with torch.no_grad():
            if isinstance(features, torch.Tensor):
                features = features.to(self.device)
            elif isinstance(features, list):  # Handle list of tensors
                features = torch.stack(features, dim=0).to(self.device)
            else:
                raise ValueError(f"Unsupported type for features: {type(features)}")

            # Perform utility update calculations
            self.util[layer_idx] *= self.decay_rate
            bias_correction = 1 - self.decay_rate ** self.ages[layer_idx]
            self.mean_feature_act[layer_idx] *= self.decay_rate

            # Calculate mean features over batch, sequence, and sub-sequence
            features_mean = features.mean(dim=(0, 1, 2))

            # Update mean feature activations
            self.mean_feature_act[layer_idx] -= (1 - self.decay_rate) * features_mean

            # Bias correction on activation
            bias_corrected_act = self.mean_feature_act[layer_idx] / bias_correction

            # Compute new utility based on features and layer weights
            output_weight_mag = self.net[-1].weight.data.abs().mean(dim=0)
            new_util = output_weight_mag * features.abs().mean(dim=(0, 1, 2)) if self.util_type == 'contribution' else output_weight_mag

            self.util[layer_idx] += (1 - self.decay_rate) * new_util
            self.bias_corrected_util[layer_idx] = self.util[layer_idx] / bias_correction

    def test_features(self, features):
        """Test and select features for replacement"""
        features_to_replace = [torch.empty(0, dtype=torch.long).to(self.device)]
        num_features_to_replace = [0]
        if self.replacement_rate == 0:
            return features_to_replace, num_features_to_replace
        self.ages[0] += 1
        self.update_utility(layer_idx=0, features=features)
        eligible_feature_indices = torch.where(self.ages[0] > self.maturity_threshold)[0]
        if eligible_feature_indices.shape[0] == 0:
            return features_to_replace, num_features_to_replace
        num_new_features_to_replace = self.replacement_rate * eligible_feature_indices.shape[0]
        num_new_features_to_replace = max(1, int(num_new_features_to_replace))

        new_features_to_replace = torch.topk(-self.bias_corrected_util[0][eligible_feature_indices], num_new_features_to_replace)[1]
        new_features_to_replace = eligible_feature_indices[new_features_to_replace]
        self.util[0][new_features_to_replace] = 0
        self.mean_feature_act[0][new_features_to_replace] = 0.0

        features_to_replace[0] = new_features_to_replace
        num_features_to_replace[0] = num_new_features_to_replace

        return features_to_replace, num_features_to_replace

    def gen_new_features(self, features_to_replace, num_features_to_replace):
        """Generate new features for replacement"""
        with torch.no_grad():
            for i in range(1):  # Only adjust for FC layer here
                if num_features_to_replace[i] == 0:
                    continue

                lstm_layer = self.net[0]  # LSTM layer
                fc_layer = self.net[-1]  # Fully connected layer

                # Update LSTM input-to-hidden weights
                for feature_idx in features_to_replace[i]:
                    lstm_layer.weight_ih_l0[feature_idx, :] *= 0.0
                    lstm_layer.weight_ih_l0[feature_idx, :] += torch.empty(
                        lstm_layer.input_size
                    ).uniform_(-self.bounds[i], self.bounds[i]).to(self.device)

                # Update LSTM hidden-to-hidden weights
                for feature_idx in features_to_replace[i]:
                    lstm_layer.weight_hh_l0[feature_idx, :] *= 0.0
                    lstm_layer.weight_hh_l0[feature_idx, :] += torch.empty(
                        lstm_layer.hidden_size
                    ).uniform_(-self.bounds[i], self.bounds[i]).to(self.device)

                # Update FC layer weights and biases
                fc_layer.bias.data += (
                    fc_layer.weight.data[:, features_to_replace[i]]
                    * self.mean_feature_act[i][features_to_replace[i]]
                ).sum(dim=1)
                fc_layer.weight.data[:, features_to_replace[i]] = 0

                # Reset ages for replaced features
                self.ages[i][features_to_replace[i]] = 0

    def update_optim_params(self, features_to_replace, num_features_to_replace):
      """Update optimizer's state after feature replacement"""
      if self.opt_type == 'adam':
          for i in range(1):  # Adjusted for LSTM to consider FC layer
              if num_features_to_replace[i] == 0:
                  continue

              lstm_layer = self.net[0]  # LSTM layer
              fc_layer = self.net[-1]  # Fully connected layer

              # Log the size of the LSTM and FC layers
              # print("LSTM Layer Weight Size: ", lstm_layer.weight_ih_l0.size())
              # print("Fully Connected Layer Weight Size: ", fc_layer.weight.size())

              # Reset optimizer states
              for param_name, param in [('weight_ih_l0', lstm_layer.weight_ih_l0), 
                                        ('weight_hh_l0', lstm_layer.weight_hh_l0), 
                                        ('bias_ih_l0', lstm_layer.bias_ih_l0), 
                                        ('bias_hh_l0', lstm_layer.bias_hh_l0)]:
                  if param not in self.opt.state:
                      continue

                  #print(f"Optimizer state for {param_name}: {self.opt.state[param]}")

                  # Reset exp_avg and exp_avg_sq for 1D tensors (if exp_avg and exp_avg_sq are 1D)
                  if 'exp_avg' in self.opt.state[param]:
                      exp_avg = self.opt.state[param]['exp_avg']
                      if len(exp_avg.shape) == 1:  # If it's a 1D tensor
                          exp_avg[features_to_replace[i]] = 0.0
                      elif len(exp_avg.shape) == 2:  # If it's a 2D tensor
                          exp_avg[features_to_replace[i], :] = 0.0
                          
                  if 'exp_avg_sq' in self.opt.state[param]:
                      exp_avg_sq = self.opt.state[param]['exp_avg_sq']
                      if len(exp_avg_sq.shape) == 1:  # If it's a 1D tensor
                          exp_avg_sq[features_to_replace[i]] = 0.0
                      elif len(exp_avg_sq.shape) == 2:  # If it's a 2D tensor
                          exp_avg_sq[features_to_replace[i], :] = 0.0

                  # Reset step for 1D tensor
                  if 'step' in self.opt.state[param]:
                      step = self.opt.state[param]['step']
                      if len(step.shape) == 1:  # If it's a 1D tensor
                          step[features_to_replace[i]] = 0
                      elif len(step.shape) == 2:  # If it's a 2D tensor
                          step[features_to_replace[i], :] = 0


    def gen_and_test(self, features):
        """Perform generate-and-test with LSTM model"""
        features_to_replace, num_features_to_replace = self.test_features(features=features)
        self.gen_new_features(features_to_replace, num_features_to_replace)
        self.update_optim_params(features_to_replace, num_features_to_replace)


class ContinualBackprop(AlgoType):
    def __init__(
        self,
        net,
        learning_rate=0.001,
        loss='nll',
        opt='adam',
        beta=0.9,
        beta_2=0.999,
        replacement_rate=0.001,
        decay_rate=0.9,
        device='cpu',
        maturity_threshold=100,
        util_type='contribution',
        init='kaiming',
        accumulate=False,
        momentum=0,
        outgoing_random=False,
        weight_decay=0,
    ):
        self.net = net  # The model is passed here
        self.device = device
        
        # Set up optimizer
        if opt == 'sgd':
            self.opt = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        elif opt == 'adam':
            self.opt = AdamGnT(self.net.parameters(), lr=learning_rate, betas=(beta, beta_2), weight_decay=weight_decay)

        # Choose the loss function
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[loss]

        # Extract GRU and FC layers for GnT
        gru_layers = [self.net.gru]  # GRU layer
        fc_layers = [self.net.fc]  # Fully connected layer

        # Initialize GnT with necessary parameters
        self.gnt = GnT(
            net=gru_layers + fc_layers,  # Include both GRU and FC layers
            hidden_activation=nn.ReLU,  # ReLU activation for FC layers
            opt=self.opt,
            replacement_rate=replacement_rate,
            decay_rate=decay_rate,
            maturity_threshold=maturity_threshold,
            util_type=util_type,
            device=device,
            loss_func=self.loss_func,
            init=init,
            accumulate=accumulate,
        )

    def learn(self, x, target):
        features = []  # Store activations from GRU and FC layers

        # Get batch size and sequence length from input tensor
        batch_size = x.size(0)
        sequence_length = x.size(1)  # Sequence length is the second dimension (x.size(1))
        input_size = self.net.input_size  # This is the vocabulary size

        # Reshape input for GRU: (batch_size, sequence_length)
        x = x.view(batch_size, sequence_length)

        # Forward pass through the GRU model
        # Ensure input is of type long before passing to embedding layer (this is handled by the GRU's embedding)
        embedded = self.net.embedding(x)  # Pass through the embedding layer
        embedded = embedded.float()  # Convert to float32 for the GRU

        # Forward pass through the GRU layer
        gru_out, _ = self.net.gru(embedded)  # This will use the embedding layer internally
        
        # Store GRU activations (from all time steps or the last time step)
        features.append(gru_out)  # Store activations from GRU
        
        # Use the output from the last time step (note that GRU is bidirectional)
        output = self.net.fc(gru_out[:, -1, :])  # Pass the last time step output through FC layer

        # Compute the loss
        loss = self.loss_func(output, target.long())

        # Perform standard gradient descent
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # Generate and test using GnT after the optimization step
        self.opt.zero_grad()  # Optional: Reset gradients
        self.gnt.gen_and_test(features=features)

        return loss, output

