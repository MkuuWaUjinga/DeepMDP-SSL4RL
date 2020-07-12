import torch
import random
from visdom import Visdom
from deepmdp.experiments.utils import VisdomLinePlotter
import pickle
import numpy as np
from deepmdp.experiments.dqn_baseline import setup_lunar_lander_with_obfuscated_states
from deepmdp.garage_mod.modules.mlp_module import MLPModule
from evaluate_experiment import quantiles


def normalize(vector):
    tiled_1th_quantile = quantiles[:, 0]
    tiled_99th_quantile = quantiles[:, 1]
    reshaped_normalized = (vector - tiled_1th_quantile) / (tiled_99th_quantile - tiled_1th_quantile)
    #assert np.min(reshaped_normalized) >= 0 and np.max(reshaped_normalized) <= 1
    return reshaped_normalized

def project_back(vector):
    tiled_1th_quantile = quantiles[:, 0]
    tiled_99th_quantile = quantiles[:, 1]
    reshaped_normalized = vector * (tiled_99th_quantile - tiled_1th_quantile) + tiled_1th_quantile
    return reshaped_normalized

def project_back_old(vector):
    tiled_1th_quantile = np.tile(quantiles[:, 0], (num_stacked_obs, 1))
    tiled_99th_quantile = np.tile(quantiles[:, 1], (num_stacked_obs, 1))
    reshaped_normalized = vector.reshape(num_unobf_states, num_stacked_obs).transpose() * (tiled_99th_quantile - tiled_1th_quantile) + tiled_1th_quantile
    return reshaped_normalized.transpose().flatten()

class StateReconstructionDataSet:
    def __init__(self, stage="train", reshape=False, only_obf=False, noise=True):
        with open(f"data_{num_stacked_obs}_50000.pkl", "rb") as f:
            data = pickle.load(f)
        if reshape:
            self.x = np.array(data["x_" + stage]).reshape(-1, num_stacked_obs, num_unobf_states)
        else:
            self.x = np.array(data["x_" + stage])
        self.y = np.array(data["y_" + stage])
        self.only_obf = only_obf
        self.noise = noise

    def __getitem__(self, i):
        y = self.y[i]
        if self.noise:
            y = self.y[i]
            normalized = normalize(y)
            normalized += np.random.normal(loc=0, scale=0.1, size=8) # np.random.uniform(-0.1, 0.1, size=8) #
            y = project_back(normalized)

        y_ordered = np.array([], dtype=np.float32)
        y_ordered = np.append(y_ordered, y[[2, 3, 5]])

        if not self.only_obf:
            y_ordered = np.append(y_ordered,y[[0, 1, 4, 6, 7]])

        return self.x[i], y_ordered

    def __len__(self):
        return len(self.x)

class ConvEncoderModel(torch.nn.Module):

    def __init__(self):
        super(ConvEncoderModel, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.LayerNorm([num_stacked_obs, 5]),
            torch.nn.Conv1d(
                        in_channels=num_stacked_obs,
                        out_channels=8,
                        kernel_size=1,
                        stride=1
                    ),
            torch.nn.ReLU(),
            torch.nn.LayerNorm([8, 5]),
            torch.nn.Conv1d(
                        in_channels=8,
                        out_channels=8,
                        kernel_size=1,
                        stride=1
                    ),
            torch.nn.Flatten()
       )
        self.lin_end = torch.nn.Sequential(
            torch.nn.LayerNorm(8*5),
            torch.nn.Linear(8*5, 64),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 8)
        )

        self.conv.apply(self._init_weights)
        self.lin_end.apply(self._init_weights)


    def forward(self, x):
        map = self.conv(x)
        return self.lin_end(map)

    @staticmethod
    def _init_weights(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)
        elif classname.find('Batchnorm') != -1:
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

def sample_transitions(k, num_stacked_obs):
    # Init obf state env
    env = setup_lunar_lander_with_obfuscated_states("LunarLander-v2",
                                                    num_stacked_obs,
                                                    do_noops=True)

    # Set env seeds
    env.seed(32)
    env.action_space.seed(32)

    # Sample k transitions and store in data set
    x = []
    y = []
    done = True
    while len(x) < k:
        if done:
            env.reset()
        act = env.action_space.sample()
        obs, _, done, env_info = env.step(act)
        x.append(obs)
        y.append(env_info["ground_truth_state"])

    with open(f"data_{num_stacked_obs}.pkl", "wb") as f:
       pickle.dump({
           "x_train": x[0:int(0.6*k)],
           "y_train": y[0:int(0.6*k)],
           "x_test": x[int(0.6*k):int(0.8*k)],
           "y_test": y[int(0.6*k):int(0.8*k)],
           "x_val": x[int(0.8*k):],
           "y_val": y[int(0.8*k):],
       }, f)

def train(model_name, reshape=False):
    model = model_map[model_name]
    print(model)
    batch_size = 32
    num_epochs = 1000
    train_print_progress_every = 500
    evaluate_only_obf_states = False
    print(f"Evaluating only obfuscated states {evaluate_only_obf_states}")
    print("Num params in model {}".format(sum(par.numel() for par in model.parameters() if par.requires_grad)))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if "conv" in model_name:
        reshape = True
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)

    # Load data
    train_data = StateReconstructionDataSet(reshape=reshape, only_obf=evaluate_only_obf_states)
    val_data = StateReconstructionDataSet(stage="val", reshape=reshape, only_obf=evaluate_only_obf_states)

    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    loss_func = torch.nn.MSELoss()
    for epoch_number in range(num_epochs):
        model.train()
        cummulated_loss = 0
        for i, (x, y) in enumerate(train_data_loader):
            optimizer.zero_grad()
            preds = model(x)
            loss = loss_func(preds, y)
            loss.backward()
            cummulated_loss += loss
            optimizer.step()
            if i % train_print_progress_every == 0:
                print('Epoch {} | Batch {}/{} | Train Loss {}'.format(epoch_number, i, len(train_data_loader), cummulated_loss/float(i+1)))

        # Validation eval
        model.eval()
        avg_val_loss = 0
        for i, (x, y) in enumerate(val_data_loader):
            with torch.no_grad():
                preds = model(x)
                avg_val_loss +=  loss_func(preds, y)
        plotter.plot("val l2 loss", model_name, "Validation loss of tested architectures", epoch_number, avg_val_loss/len(val_data_loader),
                     color=np.array([[r, g, b],]))
        print('Epoch {} | Val Loss {}'.format(epoch_number, avg_val_loss/len(val_data_loader)))


num_stacked_obs = 4
num_unobf_states = 5
torch.manual_seed(32)
np.random.seed(32)
sample_transitions(50000, num_stacked_obs)

# Init regressors
fc_regressor_64 = MLPModule(
    input_dim=num_stacked_obs * num_unobf_states,
    output_dim=8,
    hidden_sizes=[64],
    hidden_nonlinearity=torch.nn.ReLU,
    hidden_w_init=torch.nn.init.xavier_normal_,
    hidden_b_init=torch.nn.init.zeros_,
    output_nonlinearity=None,
    output_w_init=torch.nn.init.xavier_normal_,
    output_b_init=torch.nn.init.zeros_,
    layer_normalization="batch",
    output_normalization=True
)

# Init regressors
fc_regressor_64_64 = MLPModule(
    input_dim=num_stacked_obs * num_unobf_states,
    output_dim=8,
    hidden_sizes=[64, 64],
    hidden_nonlinearity=torch.nn.ReLU,
    hidden_w_init=torch.nn.init.xavier_normal_,
    hidden_b_init=torch.nn.init.zeros_,
    output_nonlinearity=None,
    output_w_init=torch.nn.init.xavier_normal_,
    output_b_init=torch.nn.init.zeros_,
    layer_normalization=False,
    output_normalization=False
)

fc_regressor_128_128 = MLPModule(
    input_dim=num_stacked_obs * num_unobf_states,
    output_dim=8,
    hidden_sizes=[128, 128],
    hidden_nonlinearity=torch.nn.ReLU,
    hidden_w_init=torch.nn.init.xavier_normal_,
    hidden_b_init=torch.nn.init.zeros_,
    output_nonlinearity=None,
    output_w_init=torch.nn.init.xavier_normal_,
    output_b_init=torch.nn.init.zeros_,
    layer_normalization="batch",
    output_normalization=True
)

conv_regressor = ConvEncoderModel()

model_map = {
    "fc64": fc_regressor_64,
    "fc128_128": fc_regressor_128_128,
    "fc64_64": fc_regressor_64_64,
    "conv": conv_regressor,
}

viz = Visdom(port=9098)
plotter = VisdomLinePlotter(viz, env_name="arch")

for model_name in model_map:
   train(model_name)
