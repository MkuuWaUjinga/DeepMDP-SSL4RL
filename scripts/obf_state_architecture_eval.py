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
        self.pool = torch.nn.MaxPool1d(8)
        #self.lin = torch.nn.Linear(5, 3)
        self.lin_end = torch.nn.Sequential(
            torch.nn.LayerNorm(8*5),
            torch.nn.Linear(8*5, 64),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 3)
        )

        self.conv.apply(self._init_weights)
        self.lin_end.apply(self._init_weights)


    def forward(self, x):
        map = self.conv(x)
        last_obs = x.flatten(1)[:, num_stacked_obs-1::num_stacked_obs]
        #skip = self.input_projection(last_obs)
        #pooled = self.pool(map.permute(0, 2, 1)).squeeze()
        #pooled = self.lin(pooled)
        #down_project = self.lin(pooled)
        #catted = torch.cat((down_project, last_obs), dim=1)
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
    num_epochs = 200
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
#sample_transitions(500000, num_stacked_obs)

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
    layer_normalization=False,
    output_normalization=False
)

# Init regressors
fc_regressor_32 = MLPModule(
    input_dim=num_stacked_obs * num_unobf_states,
    output_dim=8,
    hidden_sizes=[32],
    hidden_nonlinearity=torch.nn.ReLU,
    hidden_w_init=torch.nn.init.xavier_normal_,
    hidden_b_init=torch.nn.init.zeros_,
    output_nonlinearity=None,
    output_w_init=torch.nn.init.xavier_normal_,
    output_b_init=torch.nn.init.zeros_,
    layer_normalization=False,
    output_normalization=False
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
    layer_normalization=False,
    output_normalization=False
)

conv_regressor = torch.nn.Sequential(
    torch.nn.Conv1d(
                in_channels=num_stacked_obs,
                out_channels=16,
                kernel_size=1,
                stride=1
            ),
    torch.nn.ReLU(),
    torch.nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=1,
                stride=1
            ),
    torch.nn.AvgPool1d(1),
    torch.nn.Flatten(),
    torch.nn.Linear(32 * num_unobf_states, 8)
)

conv_regressor = ConvEncoderModel()

model_map = {
    "fc64": fc_regressor_64,
    "fc128_128": fc_regressor_128_128,
    "fc64_64": fc_regressor_64_64,
    #"fc32": fc_regressor_32,
    #"conv": conv_regressor,
}

viz = Visdom(port=9098)
plotter = VisdomLinePlotter(viz, env_name="arch")

for model_name in model_map:
   train(model_name)

#train(fc_regressor)

### Validation on obfuscated states
#Epoch 8 | Val Loss 0.0360698327422142
#Epoch 9 | Batch 0/9375 | Train Loss 0.016800491139292717
#Epoch 9 | Batch 1000/9375 | Train Loss 0.0321357436478138
#Epoch 9 | Batch 2000/9375 | Train Loss 0.03152284026145935
#Epoch 9 | Batch 3000/9375 | Train Loss 0.031437065452337265
#Epoch 9 | Batch 4000/9375 | Train Loss 0.03195047751069069
#Epoch 9 | Batch 5000/9375 | Train Loss 0.031601566821336746
#Epoch 9 | Batch 6000/9375 | Train Loss 0.031458258628845215
#Epoch 9 | Batch 7000/9375 | Train Loss 0.03130017966032028
#Epoch 9 | Batch 8000/9375 | Train Loss 0.031353551894426346
#Epoch 9 | Batch 9000/9375 | Train Loss 0.0313989631831646
#Epoch 9 | Val Loss 0.03387722000479698

#Epoch 48 | Val Loss 0.01563306525349617
#Epoch 49 | Batch 0/9375 | Train Loss 0.0006666170083917677
#Epoch 49 | Batch 1000/9375 | Train Loss 0.014545459300279617
#Epoch 49 | Batch 2000/9375 | Train Loss 0.01447675097733736
#Epoch 49 | Batch 3000/9375 | Train Loss 0.01461124513298273
#Epoch 49 | Batch 4000/9375 | Train Loss 0.014168239198625088
#Epoch 49 | Batch 5000/9375 | Train Loss 0.013844992965459824
#Epoch 49 | Batch 6000/9375 | Train Loss 0.013732480816543102
#Epoch 49 | Batch 7000/9375 | Train Loss 0.013524424284696579
#Epoch 49 | Batch 8000/9375 | Train Loss 0.01342287939041853
#Epoch 49 | Batch 9000/9375 | Train Loss 0.013385859318077564
#Epoch 49 | Val Loss 0.015088058076798916

### Validation calculated on all states
#Epoch 8 | Val Loss 0.014309137128293514
#Epoch 9 | Batch 0/9375 | Train Loss 0.003871660679578781
#Epoch 9 | Batch 1000/9375 | Train Loss 0.012485149316489697
#Epoch 9 | Batch 2000/9375 | Train Loss 0.012422861531376839
#Epoch 9 | Batch 3000/9375 | Train Loss 0.01215127483010292
#Epoch 9 | Batch 4000/9375 | Train Loss 0.012166060507297516
#Epoch 9 | Batch 5000/9375 | Train Loss 0.01232787687331438
#Epoch 9 | Batch 6000/9375 | Train Loss 0.012387661263346672
#Epoch 9 | Batch 7000/9375 | Train Loss 0.012286301702260971
#Epoch 9 | Batch 8000/9375 | Train Loss 0.012248095124959946
#Epoch 9 | Batch 9000/9375 | Train Loss 0.012354159727692604
#Epoch 9 | Val Loss 0.013362232595682144


# 8 stacked
#Epoch 8 | Val Loss 0.02951284684240818
#Epoch 9 | Batch 0/9375 | Train Loss 0.031409282237291336
#Epoch 9 | Batch 1000/9375 | Train Loss 0.027959300205111504
#Epoch 9 | Batch 2000/9375 | Train Loss 0.028334351256489754
#Epoch 9 | Batch 3000/9375 | Train Loss 0.027872402220964432
#Epoch 9 | Batch 4000/9375 | Train Loss 0.028243236243724823
#Epoch 9 | Batch 5000/9375 | Train Loss 0.028112251311540604
#Epoch 9 | Batch 6000/9375 | Train Loss 0.028136756271123886
#Epoch 9 | Batch 7000/9375 | Train Loss 0.02841773070394993
#Epoch 9 | Batch 8000/9375 | Train Loss 0.028061706572771072
#Epoch 9 | Batch 9000/9375 | Train Loss 0.027981402352452278
#Epoch 9 | Val Loss 0.02856108918786049

# Only obf states
#Epoch 8 | Val Loss 0.03807075321674347
#Epoch 9 | Batch 0/9375 | Train Loss 0.055063292384147644
#Epoch 9 | Batch 1000/9375 | Train Loss 0.03324313834309578
#Epoch 9 | Batch 2000/9375 | Train Loss 0.032573021948337555
#Epoch 9 | Batch 3000/9375 | Train Loss 0.03292786329984665
#Epoch 9 | Batch 4000/9375 | Train Loss 0.032870978116989136
#Epoch 9 | Batch 5000/9375 | Train Loss 0.033228859305381775
#Epoch 9 | Batch 6000/9375 | Train Loss 0.033142901957035065
#Epoch 9 | Batch 7000/9375 | Train Loss 0.03274071589112282
#Epoch 9 | Batch 8000/9375 | Train Loss 0.032672226428985596
#Epoch 9 | Batch 9000/9375 | Train Loss 0.03245869651436806
#Epoch 9 | Val Loss 0.03747032582759857

#Epoch 47 | Val Loss 0.02110835164785385
#Epoch 48 | Batch 0/9375 | Train Loss 0.003424539230763912
#Epoch 48 | Batch 1000/9375 | Train Loss 0.016024695709347725
#Epoch 48 | Batch 2000/9375 | Train Loss 0.017232704907655716
#Epoch 48 | Batch 3000/9375 | Train Loss 0.018149154260754585
#Epoch 48 | Batch 4000/9375 | Train Loss 0.018040286377072334
#Epoch 48 | Batch 5000/9375 | Train Loss 0.0181367639452219
#Epoch 48 | Batch 6000/9375 | Train Loss 0.018189474940299988
#Epoch 48 | Batch 7000/9375 | Train Loss 0.018370771780610085
#Epoch 48 | Batch 8000/9375 | Train Loss 0.01825123466551304
#Epoch 48 | Batch 9000/9375 | Train Loss 0.018198121339082718
#Epoch 48 | Val Loss 0.021042438223958015
#Epoch 49 | Batch 0/9375 | Train Loss 0.00415407307446003
#Epoch 49 | Batch 1000/9375 | Train Loss 0.01748443953692913
#Epoch 49 | Batch 2000/9375 | Train Loss 0.01810728758573532
#Epoch 49 | Batch 3000/9375 | Train Loss 0.01810125820338726
#Epoch 49 | Batch 4000/9375 | Train Loss 0.01812721975147724
#Epoch 49 | Batch 5000/9375 | Train Loss 0.01800587959587574
#Epoch 49 | Batch 6000/9375 | Train Loss 0.017994295805692673
#Epoch 49 | Batch 7000/9375 | Train Loss 0.017940586432814598
#Epoch 49 | Batch 8000/9375 | Train Loss 0.017944417893886566
#Epoch 49 | Batch 9000/9375 | Train Loss 0.01823481358587742
#Epoch 49 | Val Loss 0.02134205587208271

train(conv_regressor, reshape=True)


### Validation calculated on obfuscated states
#Epoch 8 | Val Loss 0.04829911142587662
#Epoch 9 | Batch 0/9375 | Train Loss 0.005763242021203041
#Epoch 9 | Batch 1000/9375 | Train Loss 0.014692454598844051
#Epoch 9 | Batch 2000/9375 | Train Loss 0.014724483713507652
#Epoch 9 | Batch 3000/9375 | Train Loss 0.014491976238787174
#Epoch 9 | Batch 4000/9375 | Train Loss 0.014537939801812172
#Epoch 9 | Batch 5000/9375 | Train Loss 0.014742338098585606
#Epoch 9 | Batch 6000/9375 | Train Loss 0.014783190563321114
#Epoch 9 | Batch 7000/9375 | Train Loss 0.014657055027782917
#Epoch 9 | Batch 8000/9375 | Train Loss 0.014598880894482136
#Epoch 9 | Batch 9000/9375 | Train Loss 0.014643541537225246
#Epoch 9 | Val Loss 0.04489811509847641


### Validation calculated on all states
#Epoch 8 | Val Loss 0.02006567269563675
#Epoch 9 | Batch 0/9375 | Train Loss 0.0059362188912928104
#Epoch 9 | Batch 1000/9375 | Train Loss 0.015959981828927994
#Epoch 9 | Batch 2000/9375 | Train Loss 0.015905195847153664
#Epoch 9 | Batch 3000/9375 | Train Loss 0.016002139076590538
#Epoch 9 | Batch 4000/9375 | Train Loss 0.01622300036251545
#Epoch 9 | Batch 5000/9375 | Train Loss 0.016144510358572006
#Epoch 9 | Batch 6000/9375 | Train Loss 0.016191063448786736
#Epoch 9 | Batch 7000/9375 | Train Loss 0.016048945486545563
#Epoch 9 | Batch 8000/9375 | Train Loss 0.015934886410832405
#Epoch 9 | Batch 9000/9375 | Train Loss 0.015953587368130684
#Epoch 9 | Val Loss 0.01926545798778534
