import torch
import pickle
import numpy as np
from deepmdp.experiments.dqn_baseline import setup_lunar_lander_with_obfuscated_states
from deepmdp.garage_mod.modules.mlp_module import MLPModule

class StateReconstructionDataSet:
    def __init__(self, stage="train", reshape=False, only_obf=False):
        with open("data.pkl", "rb") as f:
            data = pickle.load(f)
        if reshape:
            self.x = np.array(data["x_" + stage]).reshape(-1, num_stacked_obs, num_unobf_states)
        else:
            self.x = np.array(data["x_" + stage])
        self.y = np.array(data["y_" + stage])
        self.only_obf = only_obf

    def __getitem__(self, i):
        y_ordered = np.array([], dtype=np.float32)
        y_ordered = np.append(y_ordered, self.y[i][[2, 3, 5]])
        if not self.only_obf:
            y_ordered = np.append(y_ordered, self.y[i][[0, 1, 4, 6, 7]])
        return self.x[i], y_ordered

    def __len__(self):
        return len(self.x)

class ConvEncoderModel(torch.nn.Module):

    def __init__(self):
        super(ConvEncoderModel, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                        in_channels=num_stacked_obs,
                        out_channels=8,
                        kernel_size=1,
                        stride=1
                    ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                        in_channels=8,
                        out_channels=8,
                        kernel_size=1,
                        stride=1
                    ),
       )
        self.pool = torch.nn.MaxPool1d(8)
        #self.lin = torch.nn.Linear(5, 3)
        self.lin_end = torch.nn.Sequential(
            torch.nn.Linear(5, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3)
        )

        self.conv.apply(self._init_weights)
        self.lin_end.apply(self._init_weights)


    def forward(self, x):
        map = self.conv(x)
        last_obs = x.flatten(1)[:, num_stacked_obs-1::num_stacked_obs]
        #skip = self.input_projection(last_obs)
        pooled = self.pool(map.permute(0, 2, 1)).squeeze()
        #pooled = self.lin(pooled)
        #down_project = self.lin(pooled)
        #catted = torch.cat((down_project, last_obs), dim=1)
        return self.lin_end(pooled)

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
                                                    number_of_stacked_frames=num_stacked_obs,
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

    with open("data.pkl", "wb") as f:
       pickle.dump({
           "x_train": x[0:int(0.6*k)],
           "y_train": y[0:int(0.6*k)],
           "x_test": x[int(0.6*k):int(0.8*k)],
           "y_test": y[int(0.6*k):int(0.8*k)],
           "x_val": x[int(0.8*k):],
           "y_val": y[int(0.8*k):],
       }, f)

def train(model, reshape=False):
    print(model)
    batch_size = 32
    num_epochs = 10
    train_print_progress_every = 1000
    evaluate_only_obf_states = True
    print(f"Evaluating only obfuscated states {evaluate_only_obf_states}")
    print("Num params in model {}".format(sum(par.numel() for par in model.parameters() if par.requires_grad)))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Load data
    train_data = StateReconstructionDataSet(reshape=reshape, only_obf=True)
    val_data = StateReconstructionDataSet(stage="val", reshape=reshape, only_obf=True)

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
                avg_val_loss += loss_func(preds, y)
        print('Epoch {} | Val Loss {}'.format(epoch_number, avg_val_loss/len(val_data_loader)))


num_stacked_obs = 4
num_unobf_states = 5
torch.manual_seed(32)
np.random.seed(32)
#sample_transitions(500000)

# Init regressors
fc_regressor = MLPModule(
    input_dim=num_stacked_obs * num_unobf_states,
    output_dim=3,
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

conv_regressor = ConvEncoderModel()

# Only obf states
#Epoch 8 | Val Loss 0.05511641502380371
#Epoch 9 | Batch 0/9375 | Train Loss 0.04302077367901802
#Epoch 9 | Batch 1000/9375 | Train Loss 0.04673319682478905
#Epoch 9 | Batch 2000/9375 | Train Loss 0.04747447744011879
#Epoch 9 | Batch 3000/9375 | Train Loss 0.048244211822748184
#Epoch 9 | Batch 4000/9375 | Train Loss 0.04750850424170494
#Epoch 9 | Batch 5000/9375 | Train Loss 0.04659212380647659
#Epoch 9 | Batch 6000/9375 | Train Loss 0.04652475193142891
#Epoch 9 | Batch 7000/9375 | Train Loss 0.04658839479088783
#Epoch 9 | Batch 8000/9375 | Train Loss 0.04621449485421181
#Epoch 9 | Batch 9000/9375 | Train Loss 0.04622313380241394
#Epoch 9 | Val Loss 0.05076133832335472


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
