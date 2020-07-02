import torch
import pickle
import numpy as np
from deepmdp.experiments.dqn_baseline import setup_lunar_lander_with_obfuscated_states
from deepmdp.garage_mod.modules.mlp_module import MLPModule

class StateReconstructionDataSet:
    def __init__(self, stage="train", reshape=False):
        with open("data.pkl", "rb") as f:
            data = pickle.load(f)
        if reshape:
            self.x = np.array(data["x_" + stage]).reshape(-1, num_stacked_obs, num_unobf_states)
        else:
            self.x = np.array(data["x_" + stage])
        self.y = np.array(data["y_" + stage])

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)


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
    batch_size = 32
    num_epochs = 10
    train_print_progress_every = 1000
    evaluate_only_obf_states = False
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Load data
    train_data = StateReconstructionDataSet(reshape=reshape)
    val_data = StateReconstructionDataSet(stage="val", reshape=reshape)

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
                if evaluate_only_obf_states:
                    avg_val_loss += loss_func(preds[:, [2, 3, 5]], y[:, [2, 3, 5]])
                else:
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
train(fc_regressor)

### Validaiton loss only calculaed on obfuscated states
#Epoch 8 | Val Loss 0.02061530016362667
#Epoch 9 | Batch 0/9375 | Train Loss 0.003650262486189604
#Epoch 9 | Batch 1000/9375 | Train Loss 0.0057346816174685955
#Epoch 9 | Batch 2000/9375 | Train Loss 0.006480484269559383
#Epoch 9 | Batch 3000/9375 | Train Loss 0.006486681755632162
#Epoch 9 | Batch 4000/9375 | Train Loss 0.006742620840668678
#Epoch 9 | Batch 5000/9375 | Train Loss 0.006722504273056984
#Epoch 9 | Batch 6000/9375 | Train Loss 0.006714096758514643
#Epoch 9 | Batch 7000/9375 | Train Loss 0.006624280009418726
#Epoch 9 | Batch 8000/9375 | Train Loss 0.006707015912979841
#Epoch 9 | Batch 9000/9375 | Train Loss 0.006717470474541187
#Epoch 9 | Val Loss 0.01983780227601528

conv_regressor = torch.nn.Sequential(
    torch.nn.Conv1d(
                in_channels=num_stacked_obs,
                out_channels=16,
                kernel_size=1,
                stride=1
            ),
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
#train(conv_regressor, reshape=True)

### Validaiton loss only calculaed on obfuscated states
#Epoch 8 | Val Loss 0.04671218618750572
#Epoch 9 | Batch 0/9375 | Train Loss 0.010751750320196152
#Epoch 9 | Batch 1000/9375 | Train Loss 0.014699150808155537
#Epoch 9 | Batch 2000/9375 | Train Loss 0.014667750336229801
#Epoch 9 | Batch 3000/9375 | Train Loss 0.014823892153799534
#Epoch 9 | Batch 4000/9375 | Train Loss 0.014967209659516811
#Epoch 9 | Batch 5000/9375 | Train Loss 0.014859744347631931
#Epoch 9 | Batch 6000/9375 | Train Loss 0.014808968640863895
#Epoch 9 | Batch 7000/9375 | Train Loss 0.014668078161776066
#Epoch 9 | Batch 8000/9375 | Train Loss 0.014540339820086956
#Epoch 9 | Batch 9000/9375 | Train Loss 0.014547294937074184
#Epoch 9 | Val Loss 0.043373748660087585

