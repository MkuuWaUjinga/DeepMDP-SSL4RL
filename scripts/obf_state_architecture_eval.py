import torch
import pickle
from deepmdp.experiments.dqn_baseline import setup_lunar_lander_with_obfuscated_states
from deepmdp.garage_mod.modules.mlp_module import MLPModule

class StateReconstructionDataSet:
    def __init__(self, stage="train"):
        with open("data.pkl", "rb") as f:
            data = pickle.load(f)
        self.x = data["x_" + stage]
        self.y = data["y_" + stage]

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)


def sample_transitions(k):
    # Init obf state env
    env = setup_lunar_lander_with_obfuscated_states("LunarLander-v2", do_noops=True)

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

def train(model):
    batch_size = 32
    num_epochs = 10
    train_print_progress_every = 1000
    evaluate_only_obf_states = True
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Load data
    train_data = StateReconstructionDataSet()
    val_data = StateReconstructionDataSet(stage="val")

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
                    avg_val_loss += loss_func(preds[[2, 3, 5]], y[[2, 3, 5]])
                else:
                    avg_val_loss += loss_func(preds, y)
        print('Epoch {} | Val Loss {}'.format(epoch_number, avg_val_loss/len(val_data_loader)))



torch.manual_seed(32)
#sample_transitions(500000)
# Init regressor
regressor = MLPModule(
    input_dim=20,
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
train(regressor)
