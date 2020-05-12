# Experiments with masked out velocity information
- Compare DQN and DeepMDP. DQN auf ausmaskitieren aber ungestackten. 
- Have 8 scalar latent vector representation. Mask the three scalars with velocity information. Feed only 5 sclars into network.
- Stack n frames
- Log correlation of latent states with velocities
- Notiz in Overleaf: Was für Experimente will ich laufen lassen? 
- Summaries am Ende vom Training


DONE:
- Reshape von (B, act_dim*n_map, H, W) embedding
- Loop over actions and do select + calculate loss.