# Experiments with masked out velocity information
- Compare DQN and DeepMDP. DQN auf ausmaskitieren aber ungestackten. 
- Have 8 scalar latent vector representation. Mask the three scalars with velocity information. Feed only 5 sclars into network.
- Stack n frames
- Log correlation of latent states with velocities

- Why are the having 11x11 spatial resolution?
    - Start with 84x84 stride 4 same padding --> 22x22 as output res.
    - Do stride 2 with same padding --> 12x12 as output res