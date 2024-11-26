# EliteTrainer V1.0
 Pokemon Showdown AI Bot utilizing the DDQN algorithm. Usage of the Poke-Env package, Gymnasium, AsyncIO, and Pytorch.
 Requires running a local Pokemon Showdown server. 

 The epsilon-greedy algorithm explores before it converges to optimal policy, with experience replay and sophisticated rewards system. The selection of DDQN instead of DQN is due to the overestimation bias where DQN has the optimal action selection and proability of reward coupled together, leading to a biased assessment of the future observation state. DDQN mitigates this by having two networks, a target and an online network, where the target evaluates the probability of reward (Q-value) of an action and the online network chooses the optimal action without influencing the evaluation. DDQN works better with large, discrete action spaces, which Pokemon battling has.
