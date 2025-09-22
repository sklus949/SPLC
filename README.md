# SPLC

<!-- This repository contains the codes for our paper titled "Intrinsic-Motivation Multi-Robot Social Formation Navigation with Coordinated Exploration". For experiment demonstrations, please refer to the [youtube video](https://youtu.be/1MUAJavW0BE). -->

This repository contains the codes for our paper titled "SPLC: Social Preference Learning for Crowd Robot Navigation". For experiment demonstrations, please refer to the [youtube video](https://youtu.be/vkWjg4Qcybg).




<!-- ### Abstract

This paper investigates the application of reinforcement learning (RL) to multi-robot social formation navigation, a critical capability for enabling seamless human-robot coexistence. While RL offers a promising paradigm, the inherent unpredictability and often uncooperative dynamics of pedestrian behavior pose substantial challenges, particularly concerning the efficiency of coordinated exploration among robots. To address this, we propose a novel coordinated-exploration multi-robot RL algorithm introducing an intrinsic motivation exploration. Its core component is a self-learning intrinsic reward mechanism designed to collectively alleviate policy conservatism. Moreover, this algorithm incorporates a dual-sampling mode within the centralized training and decentralized execution framework to enhance the representation of both the navigation policy and the intrinsic reward, leveraging a two-time-scale update rule to decouple parameter updates. Empirical results on social formation navigation benchmarks demonstrate the proposed algorithm’s superior performance over existing state-of-the-art methods across crucial metrics. -->

### The overall framework of our SPLC algorithm

<!-- ![Logo](https://raw.githubusercontent.com/sklus949/SPLC/main/crowd_nav/figures/SPLC.png) -->
<img src="https://raw.githubusercontent.com/sklus949/SPLC/main/crowd_nav/figures/SPLC.png" 
     alt="Logo" 
     width="50%"/>




### Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install crowd_sim and crowd_nav into pip
```
pip install -e .
```
   
### Overview
This repository is organized in two parts:
* crowd_nav/ folder contains configurations and policies used in the simulator.
* crowd_sim/ folder contains the simulation environment.

### Social Preference Learning
1. Data collection
```
python mechanism.py
```
2. Reward model training
```
python train_reward_model.py
```
3. Offline Reinforcement Learning
```
cd offline
python iql.py
```

<!-- ### Training curve
Training curve comparison between different methods
| ![Success](https://raw.githubusercontent.com/czxhunzi/CEMRRL/main/figures/success.png) | ![Reward](https://raw.githubusercontent.com/czxhunzi/CEMRRL/main/figures/reward.png) | ![Distance](https://raw.githubusercontent.com/czxhunzi/CEMRRL/main/figures/distance.png) |
|:---:|:---:|:---:|
| ![Success](https://raw.githubusercontent.com/czxhunzi/CEMRRL/main/figures/att_success.png) | ![Reward](https://raw.githubusercontent.com/czxhunzi/CEMRRL/main/figures/att_reward.png) | ![Distance](https://raw.githubusercontent.com/czxhunzi/CEMRRL/main/figures/att_distance.png) | -->

### Credits
This repository contains the code for the following papers:

- [Decentralized Non-communicating Multiagent Collision Avoidance with Deep Reinforcement Learning](https://arxiv.org/abs/1609.07845).
- [Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning](https://arxiv.org/abs/1805.01956).
- [Crowd-Robot Interaction: Crowd-aware Robot Navigation with Attention-based Deep Reinforcement Learning](https://arxiv.org/abs/1809.08835).
- [CORL: Research-oriented Deep Offline Reinforcement Learning Library](https://github.com/tinkoff-ai/CORL).
- [Uni-RLHF: Universal Platform and Benchmark Suite for Reinforcement Learning with Diverse Human Feedback](https://github.com/pickxiguapi/Clean-Offline-RLHF?tab=readme-ov-file).

### Contact
If you have any questions or find any bugs, please feel free to open an issue or pull request.




