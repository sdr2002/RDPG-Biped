# RDPG-Biped
Code for 'Recurrent Network-based Deterministic Policy Gradient for Solving Bipedal Walking Challenge on Rugged Terrains: https://arxiv.org/abs/1710.02896)

1) Environment: Miniconda is recommended as pybox does not support pip
- python 2.7: print format might become an issue with python 3 but other than that, is fine
- numpy, scipy, matplotlib: up-to-date
- tensorflow 1.2 : higher versions are fine and TF-GPU compatible
- OpenAI gym and pybox: for gym, download the files in 'gym-files.tar.gz' and replace 'bipedal_walk.py(many other versions are provided in the tar file)' and 'time_limit.py' into the original files

2) Run default model (Our RDPG)
- learn and run: run 'gym_ddpg.py'
- record: run 'tester_r.py'
- display: run 'display.py' within 'display.tar.gz'

3) Other models
- DDPG(Feedforward network-based DPG): d3_6
- RDPG with parameter noise: r17_41_opt0
- Our RDPG with experience injection: TBA
