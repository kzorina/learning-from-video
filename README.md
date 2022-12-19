# Learning  to  Use  Tools  by  Watching  Videos
    
This repository contains code for the RAL 2022 paper **Learning to Manipulate Tools by Aligning Simulation to Video Demonstration**. 
In case of any question contact us at kateryna.zorina@cvut.cz or vladimir.petrik@cvut.cz.

[Project page](https://data.ciirc.cvut.cz/public/projects/2021LearningToolMotion/)

# Instalation

### Install crocoddyl from source (tested on this version)

```
git clone https://github.com/loco-3d/crocoddyl
cd crocoddyl
git checkout b0eeaa5713166d7e6955454b90aedf0fc940baa1
mkdir build
cd build
git submodule update --init
cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_BUILD_TYPE=Release -DBoost_NO_BOOST_CMAKE=TRUE -DBoost_NO_SYSTEM_PATHS=TRUE -DBOOST_ROOT:PATHNAME=$CONDA_PREFIX -DBoost_LIBRARY_DIRS:FILEPATH=$CONDA_PREFIX/lib ..
make -j4 install
```

### Required packages:
- pyphysx
- pyphysx_envs
- pinnochio

```
pip install --upgrade git+https://github.com/petrikvladimir/pyphysx.git@master 
pip install --upgrade git+https://github.com/kzorina/pyphysx_envs.git@master 
```

### Required data:

Video inputs in pickle format. To process your own video please follow [this repository](https://github.com/zongmianli/Estimating-3D-Motion-Forces).

# Project structure

The code is divided into three main modules:

- **I_alignment** - code for processsing the video input to obtain tool-only trajectory in the aligned environment with fixed scene object positions
- **II_trajectory_optimization** - code for running trajectory optimization to repeat the trajectory from the previous step with a tool attached to the robot. This is used to pretrain the policy
- **III_train_policy** - code for training the final policy with RL. The policy is initialized from the previous step.

Each stage can be run separately:

``` 
python method/I_alignment/main.py -tool scythe -vid_id 1

python method/II_trajectory_optimization/main.py -tool scythe -vid_id 1 -robot panda

method/III_train_policy/train_ppo.py -alignment_path data/alignment/pretrained/panda/scythe/video_1/align_scythe_1.pkl \
-ddp_q_path data/alignment/pretrained/panda/scythe/video_1/q_traj_scythe_1.pkl \
-pretrain_path data/alignment/pretrained/$panda/scythe/video_1/pretrained_mu_panda_scythe_1.pkl \
-yaml method/III_train_policy/train_ppo.yaml -tool scythe
```


## Citation:

```
@article{2022LearningToolMotion,
  author    = {Kateryna Zorina and
               Justin Carpentier and
               Josef Sivic and
               Vladim{\'{\i}}r Petr{\'{\i}}k},
  title     = {Learning to Manipulate Tools by Aligning Simulation to Video Demonstration},
  journal   = {{IEEE} Robotics Autom. Lett.},
  volume    = {7},
  number    = {1},
  pages     = {438--445},
  year      = {2022},
  doi       = {10.1109/LRA.2021.3127238},
}
```
