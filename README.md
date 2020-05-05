##EEE 587: Optimal Controls - Robotic Arm

###Required Python Packages

```
scipy
pyyaml
rospkg
defusedxml
theano
netifaces
```


###Testing File

1. git clone
2. catkin_make
3. source devel/setup.bash
4. roscore
5. cd ./catkin_ws_py/src/simple_arm/scripts/
6. python testing_visual_servoing.py


###Demo Video Links:

1. Data Collection: 

```https://www.youtube.com/watch?v=f0ve1BEQK2k&feature=emb_title```

2. Results with Zero Iterations:

```https://www.youtube.com/watch?v=f0ve1BEQK2k&feature=emb_title```

3. Results with Fitted Q-Iterations:

```https://www.youtube.com/watch?v=f0ve1BEQK2k&feature=emb_title```


###Misc. Notes:

To launch Robot Arm Nodes Alone (without Q Learning Node):

```
source devel/setup.bash
roslaunch simple_arm robot_spawn.launch
```

Python scripts located at:

```
catkin_ws_py/src/simple_arm/scripts
```

