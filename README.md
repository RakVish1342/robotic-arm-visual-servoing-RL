EEE 587: Optimal Controls - Robotic Arm
---

To launch all nodes:
```
source devel/setup.bash
roslaunch simple_arm robot_spawn.launch
```

Python scripts located at:
```
catkin_ws_py/src/simple_arm/scripts
```

If you add a script, ensure execute permission is given to all node scripts. Use ```chmode 776 <script_name>```


Final Testing File
---

1. git clone
2. catkin_make
3. source devel/setup.bash
4. roscore
5. cd ./catkin_ws_py/src/simple_arm/scripts/
6. python testing_visual_servoing.py


Final Python Packages
---

```
scipy
pyyaml
rospkg
defusedxml
theano
netifaces
```


