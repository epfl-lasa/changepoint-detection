## Bayesian Online Multivariate Changepoint Detector

This will be tested on the bimanual peeling task, only the active arm is being analyzed at the moment.

Install all depencies found here [bimanual-task-motion-planning](https://github.com/epfl-lasa/bimanual-task-motion-planning.git)

###Replaying a recorded demonstration of a Bimanual Task:
#####Visualization and sensor bringup (ft sensors, vision)
```
$ roslaunch kuka_lwr_bringup bimanual2_realtime.launch ft_sensors:=true not_bag:=false 
```
#####Play bag
```
$ rosbag play *.bag
```

###Run Changepoint Detector
```
$ rosrun online_changepoint_detector data_listener_detector.py
```
