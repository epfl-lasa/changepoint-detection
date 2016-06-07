#!/usr/bin/env python
import rospy
from rospy.numpy_msg import numpy_msg
import numpy

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import WrenchStamped


class Data_listener(object):
    def __init__(self, dim, pose_topic, wrench_topic):
        
        self.dim = dim
        self.init_work_variables()            
        self.pose_sub = rospy.Subscriber(pose_topic, numpy_msg(PoseStamped), self.pose_cb, queue_size=3, tcp_nodelay=True)
        self.ft_sub   = rospy.Subscriber(wrench_topic, numpy_msg(WrenchStamped), self.wrench_cb, queue_size=3, tcp_nodelay=True)

    def init_work_variables(self):
        self.X = numpy.array([0.0] * self.dim, dtype=numpy.float32) 
                                 
    def stop(self):
        '''Stop the object'''
        self.pose_sub.unregister()
        self.ft_sub.unregister()

    def pose_cb(self, msg):        
        
        # Fill out X with position
        self.X[0] = msg.pose.position.x
        self.X[1] = msg.pose.position.y
        self.X[2] = msg.pose.position.z

        # Fill out X with quaternion
        self.X[3] = msg.pose.orientation.x
        self.X[4] = msg.pose.orientation.y
        self.X[5] = msg.pose.orientation.z
        self.X[6] = msg.pose.orientation.w

    def wrench_cb(self, msg):
        self.force  = msg.wrench.force
        self.torque = msg.wrench.torque        

        # Fill out X with forces
        self.X[7] = msg.wrench.force.x
        self.X[8] = msg.wrench.force.y
        self.X[9] = msg.wrench.force.z
        
        # Fill out X with torque
        self.X[10] = msg.wrench.torque.x
        self.X[11] = msg.wrench.torque.y
        self.X[12] = msg.wrench.torque.z
        
        
        rospy.loginfo("\nX datum [%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f]", self.X[0],self.X[1],self.X[2],self.X[3],
            self.X[4],self.X[5],self.X[6],self.X[7],self.X[8],self.X[9],self.X[10],self.X[11],self.X[12])

def main():
        rospy.init_node('data_listener', anonymous=True)
        rospy.loginfo("%s: Starting" % (rospy.get_name()))

        datum_dim    = 13
        pose_topic   = "/KUKA_LeftArm/Pose"
        wrench_topic = "/tool/ft_sensor/netft_data"

        listen = Data_listener(datum_dim, pose_topic, wrench_topic)
        
        rospy.spin()
        
        rospy.loginfo("%s: Exiting" % (rospy.get_name()))
        listen.stop()

if __name__ == '__main__':
    main()