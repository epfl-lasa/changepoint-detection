#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import WrenchStamped

def pose_callback(msg):
    position = msg.pose.position
    quat = msg.pose.orientation
    
def wrench_callback(msg):
    force  = msg.wrench.force
    torque = msg.wrench.torque
    
    rospy.loginfo("EE Position: [ %f, %f, %f ]"%(position.x, position.y, position.z))
    rospy.loginfo("EE Orientation (quat): [ %f, %f, %f, %f]"%(quat.x, quat.y, quat.z, quat.w))
    rospy.loginfo("EE Force:  [ %f, %f, %f ]"%(force.x, force.y, force.z))
    rospy.loginfo("EE Torque: [ %f, %f, %f ]"%(torque.x, torque.y, torque.z))

def data_listener():
    rospy.init_node('data_listener', anonymous=True)
    rospy.Subscriber("/KUKA_LeftArm/Pose", PoseStamped, pose_callback)
    rospy.Subscriber("/tool/ft_sensor/netft_data", WrenchStamped, wrench_callback)
    rospy.spin()

if __name__ == '__main__':
    data_listener()