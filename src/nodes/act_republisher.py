#!/usr/bin/env python3
import rosparam
import rospy
from buggycontrol.msg import Actions, ActionsStamped
from std_msgs.msg import Header

def main():
    def act_cb(msg, actpub, ctr):
        act_msg = ActionsStamped()
        act_msg.header = Header(stamp=rospy.Time.now(), frame_id="base_link", seq=ctr)
        act_msg.throttle = msg.throttle
        act_msg.turn = msg.turn
        act_msg.buttonA = msg.buttonA
        act_msg.buttonB = msg.buttonB
        actpub.publish(act_msg)
        ctr += 1

    print("Starting act republisher node")
    rospy.init_node("act_republisher")
    # rosparam.set_param("/use_sim_time", True)

    ctr = 0
    actpub = rospy.Publisher("actions_stamped", ActionsStamped, queue_size=15)
    rospy.Subscriber("actions", Actions, lambda msg : act_cb(msg, actpub, ctr), queue_size=15)
    rospy.spin()

if __name__=="__main__":
    main()