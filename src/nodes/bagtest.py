import numpy as np
import rosbag
bag = rosbag.Bag('/home/tim/SW/buggy_ws/src/buggycontrol/bagfiles/dataset/_2022-02-11-11-28-03.bag')
msg_list = []

# /camera/odom/sample, /actions
act_list = []
odom_list = []
for topic, msg, t in bag.read_messages(topics=["/actions"]):
    act_list.append({"topic" : topic, "msg" : msg, "t" : t})

for topic, msg, t in bag.read_messages(topics=["/camera/odom/sample"]):
    odom_list.append({"topic" : topic, "msg" : msg, "t" : msg.header.stamp})
bag.close()

print(odom_list[0]["t"])
print(act_list[0]["t"])

#for i in range(10000):
#    print((odom_list[i+1]["t"] - odom_list[i]["t"]).to_sec())