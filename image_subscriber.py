#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import PoseStamped
import os

def getMarkerIDFromCorners(corner_set):

    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    
    marker_id = -1

    corner_set_cv = cv2.UMat(corner_set)

    for corner in corner_set_cv:
      
        corner_list = [corner.get()]

       
        marker_ids = cv2.aruco.detectMarkers(corner_list, aruco_dict)[1]
        if marker_ids is not None and len(marker_ids) > 0:
            marker_id = marker_ids[0][0]
            break  
    return marker_id

def estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeff):

    if len(corners) != 1:
        raise ValueError("This function expects exactly one marker corner set.")


    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)


    marker_id = getMarkerIDFromCorners(corners[0][0])


    obj_points = np.array([[0, 0, 0], [marker_size, 0, 0], [marker_size, marker_size, 0], [0, marker_size, 0]], dtype=np.float32)


    _, rvec, tvec = cv2.solvePnP(obj_points, corners[0], camera_matrix, dist_coeff)

    return rvec.squeeze(), tvec.squeeze(), 1


class ArucoPoseEstimator:
    
    def __init__(self):
        
        rospy.init_node('image_subscriber', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.camera_info_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback)
        self.pose_pub = rospy.Publisher('/aruco_pose', PoseStamped, queue_size=10)
        self.camera_matrix = None
        self.dist_coeff = None
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
        self.parameters = cv2.aruco.DetectorParameters_create()
        self.first_frame_saved = False 
        self.rate = rospy.Rate(0.5) 

        self.image_save_path = os.path.join(os.path.expanduser("~"), "saved_images")
        os.makedirs(self.image_save_path, exist_ok=True)

    def camera_info_callback(self, msg):

        self.camera_matrix = np.array(msg.K).reshape((3, 3))
        self.dist_coeff = np.array(msg.D)

       
    def image_callback(self, msg):
        
        if not self.first_frame_saved:
            try:
            
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

                if not self.first_frame_saved:
                    image_filename = os.path.join(self.image_save_path, "first_frame.png")
                    cv2.imwrite(image_filename, cv_image)
                    rospy.loginfo("First frame saved as %s", image_filename)
                    self.first_frame_saved = True

            
                corners, ids, _ = cv2.aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.parameters)

                if ids is not None and len(ids) > 0:
                    
                    rvec, tvec, _ = estimatePoseSingleMarkers(corners[0], 0.1, self.camera_matrix, self.dist_coeff)
                    rvec = rvec.squeeze()
                    tvec = tvec.squeeze()

                
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = rospy.Time.now()
                    pose_msg.pose.position.x = tvec[0]
                    pose_msg.pose.position.y = tvec[1]
                    pose_msg.pose.position.z = tvec[2]
                    pose_msg.pose.orientation.x = rvec[0]
                    pose_msg.pose.orientation.y = rvec[1]
                    pose_msg.pose.orientation.z = rvec[2]
                    pose_msg.pose.orientation.w = 1.0  

                    self.pose_pub.publish(pose_msg)

            except Exception as e:
                rospy.logerr("Error processing the image: %s", str(e))

    def run(self):
        while not rospy.is_shutdown():
            self.rate.sleep()

def main():
    aruco_estimator = ArucoPoseEstimator()
    try:
        aruco_estimator.run()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")

if __name__ == '__main__':
    main()
