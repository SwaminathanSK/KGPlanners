#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import PoseStamped
import os

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

    def my_estimatePoseSingleMarkers(self, corners, markerSize, mtx, distortion):

        marker_points = np.array([[-markerSize / 2, markerSize / 2, 0],
                              [markerSize / 2, markerSize / 2, 0],
                              [markerSize / 2, -markerSize / 2, 0],
                              [-markerSize / 2, -markerSize / 2, 0]], dtype=np.float32)
        trash = []
        rvecs = []
        tvecs = []
        i = 0
        for c in corners:
            lilTrash, R, t = cv2.solvePnP(marker_points, corners[i], mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
            rvecs.append(R)
            tvecs.append(t)
            trash.append(lilTrash)
        return np.array(rvecs), np.array(tvecs), trash

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
                    
                    rvec, tvec, _ = self.my_estimatePoseSingleMarkers(corners[0], 0.1, self.camera_matrix, self.dist_coeff)
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
