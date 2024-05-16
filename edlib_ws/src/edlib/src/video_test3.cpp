#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <std_msgs/Bool.h>

class ImageProcessor
{
public:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    ros::Publisher color_pub_;  // 添加一个发布器用于颜色检测结果
    image_transport::Publisher mask_pub_;
    ImageProcessor()
        : it_(nh_)
    {
        image_sub_ = it_.subscribe("/usb_cam/image_raw", 1, &ImageProcessor::imageCallback, this);
        color_pub_ = nh_.advertise<std_msgs::Bool>("/color_detection/purple_detected", 10); // 颜色检测结果的话题
        mask_pub_ = it_.advertise("/color_detection/mask_image", 1); 
        cv::namedWindow("Mask", cv::WINDOW_NORMAL);
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat hsv_image;
        cv::cvtColor(cv_ptr->image, hsv_image, cv::COLOR_BGR2HSV);
        cv::Scalar lower_purple = cv::Scalar(0, 0, 150);
        cv::Scalar upper_purple = cv::Scalar(90, 20, 255);
        cv::Mat mask;
        cv::inRange(hsv_image, lower_purple, upper_purple, mask);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        double maxArea = 0;
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area > maxArea) {
                maxArea = area;
            }
        }

        std_msgs::Bool detection_msg;
        detection_msg.data = maxArea >100000;

        color_pub_.publish(detection_msg); // 发布颜色检测结果
    	// 发布掩码图像
    	sensor_msgs::ImagePtr out_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", mask).toImageMsg();
    	mask_pub_.publish(out_msg);
        cv::imshow("Mask", mask);
        cv::waitKey(1);
        // 如果需要，将处理后的图像发布出去
        //sensor_msgs::ImagePtr out_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_ptr->image).toImageMsg();
        //image_pub_.publish(out_msg);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_processor");
    ImageProcessor ip;
    ros::spin();
    return 0;
}
