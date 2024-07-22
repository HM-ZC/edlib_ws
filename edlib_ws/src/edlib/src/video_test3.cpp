#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <std_msgs/Bool.h>
#include <deque>
class ImageProcessor
{
public:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    ros::Publisher color_pub_;  // 添加一个发布器用于颜色检测结果
    ros::Publisher edge_pub_;
    image_transport::Publisher mask_pub_;
    ImageProcessor()
        : it_(nh_)
    {
        image_sub_ = it_.subscribe("/camera1/usb_cam1/image_raw", 1, &ImageProcessor::imageCallback, this);
        color_pub_ = nh_.advertise<std_msgs::Bool>("/color_detection/purple_detected", 10); // 颜色检测结果的话题
        edge_pub_=nh_.advertise<std_msgs::Bool>("/color_detection/ball_near_top_edge",10);
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
        cv::Scalar lower_purple = cv::Scalar(119, 22, 62);
        cv::Scalar upper_purple = cv::Scalar(170, 171, 242);
        cv::Mat mask;
        cv::inRange(hsv_image, lower_purple, upper_purple, mask);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        double maxArea = 0;
        std::vector<cv::Point>maxContour;
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area > maxArea) {
                maxArea = area;
                maxContour = contour;
            }
        }
        bool ball_near_top_edge = false;
        const int edge_threshold = 1;
        for (const auto& point : maxContour){
            if (point.y < edge_threshold){
                ball_near_top_edge=true;
                break;
            }
        } 
        std_msgs::Bool detection_msg;
        detection_msg.data = (maxArea >22500)&& !ball_near_top_edge;

        color_pub_.publish(detection_msg); // 发布颜色检测结果
        
        std_msgs::Bool edge_msg;
        edge_msg.data=ball_near_top_edge;
        edge_pub_.publish(edge_msg);
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
