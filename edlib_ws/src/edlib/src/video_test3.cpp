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
    image_transport::Publisher mask_pub_;
    std::deque<cv::Mat> frames; // 用于多帧图像融合
    const int frame_count = 2;  // 多帧融合的帧数
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
        // 保存帧到队列
        frames.push_back(cv_ptr->image);
        if (frames.size() > frame_count) {
            frames.pop_front();
        }

        // 多帧图像融合
        cv::Mat fused_frame = cv::Mat::zeros(cv_ptr->image.size(), cv_ptr->image.type());
        for (const auto& frame : frames) {
            fused_frame += frame / static_cast<double>(frames.size());
        }

        // 图像预处理
        cv::GaussianBlur(fused_frame, fused_frame, cv::Size(5, 5), 0);
        cv::Mat gray_image;
        cv::cvtColor(fused_frame, gray_image, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray_image, gray_image);

        // 应用CLAHE
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(2.0);
        clahe->apply(gray_image, gray_image);

        // 自适应阈值
        cv::Mat thresh_image;
        cv::adaptiveThreshold(gray_image, thresh_image, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);
        cv::Mat hsv_image;
        cv::cvtColor(fused_frame, hsv_image, cv::COLOR_BGR2HSV);
        cv::Scalar lower_purple = cv::Scalar(119, 22, 62);
        cv::Scalar upper_purple = cv::Scalar(170, 171, 242);
        cv::Mat mask;
        cv::inRange(hsv_image, lower_purple, upper_purple, mask);
        // 将阈值处理结果与颜色检测掩码结合
        cv::Mat combined_mask;
        cv::bitwise_and(mask, thresh_image, combined_mask);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(combined_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

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