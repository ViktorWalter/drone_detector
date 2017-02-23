#define SourceDir "/home/viktor/ros_workspace/src/mbzirc/ros_nodes/drone_detector/"

#define maxTerraRange 8.0


#include <ros/ros.h>
#include <tf/tf.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Range.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/Float32.h>
using namespace std;

#include "drone_detector/SparseOptFlowOcl.h"
//#include <opencv2/gpuoptflow.hpp>
//#include <opencv2/gpulegacy.hpp>
//#include <opencv2/gpuimgproc.hpp>
//#include <time.h>



namespace enc = sensor_msgs::image_encodings;

struct PointValue
{
    int value;
    cv::Point2i location;
};



class DroneDetector
{
public:
    DroneDetector(ros::NodeHandle& node)
    {
      std::stringstream VideoPath;
      VideoPath << SourceDir << "videos/UAVfirst.MP4";

      ROS_INFO("path = %s",VideoPath.str().c_str());

      vc.open(VideoPath.str());
      if (!vc.isOpened())
      {
        ROS_INFO("video failed to open");
      }

//      cv::Mat testmat;
      vc.set(CV_CAP_PROP_POS_MSEC,(7*60+18)*1000);
//        vc.read(testmat);
//        cv::imshow("main",testmat);
//        cv::waitKey(10);

        checkAccel = false;

        ros::NodeHandle private_node_handle("~");
        private_node_handle.param("DEBUG", DEBUG, bool(false));

        private_node_handle.param("method", method, int(0));
        /* methods:
         *      0 -         */

        if(method < 3 && !useCuda){
            ROS_ERROR("Specified method support only CUDA");
        }

        private_node_handle.param("ScanRadius", scanRadius, int(8));
        private_node_handle.param("FrameSize", frameSize, int(64));
        private_node_handle.param("SamplePointSize", samplePointSize, int(8));
        private_node_handle.param("NumberOfBins", numberOfBins, int(20));


        private_node_handle.param("StepSize", stepSize, int(0));

        private_node_handle.param("gui", gui, bool(false));
        private_node_handle.param("publish", publish, bool(true));

        private_node_handle.param("useOdom",useOdom,bool(false));


        bool ImgCompressed;
        private_node_handle.param("CameraImageCompressed", ImgCompressed, bool(false));

        private_node_handle.param("ScaleFactor", ScaleFactor, int(1));

        private_node_handle.param("silentDebug", silent_debug, bool(false));


        private_node_handle.param("storeVideo", storeVideo, bool(false));




        private_node_handle.getParam("image_width", expectedWidth);

        if ((frameSize % 2) == 1)
        {
            frameSize--;
        }
        scanDiameter = (2*scanRadius+1);
        scanCount = (scanDiameter*scanDiameter);


        private_node_handle.param("cameraRotated", cameraRotated, bool(true));
        //private_node_handle.getParam("camera_rotation_matrix/data", camRot);
        private_node_handle.getParam("alpha", gamma);


        if(DEBUG)
            ROS_INFO("Waiting for camera parameters..");
        CamInfoSubscriber = node.subscribe("camera_info",1,&DroneDetector::CameraInfoCallback,this);
        gotCamInfo = false;
        ros::spinOnce();
        int i = 0;
        while(i < 100 && !gotCamInfo){
            usleep(100 * 1000); // wait 100ms
            ros::spinOnce();
            if(DEBUG)
                ROS_INFO("Still waiting for camera parameters.. (%d)",i);
            i++;
        }
        if(!gotCamInfo){
            ROS_WARN("Drone Detector missing camera calibration parameters! (nothing on camera_info topic). Loaded default parameters");
            std::vector<double> camMat;
            private_node_handle.getParam("camera_matrix/data", camMat);
            fx = camMat[0];
            cx = camMat[2];
            fy = camMat[4];
            cy = camMat[5];
            std::vector<double> distCoeffs;
            private_node_handle.getParam("distortion_coefficients/data",distCoeffs);
            k1 = distCoeffs[0];
            k2 = distCoeffs[1];
            k3 = distCoeffs[4];
            p1 = distCoeffs[2];
            p2 = distCoeffs[3];
            gotCamInfo = true;
        }

        imPrev = cv::Mat(frameSize,frameSize,CV_8UC1);
        imPrev = cv::Scalar(0);

        begin = ros::Time::now();



        //image_transport::ImageTransport iTran(node);

        VelocityPublisher = node.advertise<geometry_msgs::Twist>("velocity", 1);
        VelocitySDPublisher = node.advertise<geometry_msgs::Vector3>("velocity_stddev", 1);
        VelocityRawPublisher = node.advertise<geometry_msgs::Twist>("velocity_raw", 1);
        MaxAllowedVelocityPublisher = node.advertise<std_msgs::Float32>("max_velocity", 1);

        // Camera info subscriber

        RangeSubscriber = node.subscribe("ranger",1,&DroneDetector::RangeCallback, this);

        if (useOdom){
            TiltSubscriber = node.subscribe("odometry",1,&DroneDetector::CorrectTilt, this);
        }

        if (ImgCompressed){
            ImageSubscriber = node.subscribe("camera", 1, &DroneDetector::ProcessCompressed, this);
        }else{
            ImageSubscriber = node.subscribe("camera", 1, &DroneDetector::ProcessRaw, this);
        }

        bmm = new SparseOptFlowOcl(
                  samplePointSize,
                  scanRadius,
                  stepSize,
                  cx,
                  cy,
                  fx,
                  fy,
                  k1,
                  k2,
                  k3,
                  p1,
                  p2,
                  false);


        ProcessCycle();
    }
    ~DroneDetector(){

    }

private:

    void ProcessCycle()
    {
      cv::RNG rng(12345);
      
      std::stringstream maskpath;
      maskpath << SourceDir << "masks/first.bmp";
      cv::Mat mask = cv::imread(maskpath.str().c_str());
      cv::Mat imCurr_raw;

      int key = -1;
      while (key != 13)
      {
        vc.read(imCurr_raw);
        imCurr_raw.copyTo(imCurr,mask);
       // cv::Mat imCurr_s;       
       // cv::resize(imCurr,imCurr_s,cv::Size(192,108));
       // cv::imshow("miniature",imCurr_s);
       // 
       // cv::imshow("viewer",imCurr);
       // cv::waitKey(10);
       // continue;

/*
        std::stringstream storepath;
        storepath << SourceDir << "masks/first_t.bmp";
        cv::imwrite(storepath.str().c_str(),imCurr);
        return;
*/

        cv::Mat OptFlowImg_x;
        cv::Mat OptFlowImg_y;
        cvtColor(imCurr, imCurr, CV_RGB2GRAY);
//        cv::resize(imCurr,imCurr,cv::Size(imCurr.size().width/8,imCurr.size().height/8));

       bmm->processImage(
           imCurr,
           OptFlowImg_x,
           OptFlowImg_y);

       // cv::imshow("flowx",OptFlowImg_x);
       // cv::imshow("flowy",OptFlowImg_y);

       // std::vector<vector<cv::Point> > contours;
       // vector<cv::Vec4i> hierarchy;

       // cv::Mat FlowEdgesImg_x;
       // cv::Mat OptFlowImg_x_s;
       // OptFlowImg_x = OptFlowImg_x + scanRadius;
       // OptFlowImg_x.convertTo(OptFlowImg_x_s,CV_8UC1);
       // Canny(OptFlowImg_x_s,FlowEdgesImg_x,scanRadius,scanRadius*1.5);
       // cv::imshow("canny_x",FlowEdgesImg_x);

       // cv::Mat FlowEdgesImg_y;
       // cv::Mat OptFlowImg_y_s;
       // OptFlowImg_y = OptFlowImg_y + scanRadius;
       // OptFlowImg_y.convertTo(OptFlowImg_y_s,CV_8UC1);
       // Canny(OptFlowImg_y_s,FlowEdgesImg_y,scanRadius,scanRadius*1.5);
       // cv::imshow("canny_y",FlowEdgesImg_y);

/*
        cv::findContours(FlowEdgesImg, contours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));

        cv::Mat drawing = cv::Mat::zeros( FlowEdgesImg.size(), CV_8UC3 );
        for( int i = 0; i< contours.size(); i++ )
        {
          cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
          cv::drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, cv::Point() );
        }
        cv::imshow( "Contours", drawing );
_x

        imPrev = imCurr;*/
        key = cv::waitKey(10);
      }
    }



    void RangeCallback(const sensor_msgs::Range& range_msg)
    {
        if (range_msg.range < 0.001) //zero
        {
            return;
        }

        currentRange = range_msg.range;
        if (!useOdom)
        {
            ros::Duration sinceLast = RangeRecTime -ros::Time::now();
            Zvelocity = (currentRange - prevRange)/sinceLast.toSec();
            trueRange = currentRange;
            RangeRecTime = ros::Time::now();
            prevRange = currentRange;
        }

    }

    void CorrectTilt(const nav_msgs::Odometry odom_msg)
    {
        tf::Quaternion bt;
        tf::quaternionMsgToTF(odom_msg.pose.pose.orientation,bt);
        tf::Matrix3x3(bt).getRPY(roll, pitch, yaw);
        Zvelocity = odom_msg.twist.twist.linear.z;

        angVel = cv::Point2d(odom_msg.twist.twist.angular.y,odom_msg.twist.twist.angular.x);


        odomSpeed = cv::Point2f(odom_msg.twist.twist.linear.x,odom_msg.twist.twist.linear.y);
        odomSpeedTime = ros::Time::now();

        if (currentRange > maxTerraRange)
        {
            trueRange = odom_msg.pose.pose.position.z;
        }
        else
        {
            trueRange = cos(pitch)*cos(roll)*currentRange;
        }
    }


    void ProcessCompressed(const sensor_msgs::CompressedImageConstPtr& image_msg)
    {
        cv_bridge::CvImagePtr image;
        image = cv_bridge::toCvCopy(image_msg, enc::BGR8);
        Process(image);
    }

    void ProcessRaw(const sensor_msgs::ImageConstPtr& image_msg)
    {
        cv_bridge::CvImagePtr image;
        image = cv_bridge::toCvCopy(image_msg, enc::BGR8);
        Process(image);
    }


    void CameraInfoCallback(const sensor_msgs::CameraInfo cam_info){
        // TODO: deal with binning
        gotCamInfo = true;
        if(cam_info.binning_x != 0){
            ROS_WARN("TODO : deal with binning when loading camera parameters.");
        }
        fx = cam_info.K.at(0);
        fy = cam_info.K.at(4);
        cx = cam_info.K.at(2);
        cy = cam_info.K.at(5);
        k1 = cam_info.D.at(0);
        k2 = cam_info.D.at(1);
        p1 = cam_info.D.at(2);
        p2 = cam_info.D.at(3);
        k3 = cam_info.D.at(4);
        if(DEBUG)
            ROS_INFO("Camera params: %f %f %f %f %f %f %f %f %f",fx,fy,cx,cy,k1,k2,p1,p2,k3);
        CamInfoSubscriber.shutdown();
    }

    void Process(const cv_bridge::CvImagePtr image)
    {
        // First things first
        if (first)
        {
            /* not needed, because we are subscribed to camera_info topic
            if (ScaleFactor == 1)
            {
                int parameScale = image->image.cols/expectedWidth;
                fx = fx*parameScale;
                cx = cx*parameScale;
                fy = fy*parameScale;
                cy = cy*parameScale;
                k1 = k1*parameScale;
                k2 = k2*parameScale;
                k3 = k3*parameScale;
                p1 = p1*parameScale;
                p2 = p2*parameScale;

            }*/

            if(DEBUG){
                ROS_INFO("Source img: %dx%d", image->image.cols, image->image.rows);
            }

            first = false;
        }

        if(!gotCamInfo){
            ROS_WARN("Camera info didn't arrive yet! We don't have focus lenght coefficients. Can't publish optic flow.");
            return;
        }

        // Print out frequency
        ros::Duration dur = ros::Time::now()-begin;
        begin = ros::Time::now();
        if(DEBUG){
            ROS_INFO("freq = %fHz",1.0/dur.toSec());
        }


        // Scaling
        if (ScaleFactor != 1){
            cv::resize(image->image,imOrigScaled,cv::Size(image->image.size().width/ScaleFactor,image->image.size().height/ScaleFactor));
        }else{
            imOrigScaled = image->image.clone();
        }

        //ROS_INFO("Here 1");


        // Cropping
        if (!coordsAcquired)
        {
            imCenterX = imOrigScaled.size().width / 2;
            imCenterY = imOrigScaled.size().height / 2;
            xi = imCenterX - (frameSize/2);
            yi = imCenterY - (frameSize/2);
            frameRect = cv::Rect(xi,yi,frameSize,frameSize);
            midPoint = cv::Point2i((frameSize/2),(frameSize/2));
        }

        //ROS_INFO("Here 2");

        //  Converting color
        cv::cvtColor(imOrigScaled(frameRect),imCurr,CV_RGB2GRAY);
        
    }

private:

    cv::VideoCapture vc;

    bool first;

    ros::Time RangeRecTime;

    ros::Subscriber ImageSubscriber;
    ros::Subscriber RangeSubscriber;
    ros::Publisher VelocityPublisher;
    ros::Publisher VelocitySDPublisher;
    ros::Publisher VelocityRawPublisher;
    ros::Publisher MaxAllowedVelocityPublisher;

    ros::Subscriber CamInfoSubscriber;
    ros::Subscriber TiltSubscriber;


    cv::Mat imOrigScaled;
    cv::Mat imCurr;
    cv::Mat imPrev;

    double vxm, vym, vam;

    int imCenterX, imCenterY;    //center of original image
    int xi, xf, yi, yf; //frame corner coordinates
    cv::Point2i midPoint;
    bool coordsAcquired;
    cv::Rect frameRect;


    ros::Time begin;

    // Input arguments
    bool DEBUG;
    bool silent_debug;
    bool storeVideo;
    bool AccelerationBounding;
    //std::vector<double> camRot;
    double gamma; // rotation of camera in the helicopter frame (positive)

    int expectedWidth;
    int ScaleFactor;

    int frameSize;
    int samplePointSize;

    int scanRadius;
    int scanDiameter;
    int scanCount;
    int stepSize;

    double cx,cy,fx,fy,s;
    double k1,k2,p1,p2,k3;
    bool gotCamInfo;

    bool gui, publish, useCuda, useOdom;
    int method;

    int numberOfBins;

    bool cameraRotated;

    int RansacNumOfChosen;
    int RansacNumOfIter;
    float RansacThresholdRadSq;
    bool Allsac;

    // Ranger & odom vars
    double currentRange;
    double trueRange;
    double prevRange;
    double Zvelocity;
    double roll, pitch, yaw;


    double max_px_speed_t;
    float maxSpeed;
    float maxAccel;
    bool checkAccel;

    cv::Point2d angVel;

    cv::Point2f odomSpeed;
    ros::Time odomSpeedTime;
    float speed_noise;

    int lastSpeedsSize;
    double analyseDuration;
    SparseOptFlowOcl *bmm;
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "drone_detector");
    ros::NodeHandle nodeA;

    DroneDetector of(nodeA);
    ros::spin();
    return 0;
}

