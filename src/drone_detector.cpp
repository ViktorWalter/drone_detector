#define SourceDir "/home/viktor/ros_workspace/src/mbzirc/ros_nodes/drone_detector/"

#define maxTerraRange 8.0
#define camera_delay 0.50


#include <ros/ros.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/Float32.h>
#include <thread>
#include <mutex>

#include "drone_detector/sparseOptFlowOcl.h"
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
      ros::NodeHandle private_node_handle("~");
      private_node_handle.param("uav_name", uav_name, std::string());

      private_node_handle.param("FromBag", FromBag, bool(true));
      private_node_handle.param("Flip", Flip, bool(false));
      private_node_handle.param("FromVideo", FromVideo, bool(true));
      private_node_handle.param("VideoNumber", VideoNumber, int(1));
      switch (VideoNumber) {
        case 1:
          VideoPath << SourceDir << "videos/UAVfirst.MP4";
          MaskPath << SourceDir << "masks/first.bmp";
          break;
        case 2:
          VideoPath << SourceDir << "videos/UAVsecond.MOV";
          MaskPath << SourceDir << "masks/second.bmp";
          break;
        case 3:
          VideoPath << SourceDir << "videos/UAVthird.MOV";
          MaskPath << SourceDir << "masks/third.bmp";
          break;
        case 4:
          VideoPath << SourceDir << "videos/CameraTomas.MP4)";
          MaskPath << SourceDir << "masks/fourth.bmp";
          break;
        default:
          private_node_handle.param("MaskPath", MaskPathHard, std::string("~/drone_detector/masks/generic.bmp"));

      } 

      if (MaskPathHard.empty()){
        MaskPathHard = MaskPath.str();
      }
      ROS_INFO("VideoPath = %s",VideoPath.str().c_str());
      ROS_INFO("MaskPath = %s",MaskPathHard.c_str());

      private_node_handle.param("camNum", camNum, int(0));

      if (FromVideo){
        vc.open(VideoPath.str());
        vc.set(CV_CAP_PROP_POS_MSEC,(7*60+18)*1000);
        if (!vc.isOpened())
        {
          ROS_INFO("video failed to open");
        }
      }
      else if (!FromBag)
      {
        ROS_INFO("Using camera no. %d",camNum);
        vc.open(camNum);
        vc.set(CV_CAP_PROP_FRAME_WIDTH,1280);
        vc.set(CV_CAP_PROP_FRAME_HEIGHT,720);
        vc.set(CV_CAP_PROP_FPS,30);
        if (!vc.isOpened())
        {
          ROS_INFO("camera failed to start");
        }
      }
      
      //      cv::Mat testmat;

      private_node_handle.param("cellSize", cellSize, int(32));
      private_node_handle.param("cellOverlay", cellOverlay, int(8));
      private_node_handle.param("surroundRadius", surroundRadius, int(4));

      private_node_handle.param("DEBUG", DEBUG, bool(false));



      private_node_handle.param("SamplePointSize", samplePointSize, int(8));



      private_node_handle.param("gui", gui, bool(false));
      private_node_handle.param("publish", publish, bool(true));

      private_node_handle.param("useOdom",useOdom,bool(false));

      ROS_INFO("UseOdom? %s",useOdom?"true":"false");
      if (useOdom){

        yawRate = 0.0;
        pitchRate = 0.0;
        rollRate = 0.0;
        listener = new tf::TransformListener();
        TiltSubscriber = private_node_handle.subscribe("imu", 1, &DroneDetector::TiltCallback, this, ros::TransportHints().tcpNoDelay());
      }

      bool ImgCompressed;
      private_node_handle.param("CameraImageCompressed", ImgCompressed, bool(false));


      private_node_handle.param("silentDebug", silent_debug, bool(false));


      private_node_handle.param("storeVideo", storeVideo, bool(false));



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


      private_node_handle.param("cameraRotated", cameraRotated, bool(true));
      //private_node_handle.getParam("camera_rotation_matrix/data", camRot);
      private_node_handle.getParam("alpha", gamma);


      gotCamInfo = false;

      ros::Time::waitForValid();
      begin = ros::Time::now();
      

      bmm = new SparseOptFlowOcl(
          samplePointSize,
          cx,
          cy,
          fx,
          fy,
          k1,
          k2,
          k3,
          p1,
          p2,
          false,
          cellSize,
          cellOverlay,
          surroundRadius);

      if (FromBag){
        stopped = false;
        if (ImgCompressed){
          ImageSubscriber = node.subscribe("camera", 1, &DroneDetector::ProcessCompressed, this);
        }else{
          ImageSubscriber = node.subscribe("camera", 1, &DroneDetector::ProcessRaw, this);
        }
      }
      else{
        if (bmm->initialized){
          main_thread = std::thread(&DroneDetector::ProcessVideoInput, this);
        }
      }
    }

    ~DroneDetector(){

    }

  private:

    void ProcessVideoInput()
    {
      cv::namedWindow("cv_Main", CV_GUI_NORMAL|CV_WINDOW_AUTOSIZE); 
      cv::RNG rng(12345);

      cv::Mat mask = cv::imread(MaskPathHard.c_str());

      cv::Mat imCurr_raw;

      ros::Rate frame_rate(50);

      double yaw_local;
      double pitch_local;
      double roll_local;

      int key = -1;
      while (key != 13 && ros::ok())
      {

        /* ROS_INFO_THROTTLE(1, "The thread is running..."); */

        imCurr = cv::Scalar(0,0,0);
        vc.read(imCurr_raw);
        // cv::imwrite(MaskPath.str().c_str(),imCurr_raw);

        imCurr_raw.copyTo(imCurr, mask);
        //imCurr_raw.copyTo(imCurr);
        //cv::resize(imCurr_raw,imCurr,cv::Size(320,240));
        //cv::imshow("vw",imCurr);
        //

        if (useOdom) {

          mutex_imu.lock();
          {
            yaw_local = yawRate;
            pitch_local = pitchRate;
            roll_local = rollRate;
          }
          mutex_imu.unlock();

          /* ROS_INFO("y:%f\tp:%f\tr:%f",yaw_local,pitch_local,roll_local); */

          bmm->processImage(
              imCurr,
              imCurr_raw,
              true,
              true,
              true,
              yaw_local,
              pitch_local,
              roll_local
              );

        } else {

          bmm->processImage(
              imCurr,
              imCurr_raw
              );
        }

          key = cv::waitKey(10);
          frame_rate.sleep();
        
      }
    }

    void TiltCallback(const sensor_msgs::ImuConstPtr& imu_msg){
      imu_register.insert(imu_register.begin(),*imu_msg);
      bool caughtUp = false;
      bool reachedEnd = false;
      while (!reachedEnd) {
        if ((ros::Time::now() - imu_register.back().header.stamp) > ros::Duration(camera_delay)){
          caughtUp = true;
          imu_register.pop_back();
        }
        else 
          reachedEnd = true;
      }

        ros::Duration dur = (ros::Time::now() - imu_register.back().header.stamp);
          ROS_INFO("here, back = %f, buffer = %d",dur.toSec(), (int)imu_register.size());
          
      if (!caughtUp)
        return;

      mutex_imu.lock();
      {
        yawRate = imu_register.back().angular_velocity.z;
        pitchRate = imu_register.back().angular_velocity.y;
        rollRate = imu_register.back().angular_velocity.x;
      }
      mutex_imu.unlock();

      /* ROS_INFO( "Y:%f, P:%f, R:%f, B:%d", yawRate, pitchRate, rollRate, (int)(imu_register.size())); */
    }

    void ProcessCompressed(const sensor_msgs::CompressedImageConstPtr& image_msg)
    {
      cv_bridge::CvImagePtr image;
      image = cv_bridge::toCvCopy(image_msg, enc::BGR8);
      ProcessSingleImage(image);
    }

    void ProcessRaw(const sensor_msgs::ImageConstPtr& image_msg)
    {
      cv_bridge::CvImagePtr image;
      image = cv_bridge::toCvCopy(image_msg, enc::BGR8);
      ProcessSingleImage(image);
    }


    void ProcessSingleImage(const cv_bridge::CvImagePtr image)
    {
      if (stopped) return;
      int key = -1;

      // First things first
      if (first)
      {

        if(DEBUG){
          ROS_INFO("Source img: %dx%d", image->image.cols, image->image.rows);
        }
        cv::namedWindow("cv_Main", CV_GUI_NORMAL|CV_WINDOW_AUTOSIZE); 

        first = false;
      }
      cv::Mat localImg_raw, localImg;
      if (Flip){
        cv::flip(image->image,localImg_raw,-1);
      }
      else
        image->image.copyTo(localImg_raw);

      cv::Mat mask = cv::imread(MaskPathHard.c_str());
      localImg_raw.copyTo(localImg,mask);

      bmm->processImage(
          localImg
          ,
          localImg_raw
          );
      /* cv::imshow("fthis",image->image); */
      key = cv::waitKey(10);

      if (key == 13)
        stopped = true;

      //
      //        if(!gotCamInfo){
      //            ROS_WARN("Camera info didn't arrive yet! We don't have focus lenght coefficients. Can't publish optic flow.");
      //            return;
      //        }

      // Print out frequency
      //        ros::Duration dur = ros::Time::now()-begin;
      //        begin = ros::Time::now();
      //        if(DEBUG){
      //            ROS_INFO("freq = %fHz",1.0/dur.toSec());
      //        }


      // Scaling

      //ROS_INFO("Here 1");


      // Cropping
      //        if (!coordsAcquired)
      //        {
      //            imCenterX = imOrigScaled.size().width / 2;
      //            imCenterY = imOrigScaled.size().height / 2;
      //        }
      //
      //        //ROS_INFO("Here 2");

      //  Converting color
      //        cv::cvtColor(imOrigScaled(frameRect),imCurr,CV_RGB2GRAY);

    }

  private:

    cv::VideoCapture vc;
    std::stringstream VideoPath;
    std::stringstream MaskPath;
    std::string MaskPathHard;
    int VideoNumber;
    bool FromVideo;
    bool FromBag;
    int camNum;

    bool first;
    bool stopped;

    bool Flip;

    ros::Time RangeRecTime;


    ros::Subscriber CamInfoSubscriber;
    ros::Subscriber TiltSubscriber;
    ros::Subscriber ImageSubscriber;
    
    tf::TransformListener *listener;


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


    int samplePointSize;

    int cellSize;
    int cellOverlay;
    int surroundRadius;

    double cx,cy,fx,fy,s;
    double k1,k2,p1,p2,k3;
    bool gotCamInfo;

    bool gui, publish, useCuda, useOdom;

    int numberOfBins;

    bool cameraRotated;

    int RansacNumOfChosen;
    int RansacNumOfIter;
    float RansacThresholdRadSq;
    bool Allsac;

    double rollRate, pitchRate, yawRate;
    std::mutex mutex_imu; 

    double max_px_speed_t;
    float maxSpeed;
    float maxAccel;
    bool checkAccel;

    std::string uav_name;

    ros::Time odomSpeedTime;
    float speed_noise;

    int lastSpeedsSize;
    double analyseDuration;
    SparseOptFlowOcl *bmm;

    // thread
    std::thread main_thread;

    std::vector<sensor_msgs::Imu> imu_register;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "drone_detector");
  ros::NodeHandle nodeA;
  DroneDetector of(nodeA);

  ROS_INFO("NODE initiated");

  ros::spin();

  return 0;
}

