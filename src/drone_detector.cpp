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
        ros::NodeHandle private_node_handle("~");

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
      } 
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

        private_node_handle.param("cellSize", cellSize, int(32));
        private_node_handle.param("cellOverlay", cellOverlay, int(8));
        private_node_handle.param("surroundRadius", surroundRadius, int(4));

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


        gotCamInfo = false;
        ros::spinOnce();

        imPrev = cv::Mat(frameSize,frameSize,CV_8UC1);
        imPrev = cv::Scalar(0);

        begin = ros::Time::now();





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
                  false,
                  cellSize,
                  cellOverlay,
                  surroundRadius);

        if (bmm->initialized)
          ProcessCycle();
    }
    ~DroneDetector(){

    }

private:

    void ProcessCycle()
    {
      cv::namedWindow("Main", CV_GUI_NORMAL|CV_WINDOW_AUTOSIZE); 
      cv::RNG rng(12345);
      
      cv::Mat mask = cv::imread(MaskPath.str().c_str());
      cv::Mat imCurr_raw;

      int key = -1;
      while (key != 13)
      {
        imCurr = cv::Scalar(0);
        vc.read(imCurr_raw);
       // cv::imwrite(MaskPath.str().c_str(),imCurr_raw);

        imCurr_raw.copyTo(imCurr,mask);
        cvtColor(imCurr, imCurr, CV_RGB2GRAY);

       bmm->processImage(
           imCurr,
           imCurr_raw
           );

        key = cv::waitKey(10);
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


    void Process(const cv_bridge::CvImagePtr image)
    {
        // First things first
        if (first)
        {

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
    std::stringstream VideoPath;
    std::stringstream MaskPath;
    int VideoNumber;

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
    int cellSize;
    int cellOverlay;
    int surroundRadius;

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

