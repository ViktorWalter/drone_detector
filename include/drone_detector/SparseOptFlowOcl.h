#ifndef SPARSEOPTFLOW_OCL_H
#define SPARSEOPTFLOW_OCL_H

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <image_transport/image_transport.h>
#include <opencv2/core/core.hpp>
#include <CL/cl.hpp>
#include <opencv2/ocl/ocl.hpp>


class SparseOptFlowOcl
{
private:
    bool initialized;
    bool first;
    char* kernelSource;
    cv::ocl::ProgramSource* program;
  
    int max_wg_size;

    cv::ocl::oclMat imCurr_g;
    cv::ocl::oclMat imPrev_g;

    cv::ocl::oclMat foundPointsX_g;
    cv::ocl::oclMat foundPointsY_g;
    cv::ocl::oclMat foundPointsX_prev_g;
    cv::ocl::oclMat foundPointsY_prev_g;
    cl_mem numFoundBlock_g;
    cl_mem numFoundBlock_prev_g;

    int samplePointSize;
    int scanRadius;
    int stepSize;
    int scanBlock;
    int viableSD;
    cl_int EfficientWGSize;
    int foundPtsSize;
    double cx,cy,fx,fy;
    double k1,k2,k3,p1,p2;
    bool storeVideo;

    cv::VideoWriter outputVideo;
    
    cv::Mat imPrev, imCurr, imView;

public:
    SparseOptFlowOcl(int i_samplePointSize,
                                         int i_scanRadius,
                                         int i_stepSize,
                                         int i_cx,int i_cy,int i_fx,int i_fy,
                                         int i_k1,int i_k2,int i_k3,int i_p1,int i_p2,
                                         bool i_storeVideo
                           );

    std::vector<cv::Point2f> processImage(
        cv::Mat imCurr_t,cv::Mat &flow_x,cv::Mat &flow_y,bool gui=true,bool debug=true);

    void setImPrev(cv::Mat imPrev_t)
    {
      imPrev = imPrev_t;
    }


private:
    void showFlow(const cv::Mat posx, const cv::Mat posy, const cv::Mat flowx, const cv::Mat flowy );
    void drawOpticalFlow(
        const cv::Mat_<ushort>& posx,
        const cv::Mat_<ushort>& posy,
        const cv::Mat_<short>& flowx,
        const cv::Mat_<short>& flowy,
        cv::Mat& dst);



};


#endif // SPARSEOPTFLOW_OCL_H
