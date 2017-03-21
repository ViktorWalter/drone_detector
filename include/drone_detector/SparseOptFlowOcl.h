#ifndef SPARSEOPTFLOW_OCL_H
#define SPARSEOPTFLOW_OCL_H

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <image_transport/image_transport.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <CL/cl.hpp>


class SparseOptFlowOcl
{
private:
    bool first;
    char* kernelSource;
    cv::ocl::ProgramSource* program;
    cl_program prog_CornerPoints;
    cl_program prog_OptFlow;
    cl_kernel kernel_CornerPoints;
    cl_kernel kernel_OptFlow;
    cv::ocl::Kernel k_OptFlow, k_CornerPoints, k_BordersSurround;
    cv::ocl::Kernel k_Tester;
    cv::ocl::Device device;
    cv::ocl::Context context;
    cv::ocl::Queue queue;
    cl_device_id did;
    cl_context ctx;
    cl_command_queue cqu;
  
    int max_wg_size;

    cv::UMat imCurr_g;
    cv::UMat imCurr_bw_g;
    cv::UMat imPrev_g;

//    cv::UMat foundPointsX_g;
//    cv::UMat foundPointsY_g;
//    cv::UMat foundPointsX_prev_g;
//    cv::UMat foundPointsY_prev_g;

    cv::UMat foundPointsX_g;
    cv::UMat foundPointsY_g;
    cv::UMat foundPointsX_prev_g;
    cv::UMat foundPointsY_prev_g;
    cv::UMat activationMap_g;
    cv::UMat activationMap_prev_g;
    cv::UMat averageX_g;
    cv::UMat averageY_g;
    cl_mem foundPtsSize_g;
    cl_mem numFoundBlock_g;
    cl_mem numFoundBlock_prev_g;
    cl_mem cellFlowX_g;
    cl_mem cellFlowY_g;
    cl_mem cellFlowNum_g;
    cl_mem exclusions_g;

    int samplePointSize;
    int scanRadius;
    int stepSize;
    int scanBlock;
    int viableSD;
    int cellSize;
    int cellOverlay;
    int surroundRadius;
    cl_int EfficientWGSize;
    int foundPtsSize;
    double cx,cy,fx,fy;
    double k1,k2,k3,p1,p2;
    bool storeVideo;
    cl_int *emptyCellGrid;

    cv::VideoWriter outputVideo;
   
    cv::Mat imView;

    cv::Size monitorSize;

public:
    SparseOptFlowOcl(int i_samplePointSize,
                                         int i_scanRadius,
                                         int i_stepSize,
                                         int i_cx,int i_cy,int i_fx,int i_fy,
                                         int i_k1,int i_k2,int i_k3,int i_p1,int i_p2,
                                         bool i_storeVideo,
                                         int i_cellSize,
                                         int i_cellOverlay,
                                         int i_surroundRadius
                           );

    std::vector<cv::Point2f> processImage(
        cv::Mat imCurr_t,cv::Mat imView_t,bool gui=true,bool debug=true);

    void setImPrev(cv::Mat imPrev_t)
    {
      imPrev_g = imPrev_t.getUMat(cv::ACCESS_READ);
    }

    bool initialized;


private:
    void showFlow(
        const cv::UMat posx,
        const cv::UMat posy,
        const cv::UMat flowx,
        const cv::UMat flowy,
        bool blankBG,
        const cv::UMat actMap,
        const cv::UMat avgX,
        const cv::UMat avgY
        );
    void drawOpticalFlow(
        const cv::Mat_<ushort>& posx,
        const cv::Mat_<ushort>& posy,
        const cv::Mat_<short>& flowx,
        const cv::Mat_<short>& flowy,
        bool blankBG,
        cv::Mat& dst);
    void drawActivation(
        const cv::Mat_<short>& actMap,
        const cv::Mat_<short>& avgX,
        const cv::Mat_<short>& avgY
        );



};


#endif // SPARSEOPTFLOW_OCL_H
