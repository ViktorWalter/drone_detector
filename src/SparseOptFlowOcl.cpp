#include "../include/drone_detector/SparseOptFlowOcl.h"
#include <ostream>
#include <dirent.h>
#include "ros/package.h"

SparseOptFlowOcl::SparseOptFlowOcl(int i_samplePointSize,
                                     int i_scanRadius,
                                     int i_stepSize,
                                     int i_cx,int i_cy,int i_fx,int i_fy,
                                     int i_k1,int i_k2,int i_k3,int i_p1,int i_p2,
                                     bool i_storeVideo)
    {
    initialized = false;
    first = true;
    scanRadius = i_scanRadius;
    samplePointSize = i_samplePointSize;
    stepSize = i_stepSize;
    cx = i_cx;
    cy = i_cy;
    fx = i_fx;
    fy = i_fy;
    k1 = i_k1;
    k2 = i_k2;
    k3 = i_k3;
    p1 = i_p1;
    p2 = i_p2;

    
    storeVideo = i_storeVideo;
    if (storeVideo) 
    {
      outputVideo.open("newCapture.avi", CV_FOURCC('M','P','E','G'),50,cv::Size(240,240),false);
      if (!outputVideo.isOpened())
        ROS_INFO("Could not open output video file");
    }


    cv::ocl::DevicesInfo devsInfo;
    if (cv::ocl::getOpenCLDevices(devsInfo,cv::ocl::CVCL_DEVICE_TYPE_ALL))
    {
        for (int i=0; i<devsInfo.size(); i++)
        {
            std::cout << "Device " << i+1 << ": " << devsInfo[i]->deviceName << std::endl;
        }
    }
    else
    {
        std::cout << "No devices found." << std::endl;
        return;
    }
    cv::ocl::setDevice(devsInfo[0]);
    int max_wg_size = (cv::ocl::Context::getContext()->getDeviceInfo().maxWorkGroupSize);
    int viableSD = floor(sqrt(max_wg_size));
    int viableSR = ((viableSD % 2)==1) ? ((viableSD-1)/2) : ((viableSD-2)/2);

    scanBlock = (scanRadius <= viableSR ? (2*scanRadius+1) : viableSD);
    ROS_INFO("Using scaning block of %d.",scanBlock);
    if (scanRadius > viableSR)
        ROS_INFO("This will require repetitions within threads.");

    FILE *program_handle;
    size_t program_size;
    //ROS_INFO((ros::package::getPath("optic_flow")+"/src/FastSpacedBMMethod.cl").c_str());
    program_handle = fopen((ros::package::getPath("drone_detector")+"/src/FSCoolPointOptFlow.cl").c_str(),"r");
    if(program_handle == NULL)
    {
        std::cout << "Couldn't find the program file" << std::endl;
        return;
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    kernelSource = (char*)malloc(program_size + 1);
    kernelSource[program_size] = '\0';
    fread(kernelSource, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    program = new cv::ocl::ProgramSource("OptFlow",kernelSource);
    initialized = true;
    ROS_INFO("OCL context initialized.");
    return ;

}

std::vector<cv::Point2f> SparseOptFlowOcl::processImage(
                                              cv::Mat imCurr_t,
                                              cv::Mat &flow_x,
                                              cv::Mat &flow_y,
                                              bool gui,
                                              bool debug)
{
    if (first)
    {
      first = false;
      imPrev = imCurr_t.clone();
    }
    
    std::vector<cv::Point2f> outvec;

    if (!initialized)
    {
        std::cout << "Structure was not initialized; Returning.";
        return outvec;
    }

    int scanDiameter = (2*scanRadius)+1;
    int blockszX = samplePointSize+stepSize;
    int blockszY = samplePointSize+stepSize;

    imCurr = imCurr_t.clone();

    imPrev_g.upload(imPrev);
    imCurr_g.upload(imCurr);

    std::size_t grid[3] = {(imPrev.cols-scanRadius*2)/blockszX,
                           (imPrev.rows-scanRadius*2)/blockszY,
                           1};
    std::size_t block[3] = {scanBlock,scanBlock,1};
    std::size_t global[3] = {grid[0]*block[0],grid[1]*block[1],1};
    std::size_t one[3] = {1,1,1};

    ROS_INFO("%dx%d",grid[0],grid[1]);


    imflowX_g = cv::ocl::oclMat(cv::Size(grid[0],grid[1]),CV_8SC1);
    imflowY_g = cv::ocl::oclMat(cv::Size(grid[0],grid[1]),CV_8SC1);
    cv::ocl::oclMat imshowcorn_g = cv::ocl::oclMat(imCurr_g.size(),CV_8UC1);
    imshowcorn_g = cv::Scalar(0);
    int showCornWidth_g = imshowcorn_g.step / imshowcorn_g.elemSize();
    int imSrcWidth_g = imCurr_g.step / imCurr_g.elemSize();
    int imSrcOffset_g = imCurr_g.offset / imCurr_g.elemSize();
    int imDstWidth_g = imflowX_g.step / imflowX_g.elemSize();
    int imDstOffset_g = imflowX_g.offset/ imflowX_g.elemSize();
    int samplePointSize_g = samplePointSize;
    int stepSize_g = stepSize;
    int scanRadius_g = scanRadius;

//    ROS_INFO("\nsrcwidth:%d\nsrcoffset:%d\ndstwidth:%d\ndstoffset:%d\nsps:%d\nss:%d\nsr:%d\nsd:%d\n",imSrcWidth_g,imSrcOffset_g,imDstWidth_g,imDstOffset_g,samplePointSize,stepSize,scanRadius,scanDiameter);

    std::vector<std::pair<size_t , const void *> > args;
    args.clear();
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &imCurr_g.data ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &imPrev_g.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &imSrcWidth_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &imSrcOffset_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &imDstWidth_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &imDstOffset_g));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &imflowX_g.data ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &imflowY_g.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &showCornWidth_g));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &imshowcorn_g.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &samplePointSize_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &stepSize_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &scanRadius_g));

    cv::ocl::openCLExecuteKernelInterop(cv::ocl::Context::getContext(),
                                        *program,
                                        "OptFlow",
                                        global,
                                        block,
                                        args,
                                        1,
                                        0,
                                        NULL);


    imPrev_g.release();
    imCurr_g.release();
    cv::Mat flowx(cv::Size(grid[0],grid[1]),CV_8SC1);
    cv::Mat flowy(cv::Size(grid[0],grid[1]),CV_8SC1);
    imflowX_g.download(flowx);
    imflowY_g.download(flowy);
    imflowX_g.release();
    imflowY_g.release();
    cv::Mat imshowcorn;
    imshowcorn_g.download(imshowcorn);
    imflowY_g.release();
    clFinish(*(cl_command_queue*)(cv::ocl::Context::getContext()->getOpenCLCommandQueuePtr()));

    if (debug)
    {
       // ROS_INFO("out: %dx%d",outX_l.cols,outX_l.rows);
    }
    if (gui)
    {
      cv::imshow("corners",imshowcorn);
        showFlow(flowx,flowy);
    }

    imPrev = imCurr.clone();

    flow_x = flowx;    
    flow_y = flowy;    

    return outvec;

}

void SparseOptFlowOcl::showFlow(const cv::Mat flowx, const cv::Mat flowy )
{
    cv::Mat out;
    drawOpticalFlow(flowx, flowy, out, 10, stepSize);

    cv::imshow("Main", imView);
    if (storeVideo)
      outputVideo << imView;
    cv::waitKey(10);
}

void SparseOptFlowOcl::drawOpticalFlow(
    const cv::Mat_<signed char>& flowx,
    const cv::Mat_<signed char>& flowy,
    cv::Mat& dst,
    float maxmotion,
    int step)
{
    imView = imCurr.clone();

    for (int y = 0; y < flowx.rows; y++)
    {
        for (int x = 0; x < flowx.cols; x++)
        {
            if ((abs(flowx(y, x)) > scanRadius) || (abs(flowy(y, x))> scanRadius))
            {
                //ROS_WARN("Flow out of bounds: X:%d, Y:%d",flowx(y, x),flowy(y, x));
                //continue;
            }
            cv::Point2i startPos(x*(step+samplePointSize)+(samplePointSize/2+scanRadius),
                                 y*(step+samplePointSize)+(samplePointSize/2+scanRadius));

            cv::Point2i u(flowx(y, x), flowy(y, x));
            cv::line(imView,
                 startPos,
                 startPos+u,
                 cv::Scalar(255));

        }
    }
    dst = imView;
}

