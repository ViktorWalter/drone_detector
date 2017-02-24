#include "../include/drone_detector/SparseOptFlowOcl.h"
#include <ostream>
#include <dirent.h>
#include "ros/package.h"

#define maxCornersPerBlock 80

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
  max_wg_size = (cv::ocl::Context::getContext()->getDeviceInfo().maxWorkGroupSize);
  viableSD = floor(sqrt(max_wg_size));
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
    imPrev = imCurr_t.clone();
  }

  std::vector<cv::Point2f> outvec;

  if (!initialized)
  {
    std::cout << "Structure was not initialized; Returning.";
    return outvec;
  }

  int scanDiameter = (2*scanRadius)+1;
  int blockszX = viableSD;
  int blockszY = viableSD;

  imCurr = imCurr_t.clone();

  imPrev_g.upload(imPrev);
  imCurr_g.upload(imCurr);

  std::size_t gridA[3] = {(imPrev.cols)/(blockszX),(imPrev.rows)/(blockszY),1};
  std::size_t blockA[3] = {viableSD,viableSD,1};
  std::size_t globalA[3] = {gridA[0]*blockA[0],gridA[1]*blockA[1],1};
  std::size_t one[3] = {1,1,1};

  if (first){
    ROS_INFO("%dx%d",gridA[0],gridA[1]);

    foundPointsX_g = cv::ocl::oclMat(cv::Size(maxCornersPerBlock,gridA[1]*gridA[0]),CV_16UC1);
    foundPointsY_g = cv::ocl::oclMat(cv::Size(maxCornersPerBlock,gridA[1]*gridA[0]),CV_16UC1);
    foundPointsX_prev_g = cv::ocl::oclMat(cv::Size(maxCornersPerBlock,gridA[1]*gridA[0]),CV_16UC1);
    foundPointsY_prev_g = cv::ocl::oclMat(cv::Size(maxCornersPerBlock,gridA[1]*gridA[0]),CV_16UC1);
    foundPointsX_prev_g = cv::Scalar(0);
    foundPointsY_prev_g = cv::Scalar(0);
    numFoundBlock_g =
      clCreateBuffer(
          *(cl_context*)(cv::ocl::Context::getContext()->getOpenCLContextPtr()),
          CL_MEM_READ_WRITE,
          sizeof(cl_int)*gridA[0]*gridA[1],
          NULL,
          NULL);

    numFoundBlock_prev_g = 
      clCreateBuffer(
          *(cl_context*)(cv::ocl::Context::getContext()->getOpenCLContextPtr()),
          CL_MEM_READ_WRITE,
          sizeof(cl_int)*gridA[0]*gridA[1],
          NULL,
          NULL);

  }
  cv::ocl::oclMat foundPtsX_ord_g = cv::ocl::oclMat(cv::Size(gridA[1]*gridA[0]*maxCornersPerBlock,1),CV_16SC1);
  cv::ocl::oclMat foundPtsY_ord_g = cv::ocl::oclMat(cv::Size(gridA[1]*gridA[0]*maxCornersPerBlock,1),CV_16SC1);

  int foundPtsSize = 0;
  cl_mem foundPtsSize_g =
    clCreateBuffer(
        *(cl_context*)(cv::ocl::Context::getContext()->getOpenCLContextPtr()),
        CL_MEM_READ_WRITE,
        sizeof(cl_int),
        NULL,
        NULL);
  clEnqueueWriteBuffer(
      *(cl_command_queue*)(cv::ocl::Context::getContext()->getOpenCLCommandQueuePtr()),
      foundPtsSize_g,
      CL_TRUE,
      0,
      sizeof(cl_int),
      &foundPtsSize,
      0,
      NULL,
      NULL);

  cv::ocl::oclMat imShowcorn_g = cv::ocl::oclMat(imCurr_g.size(),CV_8UC1);
  imShowcorn_g = cv::Scalar(0);
  foundPointsX_g = cv::Scalar(0);
  foundPointsY_g = cv::Scalar(0);
  foundPtsX_ord_g = cv::Scalar(0);
  foundPtsY_ord_g = cv::Scalar(0);
  int imShowCornWidth_g = imShowcorn_g.step / imShowcorn_g.elemSize();
  int imShowCornOffset_g = imShowcorn_g.offset / imShowcorn_g.elemSize();
  int imSrcWidth_g = imCurr_g.step / imCurr_g.elemSize();
  int imSrcOffset_g = imCurr_g.offset / imCurr_g.elemSize();
  int imSrcTrueWidth_g = imCurr.size().width;
  int imSrcTrueHeight_g = imCurr.size().height;
  int foundPointsWidth_g = foundPointsX_g.step / foundPointsX_g.elemSize();
  int foundPointsOffset_g = foundPointsX_g.offset/ foundPointsX_g.elemSize();
  int foundPtsXOrdOffset_g =foundPtsX_ord_g.offset/ foundPtsX_ord_g.elemSize();
  int maxCornersPerBlock_g = maxCornersPerBlock;
  int foundPointsPrevWidth_g = foundPointsX_prev_g.step / foundPointsX_prev_g.elemSize();

  //    ROS_INFO("\nsrcwidth:%d\nsrcoffset:%d\ndstwidth:%d\ndstoffset:%d\nsps:%d\nss:%d\nsr:%d\nsd:%d\n",imSrcWidth_g,imSrcOffset_g,imDstWidth_g,imDstOffset_g,samplePointSize,stepSize,scanRadius,scanDiameter);

  std::vector<std::pair<size_t , const void *> > args;
  args.clear();
  args.push_back( std::make_pair( sizeof(cl_mem), (void *) &imCurr_g.data ));
  args.push_back( std::make_pair( sizeof(cl_int), (void *) &imSrcWidth_g));
  args.push_back( std::make_pair( sizeof(cl_int), (void *) &imSrcOffset_g));
  args.push_back( std::make_pair( sizeof(cl_mem), (void *) &imShowcorn_g.data ));
  args.push_back( std::make_pair( sizeof(cl_int), (void *) &imShowCornWidth_g));
  args.push_back( std::make_pair( sizeof(cl_int), (void *) &imShowCornOffset_g));
  args.push_back( std::make_pair( sizeof(cl_mem), (void *) &foundPointsX_g.data));
  args.push_back( std::make_pair( sizeof(cl_mem), (void *) &foundPointsY_g.data));
  args.push_back( std::make_pair( sizeof(cl_mem), (void *) &numFoundBlock_g));
  args.push_back( std::make_pair( sizeof(cl_int), (void *) &foundPointsWidth_g));
  args.push_back( std::make_pair( sizeof(cl_int), (void *) &foundPointsOffset_g));
  args.push_back( std::make_pair( sizeof(cl_mem), (void *) &foundPtsX_ord_g.data));
  args.push_back( std::make_pair( sizeof(cl_mem), (void *) &foundPtsY_ord_g.data));
  args.push_back( std::make_pair( sizeof(cl_mem), (void *) &foundPtsSize_g));
  args.push_back( std::make_pair( sizeof(cl_int), (void *) &foundPtsXOrdOffset_g));
  args.push_back( std::make_pair( sizeof(cl_int), (void *) &maxCornersPerBlock_g));

  cv::ocl::openCLExecuteKernelInterop(cv::ocl::Context::getContext(),
      *program,
      "CornerPoints",
      globalA,
      blockA,
      args,
      1,
      0,
      NULL);

  clEnqueueReadBuffer(
      *(cl_command_queue*)(cv::ocl::Context::getContext()->getOpenCLCommandQueuePtr()),
      foundPtsSize_g,
      CL_TRUE,
      0,
      sizeof(cl_int),
      &foundPtsSize,
      0,
      NULL,
      NULL);
  ROS_INFO("Number of points for next phase is %d",foundPtsSize);
  //    int *fondPtsX_ord = new int[foundPtsSize];
  //    clEnqueueReadBuffer(
  //          *(cl_command_queue*)(cv::ocl::Context::getContext()->getOpenCLCommandQueuePtr()),
  //          foundPtsX_ord_g,
  //          CL_TRUE,
  //          0,
  //          sizeof(cl_int)*foundPtsSize,
  //          foundPtsX_ord,
  //          0,
  //          NULL,
  //          NULL);
  if ( (foundPtsSize > 0) && (true))
  {
    std::size_t blockB[3] = {max_wg_size,1,1};
    std::size_t gridB[3] = {foundPtsSize,1,1};
    std::size_t globalB[3] = {gridB[0]*blockB[0],1,1};

    int blockA_g = blockA[0];
    int gridA_g = gridA[0];

    args.clear();
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &imCurr_g.data ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &imPrev_g.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &imSrcWidth_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &imSrcOffset_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &imSrcTrueWidth_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &imSrcTrueHeight_g));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &foundPtsX_ord_g.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &foundPtsY_ord_g.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &foundPointsX_prev_g.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &foundPointsY_prev_g.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &foundPointsPrevWidth_g));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &numFoundBlock_prev_g));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &imShowcorn_g.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &imShowCornWidth_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &imShowCornOffset_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &scanRadius));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &samplePointSize));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &blockA_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &gridA_g));

    cv::ocl::openCLExecuteKernelInterop(cv::ocl::Context::getContext(),
        *program,
        "OptFlowReduced",
        globalB,
        blockB,
        args,
        1,
        0,
        NULL);
  }     
  imPrev_g.release();
  imCurr_g.release();
  //    cv::Mat foundPointsX(cv::Size(maxCornersPerBlock,grid[1]*grid[0]),CV_16UC1);
  //    cv::Mat foundPointsY(cv::Size(maxCornersPerBlock,grid[1]*grid[0]),CV_16UC1);
  //    foundPointsX_g.download(foundPointsX);
  //    foundPointsY_g.download(foundPointsY);
  cv::Mat imshowcorn;
  imShowcorn_g.download(imshowcorn);
  imShowcorn_g.release();
  foundPtsX_ord_g.release();
  foundPtsY_ord_g.release();
  foundPointsX_g.release();
  foundPointsY_g.release();
  clReleaseMemObject(foundPtsSize_g);

  clEnqueueCopyBuffer(
      *(cl_command_queue*)(cv::ocl::Context::getContext()->getOpenCLCommandQueuePtr()),
      numFoundBlock_g,
      numFoundBlock_prev_g,
      0,
      0,
      sizeof(cl_int)*gridA[0]*gridA[1],
      0,
      NULL,
      NULL);






  clFinish(*(cl_command_queue*)(cv::ocl::Context::getContext()->getOpenCLCommandQueuePtr()));


  if (debug)
  {
    // ROS_INFO("out: %dx%d",outX_l.cols,outX_l.rows);
  }
  if (gui)
  {
    cv::imshow("corners",imshowcorn);
    // cv::imshow("X",foundPointsX);
    //cv::imshow("Y",foundPointsY);
    //  showFlow(flowx,flowy);
  }

  imPrev = imCurr.clone();
  foundPointsX_prev_g = foundPointsX_g;
  foundPointsY_prev_g = foundPointsY_g;

  first = false;
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

