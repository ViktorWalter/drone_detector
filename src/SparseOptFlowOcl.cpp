#include "../include/drone_detector/SparseOptFlowOcl.h"
#include <ostream>
#include <dirent.h>
#include "ros/package.h"

#define maxCornersPerBlock 96
#define invalidFlow -5555

size_t roundUp(size_t sz, size_t n)
{
  size_t t = sz + n - 1;
  size_t rem = t % n;
  size_t result = t - rem;
  return result;
}

cl_kernel createKernel(
    const char* source,
    const char* kernelName)
{
  cl_kernel kernel;
  cl_int ret;
  cl_context context = *((cl_context*)(cv::ocl::Context::getContext()->getOpenCLContextPtr()));
  cl_device_id device_id = *((cl_device_id*)(cv::ocl::Context::getContext()->getOpenCLDeviceIDPtr()));
  const size_t lenSource = std::strlen(source);
  cl_program program = clCreateProgramWithSource(
      context,
      1,
      (const char**)&source,
      NULL,
      NULL);
  CV_Assert(program != NULL);
  ret = clBuildProgram(
      program,
      1,
      (cl_device_id*)cv::ocl::Context::getContext()->getOpenCLDeviceIDPtr(),
      NULL,
      NULL,
      NULL);
  kernel = clCreateKernel(
      program,
      kernelName,
      NULL);
  clReleaseProgram(program);

  return kernel;
}

void executeKernel(
    cl_kernel kernel,
    size_t globalThreads[3],
    size_t localThreads[3],
    std::vector< std::pair<size_t, const void *> > &args,
     bool debug = false)
{
  cv::ocl::Context *context = cv::ocl::Context::getContext();
  cl_command_queue queue = *(cl_command_queue*)context->getOpenCLCommandQueuePtr();
  globalThreads[0] = roundUp(globalThreads[0], localThreads[0]);
  globalThreads[1] = roundUp(globalThreads[1], localThreads[1]);
  globalThreads[2] = roundUp(globalThreads[2], localThreads[2]);

  for(size_t i = 0; i < args.size(); i ++)
    clSetKernelArg(kernel, i, args[i].first, args[i].second);

  if (!debug) {
    clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalThreads,
        localThreads, 0, NULL, NULL);
  }
  else {
    cl_event event = NULL;
    clEnqueueNDRangeKernel(queue, kernel, 3, NULL, globalThreads,
        localThreads, 0, NULL, &event);

    cl_ulong start_time, end_time, queue_time;
    double execute_time = 0;
    double total_time   = 0;

    clWaitForEvents(1, &event);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
        sizeof(cl_ulong), &start_time, 0);

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong), &end_time, 0);

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED,
        sizeof(cl_ulong), &queue_time, 0);

    execute_time = (double)(end_time - start_time) / (1000 * 1000);
    total_time = (double)(end_time - queue_time) / (1000 * 1000);

    ROS_INFO("Kernel execution time: %f",execute_time);
    ROS_INFO("Kernel total time: %f",total_time);
    clReleaseEvent(event);
  }

  clFlush(queue);
}

SparseOptFlowOcl::SparseOptFlowOcl(int i_samplePointSize,
    int i_scanRadius,
    int i_stepSize,
    int i_cx,int i_cy,int i_fx,int i_fy,
    int i_k1,int i_k2,int i_k3,int i_p1,int i_p2,
    bool i_storeVideo,
    int i_cellSize,
    int i_cellOverlay,
    int i_surroundRadius)
{
  initialized = false;
  first = true;
  scanRadius = i_scanRadius;
  samplePointSize = i_samplePointSize;
  stepSize = i_stepSize;
  cellSize = i_cellSize;
  cellOverlay = i_cellOverlay;
  surroundRadius = i_surroundRadius;
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
  cl_device_id devID = *(cl_device_id*)(cv::ocl::Context::getContext()->getOpenCLDeviceIDPtr());
  max_wg_size = (cv::ocl::Context::getContext()->getDeviceInfo().maxWorkGroupSize);
  char Vendor[50];
  cl_int ErrCode = clGetDeviceInfo(
      devID,
      CL_DEVICE_VENDOR,
      49,
      &Vendor,
      NULL);
  ROS_INFO("Device vendor is %s",Vendor);
  if (strcmp(Vendor,"NVIDIA Corporation") == 0)
    ErrCode = clGetDeviceInfo(
        devID,
        CL_DEVICE_WARP_SIZE_NV,
        sizeof(cl_uint),
        &EfficientWGSize,
        NULL);
  else
    ErrCode = clGetDeviceInfo(
        devID,
        CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        sizeof(cl_uint),
        &EfficientWGSize,
        NULL);

  ROS_INFO("Warp size is %d",EfficientWGSize);
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

  //kernel_CornerPoints = createKernel(kernelSource,"CornerPoints");
  //kernel_OptFlow = createKernel(kernelSource,"OptFlowReduced");

  initialized = true;
  ROS_INFO("OCL context initialized.");
  return ;

}

std::vector<cv::Point2f> SparseOptFlowOcl::processImage(
    cv::Mat imCurr_t,
    cv::Mat imView_t,
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
  imView = imView_t.clone();

  imPrev_g.upload(imPrev);
  imCurr_g.upload(imCurr);

  std::size_t gridA[3] = {(imPrev.cols)/(blockszX),(imPrev.rows)/(blockszY),1};
  std::size_t blockA[3] = {viableSD,viableSD,1};
  std::size_t globalA[3] = {gridA[0]*blockA[0],gridA[1]*blockA[1],1};
  std::size_t one[3] = {1,1,1};
  
  int d = cellSize-cellOverlay;
  std::size_t gridC[3] = {imCurr.cols/d,imCurr.rows/d,1};
  std::size_t blockC[3] = {max_wg_size,1,1};
  std::size_t globalC[3] = {gridC[0]*blockC[0],gridC[1]*blockC[1],1};

  if (first){
    ROS_INFO("%dx%d",gridA[0],gridA[1]);

    foundPointsX_g      = cv::ocl::oclMat(cv::Size(maxCornersPerBlock,gridA[1]*gridA[0]),CV_16UC1);
    foundPointsY_g      = cv::ocl::oclMat(cv::Size(maxCornersPerBlock,gridA[1]*gridA[0]),CV_16UC1);
    foundPointsX_prev_g = cv::ocl::oclMat(cv::Size(maxCornersPerBlock,gridA[1]*gridA[0]),CV_16UC1);
    foundPointsY_prev_g = cv::ocl::oclMat(cv::Size(maxCornersPerBlock,gridA[1]*gridA[0]),CV_16UC1);
    foundPointsX_prev_g = cv::Scalar(0);
    foundPointsY_prev_g = cv::Scalar(0);

    cellFlowX_g =
      clCreateBuffer(
          *(cl_context*)(cv::ocl::Context::getContext()->getOpenCLContextPtr()),
          CL_MEM_READ_WRITE,
          sizeof(cl_int)*gridC[0]*gridC[1],
          NULL,
          NULL);

    cellFlowY_g = 
      clCreateBuffer(
          *(cl_context*)(cv::ocl::Context::getContext()->getOpenCLContextPtr()),
          CL_MEM_READ_WRITE,
          sizeof(cl_int)*gridC[0]*gridC[1],
          NULL,
          NULL);
    cellFlowNum_g = 
      clCreateBuffer(
          *(cl_context*)(cv::ocl::Context::getContext()->getOpenCLContextPtr()),
          CL_MEM_READ_WRITE,
          sizeof(cl_int)*gridC[0]*gridC[1],
          NULL,
          NULL);
    
    emptyCellGrid = new cl_int[gridC[0]*gridC[1]];
    for (int i = 0; i < gridC[0]*gridC[1]; i++) {
      emptyCellGrid[i] = 0;
    }
    
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

    foundPtsSize_g =
      clCreateBuffer(
          *(cl_context*)(cv::ocl::Context::getContext()->getOpenCLContextPtr()),
          CL_MEM_READ_WRITE,
          sizeof(cl_int),
          NULL,
          NULL);

  }
  cv::ocl::oclMat foundPtsX_ord_g = cv::ocl::oclMat(cv::Size(gridA[1]*gridA[0]*maxCornersPerBlock,1),CV_16UC1);
  cv::ocl::oclMat foundPtsY_ord_g = cv::ocl::oclMat(cv::Size(gridA[1]*gridA[0]*maxCornersPerBlock,1),CV_16UC1);
  cv::ocl::oclMat foundPtsX_ord_flow_g = cv::ocl::oclMat(cv::Size(gridA[1]*gridA[0]*maxCornersPerBlock,1),CV_16SC1);
  cv::ocl::oclMat foundPtsY_ord_flow_g = cv::ocl::oclMat(cv::Size(gridA[1]*gridA[0]*maxCornersPerBlock,1),CV_16SC1);
  cv::ocl::oclMat outA_g = cv::ocl::oclMat(cv::Size(gridC[0],gridC[1]),CV_16SC1);
  cv::ocl::oclMat outB_g = cv::ocl::oclMat(cv::Size(gridC[0],gridC[1]),CV_16SC1);
  cv::ocl::oclMat outC_g = cv::ocl::oclMat(cv::Size(gridC[0],gridC[1]),CV_16SC1);

  clEnqueueWriteBuffer(
      *(cl_command_queue*)(cv::ocl::Context::getContext()->getOpenCLCommandQueuePtr()),
      cellFlowX_g,
      CL_TRUE,
      0,
      sizeof(cl_int)*gridC[0]*gridC[1],
      emptyCellGrid,
      0,
      NULL,
      NULL);
  clEnqueueWriteBuffer(
      *(cl_command_queue*)(cv::ocl::Context::getContext()->getOpenCLCommandQueuePtr()),
      cellFlowY_g,
      CL_TRUE,
      0,
      sizeof(cl_int)*gridC[0]*gridC[1],
      emptyCellGrid,
      0,
      NULL,
      NULL);
  clEnqueueWriteBuffer(
      *(cl_command_queue*)(cv::ocl::Context::getContext()->getOpenCLCommandQueuePtr()),
      cellFlowNum_g,
      CL_TRUE,
      0,
      sizeof(cl_int)*gridC[0]*gridC[1],
      emptyCellGrid,
      0,
      NULL,
      NULL);
  
  foundPtsSize = 0;
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
  //foundPointsX_g = cv::Scalar(0);
  //foundPointsY_g = cv::Scalar(0);
  foundPtsX_ord_g = cv::Scalar(0);
  foundPtsY_ord_g = cv::Scalar(0);
  foundPtsX_ord_flow_g = cv::Scalar(invalidFlow);
  foundPtsY_ord_flow_g = cv::Scalar(invalidFlow);
  int imShowCornWidth_g = imShowcorn_g.step / imShowcorn_g.elemSize();
  int imShowCornOffset_g = imShowcorn_g.offset / imShowcorn_g.elemSize();
  int imSrcWidth_g = imCurr_g.step / imCurr_g.elemSize();
  int imSrcOffset_g = imCurr_g.offset / imCurr_g.elemSize();
  int imSrcTrueWidth_g = imCurr.size().width;
  int imSrcTrueHeight_g = imCurr.size().height;
  int maxCornersPerBlock_g = maxCornersPerBlock;
  int invalidFlowVal_g = invalidFlow;
  int foundPointsBlockWidth_g = foundPointsX_g.step /foundPointsX_g.elemSize();

  //    ROS_INFO("\nsrcwidth:%d\nsrcoffset:%d\ndstwidth:%d\ndstoffset:%d\nsps:%d\nss:%d\nsr:%d\nsd:%d\n",imSrcWidth_g,imSrcOffset_g,imDstWidth_g,imDstOffset_g,samplePointSize,stepSize,scanRadius,scanDiameter);

  std::vector<std::pair<size_t , const void *> > args;
  args.clear();
  args.push_back( std::make_pair( sizeof(cl_mem), (void *) &imCurr_g.data ));
  args.push_back( std::make_pair( sizeof(cl_int), (void *) &imSrcWidth_g));
  args.push_back( std::make_pair( sizeof(cl_int), (void *) &imSrcOffset_g));
  args.push_back( std::make_pair( sizeof(cl_int), (void *) &imSrcTrueWidth_g));
  args.push_back( std::make_pair( sizeof(cl_mem), (void *) &imShowcorn_g.data ));
  args.push_back( std::make_pair( sizeof(cl_int), (void *) &imShowCornWidth_g));
  args.push_back( std::make_pair( sizeof(cl_int), (void *) &imShowCornOffset_g));
  args.push_back( std::make_pair( sizeof(cl_mem), (void *) &foundPointsX_g.data));
  args.push_back( std::make_pair( sizeof(cl_mem), (void *) &foundPointsY_g.data));
  args.push_back( std::make_pair( sizeof(cl_int), (void *) &foundPointsBlockWidth_g));
  args.push_back( std::make_pair( sizeof(cl_mem), (void *) &numFoundBlock_g));
  args.push_back( std::make_pair( sizeof(cl_mem), (void *) &foundPtsX_ord_g.data));
  args.push_back( std::make_pair( sizeof(cl_mem), (void *) &foundPtsY_ord_g.data));
  args.push_back( std::make_pair( sizeof(cl_mem), (void *) &foundPtsSize_g));
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

  if ( (foundPtsSize > 0) && (!first) && (true))
  {
    std::size_t blockB[3] = {EfficientWGSize,EfficientWGSize,1};
    std::size_t gridB[3] = {foundPtsSize,1,1};
    std::size_t globalB[3] = {gridB[0]*blockB[0],1,1};

    int blockA_g = blockA[0];
    int gridA_width_g = gridA[0];
    int gridA_height_g = gridA[1];
    int gridC_width_g = gridC[0];
    int gridC_height_g = gridC[1];
    int cellSize_g = cellSize;
    int surroundRadius_g = surroundRadius;
    int cellOverlay_g = cellOverlay;

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
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &foundPointsBlockWidth_g));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &numFoundBlock_prev_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &maxCornersPerBlock_g));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &imShowcorn_g.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &imShowCornWidth_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &imShowCornOffset_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &samplePointSize));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &blockA_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &gridA_width_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &gridA_height_g));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &foundPtsX_ord_flow_g.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &foundPtsY_ord_flow_g.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &cellFlowX_g));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &cellFlowY_g));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &cellFlowNum_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &gridC_width_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &cellSize_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &cellOverlay_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &invalidFlowVal_g));

    
    cv::ocl::openCLExecuteKernelInterop(cv::ocl::Context::getContext(),
        *program,
        "OptFlowReduced",
        globalB,
        blockB,
        args,
        1,
        0,
        NULL);

    outA_g = cv::Scalar(0);
    outB_g = cv::Scalar(0);
    outC_g = cv::Scalar(0);
    int outWidth_g = outA_g.step / outA_g.elemSize(); 

    args.clear();
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &outA_g.data ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &outB_g.data ));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &outC_g.data ));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &outWidth_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &cellSize_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &cellOverlay_g));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &cellFlowX_g));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &cellFlowY_g));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *) &cellFlowNum_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &gridC_width_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &invalidFlowVal_g));
    args.push_back( std::make_pair( sizeof(cl_int), (void *) &surroundRadius_g));

    
    cv::ocl::openCLExecuteKernelInterop(cv::ocl::Context::getContext(),
        *program,
        "BordersSurround",
        globalC,
        blockC,
        args,
        1,
        0,
        NULL);
    
  }     
  cv::Mat imshowcorn;
  imShowcorn_g.download(imshowcorn);
  cv::Mat foundPtsX_ord_flow, foundPtsY_ord_flow, foundPtsX_ord, foundPtsY_ord;
  foundPtsX_ord_flow_g.download(foundPtsX_ord_flow);
  foundPtsY_ord_flow_g.download(foundPtsY_ord_flow);
  foundPtsX_ord_g.download(foundPtsX_ord);
  foundPtsY_ord_g.download(foundPtsY_ord);
  outA_g.download(activationmap);
  outB_g.download(averageX);
  outC_g.download(averageY);

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

  foundPointsX_g.copyTo(foundPointsX_prev_g);
  foundPointsY_g.copyTo(foundPointsY_prev_g);

  clFinish(*(cl_command_queue*)(cv::ocl::Context::getContext()->getOpenCLCommandQueuePtr()));


  if (debug)
  {
    // ROS_INFO("out: %dx%d",outX_l.cols,outX_l.rows);
  }
  if (gui)
  {
    if (false)
      cv::imshow("corners",imshowcorn);
    else
      showFlow(
          foundPtsX_ord,
          foundPtsY_ord,
          foundPtsX_ord_flow,
          foundPtsY_ord_flow,
          false,
          activationmap,
          averageX,
          averageY);
  }

  imPrev = imCurr.clone();

  first = false;
  return outvec;

}

void SparseOptFlowOcl::showFlow(
    const cv::Mat posx,
    const cv::Mat posy,
    const cv::Mat flowx,
    const cv::Mat flowy,
    bool blankBG,
    const cv::Mat actMap,
    const cv::Mat avgX,
    const cv::Mat avgY)
{
  cv::Mat out;

  drawOpticalFlow(posx, posy, flowx, flowy, blankBG, out );
  drawActivation(actMap,avgX,avgY);

  cv::imshow("Main", imView);
  if (storeVideo)
    outputVideo << out;
}

void SparseOptFlowOcl::drawOpticalFlow(
    const cv::Mat_<ushort>& posx,
    const cv::Mat_<ushort>& posy,
    const cv::Mat_<short>& flowx,
    const cv::Mat_<short>& flowy,
    bool blankBG,
    cv::Mat& dst)
{
  if (blankBG){
    imView = cv::Mat(imCurr.size(),CV_8UC3); 
    imView = cv::Scalar(30);
  }

  int blockX = imView.cols/viableSD;
  int blockY = imView.rows/viableSD;
  if (true){
    for (int i = 0; i < blockX; i++) {
      cv::line(
          imView,
          cv::Point2i(i*viableSD,0),
          cv::Point2i(i*viableSD,imView.rows-1),
          cv::Scalar(150,150,150));

    }
    for (int i = 0; i < blockY; i++) {
      cv::line(
          imView,
          cv::Point2i(0,i*viableSD),
          cv::Point2i(imView.cols-1,i*viableSD),
          cv::Scalar(150,150,150));
    }

  }

  for (int i = 0; i < foundPtsSize; i++)
  {
    if (flowx.at<short>(0, i) == invalidFlow)
    {
      imView.at<cv::Vec3b>(posy.at<ushort>(0,i),posx.at<ushort>(0,i)) = cv::Vec3b(0,0,0);
    }
    else if (flowx.at<short>(0,i) == 8000)
    {
      cv::circle(imView,cv::Point2i(posx.at<ushort>(0,i),posy.at<ushort>(0,i)),2,cv::Scalar(0,0,0),CV_FILLED);
      //  imView.at<char>(posy.at<ushort>(0,i),posx.at<ushort>(0,i)) = 0;
    }
    else
    {
      cv::Point2i startPos(posx.at<ushort>(0,i),posy.at<ushort>(0,i));
      cv::Point2i u(flowx.at<short>(0, i), flowy.at<short>(0, i));
      cv::line(imView,
          startPos,
          u,
          cv::Scalar(255,255,255));
      cv::circle(imView,cv::Point2i(posx.at<ushort>(0,i),posy.at<ushort>(0,i)),3,cv::Scalar(0,0,0));
    }
  }
  dst = imView;
}

void SparseOptFlowOcl::drawActivation(
    const cv::Mat_<short>& actMap,
    const cv::Mat_<short>& avgX,
    const cv::Mat_<short>& avgY
    ) {
  int d = cellSize - cellOverlay;
  int m = cellSize/2;
  for (int j = 0; j < activationmap.rows; j++) {
    for (int i = 0; i < activationmap.cols; i++) {
      cv::circle(
          imView,
          cv::Point2i(i*d+m,j*d+m),
          abs(activationmap.at<short>(j,i)),
          cv::Scalar(0,0,255),
          2);
      int dx = avgX.at<short>(j,i)*2;
      int dy = avgY.at<short>(j,i)*2;
      cv::line(
          imView,
          cv::Point2i(i*d+m,j*d+m),
          cv::Point2i(i*d+m+dx,j*d+m+dy),
          cv::Scalar(255,0,0),
          2);
      
    }
    
  }
}
