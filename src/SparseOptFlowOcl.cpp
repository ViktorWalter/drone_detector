#include "../include/drone_detector/SparseOptFlowOcl.h"
#include <ostream>
#include <dirent.h>
#include <algorithm>
#include "ros/package.h"
#include <X11/Xlib.h>


#define maxCornersPerBlock 96
#define invalidFlow -5555
#define enableBlankBG false
#define maxPassedPoints 2000
#define maxConsideredWindows 4 
#define windowAvgMin 10
#define windowExtendedShell 20
#define simpleDisplay false
#define maxWindowNumber 3

cv::Mat ResizeToFitRectangle(cv::Mat& src, cv::Size size) {
  if ((src.cols <= size.width) && (src.rows <= size.height))
    return src;
  cv::Size newsize;
  float srcaspect, dstaspect;
  srcaspect = src.cols/(float)src.rows;
  dstaspect = size.width/(float)size.height;
 // ROS_INFO("src: %f, dst: %f",srcaspect,dstaspect);

  if (srcaspect > dstaspect) {
    newsize.width = size.width;
    newsize.height = size.width/srcaspect;
  }
  else {
    newsize.height = size.height;
    newsize.width = size.height*srcaspect;
  }
  
  cv::Mat dst;
  cv::resize(src,dst,newsize);
  return dst;

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

  int d = cellSize - cellOverlay;
  baseWindowSize = d*3;
  imShowWindows = cv::Mat(cv::Size(baseWindowSize*maxWindowNumber,baseWindowSize),CV_8UC3);
  
  Display* disp = XOpenDisplay(NULL);
  Screen* scrn = DefaultScreenOfDisplay(disp);
  monitorSize.height = scrn->height;
  monitorSize.width = scrn->width;


  storeVideo = i_storeVideo;
  if (storeVideo) 
  {
    outputVideo.open("newCapture.avi", CV_FOURCC('M','P','E','G'),50,cv::Size(240,240),false);
    if (!outputVideo.isOpened())
      ROS_INFO("Could not open output video file");
    
  }

  cv::ocl::setUseOpenCL(true);

  cv::ocl::Context* mainContext = new cv::ocl::Context();
  mainContext->create(cv::ocl::Device::TYPE_GPU);
  cv::ocl::Device(mainContext->device(0));

  if (mainContext->ndevices() == 0)
  {
    ROS_INFO("No devices found.");
    return;
  }
  else
    ROS_INFO("%s",mainContext->device(0).name().c_str());

  device = cv::ocl::Device::getDefault();
  context = cv::ocl::Context::getDefault();
  queue = cv::ocl::Queue::getDefault();

  did = (cl_device_id)device.ptr();
  ctx = (cl_context)context.ptr();
  cqu = (cl_command_queue)queue.ptr();

  max_wg_size = device.maxWorkGroupSize();
  char Vendor[50];
  cl_int ErrCode = clGetDeviceInfo(
      did,
      CL_DEVICE_VENDOR,
      49,
      &Vendor,
      NULL);

  ROS_INFO("Device vendor is %s",Vendor);
  if (device.isNVidia()){
#ifdef CL_DEVICE_WARP_SIZE_NV
    cl_int ErrCode = clGetDeviceInfo(
        did,
        CL_DEVICE_WARP_SIZE_NV,
        sizeof(cl_uint),
        &EfficientWGSize,
        NULL);}
#else 
  return;}
#endif
  else if (device.isIntel()){
    ROS_INFO("The device is INTEL");
     cl_int ErrCode = clGetDeviceInfo(
         did,
         CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
         sizeof(cl_uint),
         &EfficientWGSize,
         NULL);
    if ((EfficientWGSize <= 0) || (EfficientWGSize > 128)) EfficientWGSize = 32;
  }
  else {
    ROS_INFO("Only NVIDIA and Intel cards are supported");
    return;
  }

  ROS_INFO("Warp size is %d",EfficientWGSize);
  viableSD = floor(sqrt(max_wg_size));
  ROS_INFO("Viable SD:%d",viableSD);
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
  int err = fread(kernelSource, sizeof(char), program_size, program_handle);
  fclose(program_handle);
  program = new cv::ocl::ProgramSource(kernelSource);

  cv::String ErrMsg;
  ROS_INFO("Here CP");
  k_CornerPoints = cv::ocl::Kernel("CornerPoints", *program,
      cv::format( "-D maxCornersPerBlock=%d -D invalidFlowVal=%d "
        "-D samplePointSize=%d -D outputFlowFieldSize=%d -D outputFlowFieldOverlay=%d "
        "-D firstStepBlockSize=%d -D surroundRadius=%d"
        ,maxCornersPerBlock,invalidFlow,samplePointSize,cellSize,cellOverlay,viableSD,surroundRadius));
    if (k_CornerPoints.empty()){
      ROS_INFO("kernel CornerPoints failed to initialize!");
      return;
    }
  ROS_INFO("Here OF");
  k_OptFlow = cv::ocl::Kernel("OptFlowReduced", *program,
      cv::format( "-D maxCornersPerBlock=%d -D invalidFlowVal=%d "
        "-D samplePointSize=%d -D outputFlowFieldSize=%d -D outputFlowFieldOverlay=%d "
        "-D firstStepBlockSize=%d -D surroundRadius=%d "
        ,maxCornersPerBlock,invalidFlow,samplePointSize,cellSize,cellOverlay,viableSD,surroundRadius));
    if (k_OptFlow.empty()){
      ROS_INFO("kernel OptFlowReduced failed to initialize!");
      return;
    }
  ROS_INFO("Here BS");
  k_BordersSurround = cv::ocl::Kernel("BordersSurround", *program,
      cv::format( "-D maxCornersPerBlock=%d -D invalidFlowVal=%d "
        "-D samplePointSize=%d -D outputFlowFieldSize=%d -D outputFlowFieldOverlay=%d "
        "-D firstStepBlockSize=%d -D surroundRadius=%d "
        ,maxCornersPerBlock,invalidFlow,samplePointSize,cellSize,cellOverlay,viableSD,surroundRadius));
    if (k_BordersSurround.empty()){
      ROS_INFO("kernel BordersSurround failed to initialize!");
      return;
    }
  ROS_INFO("Here BE");
  k_BordersEgoMovement = cv::ocl::Kernel("BordersEgoMovement", *program,
      cv::format( "-D maxCornersPerBlock=%d -D invalidFlowVal=%d "
        "-D samplePointSize=%d -D outputFlowFieldSize=%d -D outputFlowFieldOverlay=%d "
        "-D firstStepBlockSize=%d -D surroundRadius=%d "
        ,maxCornersPerBlock,invalidFlow,samplePointSize,cellSize,cellOverlay,viableSD,surroundRadius));
    if (k_BordersEgoMovement.empty()){
      ROS_INFO("kernel BordersEgoMovement failed to initialize!");
      return;
    }
  ROS_INFO("Here TS");
  k_Tester = cv::ocl::Kernel("Tester", *program,
      cv::format( "-D maxCornersPerBlock=%d -D invalidFlowVal=%d "
        "-D samplePointSize=%d -D outputFlowFieldSize=%d -D outputFlowFieldOverlay=%d "
        "-D firstStepBlockSize=%d -D surroundRadius=%d "
        ,maxCornersPerBlock,invalidFlow,samplePointSize,cellSize,cellOverlay,viableSD,surroundRadius));
    if (k_Tester.empty()){
      ROS_INFO("kernel BordersSurround failed to initialize!");
      return;
    }

  initialized = true;
  ROS_INFO("OCL context initialized.");
  return ;

}

std::vector<cv::Point2f> SparseOptFlowOcl::processImage(
    cv::Mat imCurr_t,
    cv::Mat imView_t,
    bool gui,
    bool debug,
    bool gotEgo,
    float YawRate,
    float PitchRate,
    float RollRate)
{
  
  currframe = clock();
  float dt = double(currframe - prevframe) / CLOCKS_PER_SEC;

  if (first)
  {
    imCurr_t.copyTo(imPrev_g);
    prevframe = clock();
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

  imCurr_t.copyTo(imCurr_g);
  imView = imView_t.clone();
  cv::cvtColor(imCurr_g, imCurr_bw_g, CV_RGB2GRAY);


  std::size_t gridA[3] = {(imCurr_g.cols)/(blockszX),(imCurr_g.rows)/(blockszY),1};
  std::size_t blockA[3] = {viableSD,viableSD,1};
  std::size_t globalA[3] = {gridA[0]*blockA[0],gridA[1]*blockA[1],1};
  std::size_t one[3] = {1,1,1};
  
  int d = cellSize-cellOverlay;
  std::size_t gridC[3] = {imCurr_g.cols/d,imCurr_g.rows/d,1};
  std::size_t blockC[3] = {1,1,1};
  std::size_t globalC[3] = {gridC[0]*blockC[0],gridC[1]*blockC[1],1};

  if (first){
    ROS_INFO("%dx%d",(int)gridA[0],(int)gridA[1]);

    foundPointsX_g      = cv::UMat(cv::Size(maxCornersPerBlock,gridA[1]*gridA[0]),CV_16UC1);
    foundPointsY_g      = cv::UMat(cv::Size(maxCornersPerBlock,gridA[1]*gridA[0]),CV_16UC1);
    foundPointsX_prev_g = cv::UMat(cv::Size(maxCornersPerBlock,gridA[1]*gridA[0]),CV_16UC1);
    foundPointsY_prev_g = cv::UMat(cv::Size(maxCornersPerBlock,gridA[1]*gridA[0]),CV_16UC1);
    foundPointsX_prev_g = cv::Scalar(0);
    foundPointsY_prev_g = cv::Scalar(0);
    foundPointsX_g = cv::Scalar(0);
    foundPointsY_g = cv::Scalar(0);
    activationMap_g = cv::UMat(cv::Size(gridC[0],gridC[1]),CV_16UC1);
    activationMap_prev_g = cv::UMat(cv::Size(gridC[0],gridC[1]),CV_16SC1);
    averageX_g = cv::UMat(cv::Size(gridC[0],gridC[1]),CV_16SC1);
    averageY_g = cv::UMat(cv::Size(gridC[0],gridC[1]),CV_16SC1);
    activationMap_g = cv::Scalar(0);
    activationMap_prev_g = cv::Scalar(0);
    averageX_g = cv::Scalar(0);
    averageY_g = cv::Scalar(0);


    cellFlowX_g =
      clCreateBuffer(
          ctx,
          CL_MEM_READ_WRITE,
          sizeof(cl_int)*gridC[0]*gridC[1],
          NULL,
          NULL);

    cellFlowY_g = 
      clCreateBuffer(
          ctx,
          CL_MEM_READ_WRITE,
          sizeof(cl_int)*gridC[0]*gridC[1],
          NULL,
          NULL);
    cellFlowNum_g = 
      clCreateBuffer(
          ctx,
          CL_MEM_READ_WRITE,
          sizeof(cl_int)*gridC[0]*gridC[1],
          NULL,
          NULL);
    cellFlowNum_prev_g = 
      clCreateBuffer(
          ctx,
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
          ctx,
          CL_MEM_READ_WRITE,
          sizeof(cl_int)*gridA[0]*gridA[1],
          NULL,
          NULL);

    numFoundBlock_prev_g = 
      clCreateBuffer(
          ctx,
          CL_MEM_READ_WRITE,
          sizeof(cl_int)*gridA[0]*gridA[1],
          NULL,
          NULL);
    clEnqueueWriteBuffer(
        cqu,
        cellFlowNum_prev_g,
        CL_TRUE,
        0,
        sizeof(cl_int)*gridC[0]*gridC[1],
        emptyCellGrid,
        0,
        NULL,
        NULL);

    foundPtsSize_g =
      clCreateBuffer(
          ctx,
          CL_MEM_READ_WRITE,
          sizeof(cl_int),
          NULL,
          NULL);

  }
  cv::UMat foundPtsX_ord_g = cv::UMat(cv::Size(gridA[1]*gridA[0]*maxCornersPerBlock,1),CV_16UC1);
  cv::UMat foundPtsY_ord_g = cv::UMat(cv::Size(gridA[1]*gridA[0]*maxCornersPerBlock,1),CV_16UC1);
  cv::UMat foundPtsX_ord_flow_g = cv::UMat(cv::Size(gridA[1]*gridA[0]*maxCornersPerBlock,1),CV_16SC1);
  cv::UMat foundPtsY_ord_flow_g = cv::UMat(cv::Size(gridA[1]*gridA[0]*maxCornersPerBlock,1),CV_16SC1);

  clEnqueueWriteBuffer(
      cqu,
      cellFlowX_g,
      CL_TRUE,
      0,
      sizeof(cl_int)*gridC[0]*gridC[1],
      emptyCellGrid,
      0,
      NULL,
      NULL);
  clEnqueueWriteBuffer(
      cqu,
      cellFlowY_g,
      CL_TRUE,
      0,
      sizeof(cl_int)*gridC[0]*gridC[1],
      emptyCellGrid,
      0,
      NULL,
      NULL);
  clEnqueueWriteBuffer(
      cqu,
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
      cqu,
      foundPtsSize_g,
      CL_TRUE,
      0,
      sizeof(cl_int),
      &foundPtsSize,
      0,
      NULL,
      NULL);

  cv::UMat imShowCorn_g = cv::UMat(imCurr_g.size(),CV_8UC1);
  imShowCorn_g = cv::Scalar(0);
  foundPtsX_ord_g = cv::Scalar(0);
  foundPtsY_ord_g = cv::Scalar(0);
  foundPtsX_ord_flow_g = cv::Scalar(invalidFlow);
  foundPtsY_ord_flow_g = cv::Scalar(invalidFlow);

  k_CornerPoints.args(
      cv::ocl::KernelArg::ReadOnly(imCurr_bw_g),
      cv::ocl::KernelArg::WriteOnlyNoSize(imShowCorn_g),
      cv::ocl::KernelArg::PtrWriteOnly(foundPointsX_g),
      cv::ocl::KernelArg::WriteOnly(foundPointsY_g),
      cv::ocl::KernelArg::PtrWriteOnly(foundPtsX_ord_g),
      cv::ocl::KernelArg::PtrWriteOnly(foundPtsY_ord_g),
      numFoundBlock_g,
      foundPtsSize_g);


  k_CornerPoints.run(2,globalA,blockA,true);
//  k_Tester.args(
//      cv::ocl::KernelArg::ReadWriteNoSize(imShowcorn_g));
//  k_Tester.run(2,globalA,blockA,true);
  
  clEnqueueReadBuffer(
      cqu,
      foundPtsSize_g,
      CL_TRUE,
      0,
      sizeof(cl_int),
      &foundPtsSize,
      0,
      NULL,
      NULL);

  ROS_INFO("Number of found points for next phase: %d",foundPtsSize);

  if ( (foundPtsSize > 0) && (!first) && (true))
  {
    int divider = EfficientWGSize;
    std::size_t blockB[3] = {max_wg_size/divider,divider,1};
    std::size_t gridB[3] = {foundPtsSize,1,1};
    std::size_t globalB[3] = {gridB[0]*blockB[0],gridB[1]*blockB[1],1};

    int blockA_g = blockA[0];
    int gridA_width_g = gridA[0];
    int gridA_height_g = gridA[1];
    int gridC_width_g = gridC[0];
    int gridC_height_g = gridC[1];
    int surroundRadius_g = surroundRadius;

    if (foundPtsSize > maxPassedPoints){
      ROS_INFO("Trimming to approximately %d",maxPassedPoints);
    }
    cl_bool *exclusions = new cl_bool[foundPtsSize];
    float ratio = (maxPassedPoints/(float)foundPtsSize);
    for (int i=0; i<foundPtsSize; i++){
      float rnd = std::rand()/(float)RAND_MAX;
      exclusions[i] = rnd > ratio; 
    }

    exclusions_g =
      clCreateBuffer(
          ctx,
          CL_MEM_READ_WRITE,
          sizeof(cl_bool)*foundPtsSize,
          NULL,
          NULL);
    clEnqueueWriteBuffer(
        cqu,
        exclusions_g,
        CL_TRUE,
        0,
        sizeof(cl_bool)*foundPtsSize,
        exclusions,
        0,
        NULL,
        NULL);

    int elemSize_g = (int)imCurr_g.elemSize();
//    ROS_INFO("offset: %d, step: %d, channels: %d",(int)imCurr_g.offset,(int)imCurr_g.step,elemSize_g);
    
    int i;
    i = k_OptFlow.set(0, cv::ocl::KernelArg::PtrReadOnly(imCurr_g));
    i = k_OptFlow.set(i, cv::ocl::KernelArg::ReadOnly(imPrev_g));
    i = k_OptFlow.set(i, elemSize_g);
    i = k_OptFlow.set(i, cv::ocl::KernelArg::PtrReadOnly(foundPtsX_ord_g));
    i = k_OptFlow.set(i, cv::ocl::KernelArg::PtrReadOnly(foundPtsY_ord_g));
    i = k_OptFlow.set(i, exclusions_g);
    i = k_OptFlow.set(i, cv::ocl::KernelArg::PtrReadOnly(foundPointsX_prev_g));
    i = k_OptFlow.set(i, cv::ocl::KernelArg::ReadOnlyNoSize(foundPointsY_prev_g));
    i = k_OptFlow.set(i, numFoundBlock_prev_g);
    i = k_OptFlow.set(i, cv::ocl::KernelArg::WriteOnlyNoSize(imShowCorn_g));
    i = k_OptFlow.set(i, gridA_width_g);
    i = k_OptFlow.set(i, gridA_height_g);
    i = k_OptFlow.set(i, cv::ocl::KernelArg::PtrWriteOnly(foundPtsX_ord_flow_g));
    i = k_OptFlow.set(i, cv::ocl::KernelArg::PtrWriteOnly(foundPtsY_ord_flow_g));
    i = k_OptFlow.set(i, cellFlowX_g);
    i = k_OptFlow.set(i, cellFlowY_g);
    i = k_OptFlow.set(i, cellFlowNum_g);
    i = k_OptFlow.set(i, gridC_width_g);

    clock_t begin = clock();

    k_OptFlow.run(2,globalB,blockB,true);

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    ROS_INFO("OptFlow took %f seconds",elapsed_secs);
       

    float freq_g = 1/dt;
    float cx_g = (float)cx;
    float cy_g = (float)cy;
    float focal_g = (float)fx;

    if (!gotEgo){
      k_BordersSurround.args(
          cv::ocl::KernelArg::PtrWriteOnly(activationMap_g),
          cv::ocl::KernelArg::PtrReadOnly(activationMap_prev_g),
          cv::ocl::KernelArg::PtrWriteOnly(averageX_g),
          cv::ocl::KernelArg::WriteOnlyNoSize(averageY_g),
          cellFlowX_g,
          cellFlowY_g,
          cellFlowNum_g,
          cellFlowNum_prev_g,
          gridC_width_g,
          freq_g
          );

      begin = clock();

      k_BordersSurround.run(2,globalC,blockC,true);

      end = clock();
      elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
      ROS_INFO("BordersSurround took %f seconds",elapsed_secs);
    }
    else {
      float YawRate_g = YawRate;
      float PitchRate_g = PitchRate;
      float RollRate_g = RollRate;
      k_BordersEgoMovement.args(
          cv::ocl::KernelArg::PtrWriteOnly(activationMap_g),
          cv::ocl::KernelArg::PtrReadOnly(activationMap_prev_g),
          cv::ocl::KernelArg::PtrWriteOnly(averageX_g),
          cv::ocl::KernelArg::WriteOnlyNoSize(averageY_g),
          cellFlowX_g,
          cellFlowY_g,
          cellFlowNum_g,
          cellFlowNum_prev_g,
          gridC_width_g,
          freq_g,
          focal_g,
          cx_g,
          cy_g,
          YawRate_g,
          PitchRate_g,
          RollRate_g
          );

      begin = clock();

      k_BordersEgoMovement.run(2,globalC,blockC,true);

      end = clock();
      elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
      ROS_INFO("BordersEgoMovement took %f seconds",elapsed_secs);
    }
    
  }     

  cv::Mat scaledActMap;
  activationMap_g.convertTo(scaledActMap,CV_8UC1);
//  cv::resize(scaledActMap*10, scaledActMap, imView.size(),0,0,cv::INTER_NEAREST);
//  cv::imshow("Activation", scaledActMap);

  clEnqueueCopyBuffer(
      cqu,
      numFoundBlock_g,
      numFoundBlock_prev_g,
      0,
      0,
      sizeof(cl_int)*gridA[0]*gridA[1],
      0,
      NULL,
      NULL);
  clEnqueueCopyBuffer(
      cqu,
      cellFlowNum_g,
      cellFlowNum_prev_g,
      0,
      0,
      sizeof(cl_int)*gridC[0]*gridC[1],
      0,
      NULL,
      NULL);

  foundPointsX_g.copyTo(foundPointsX_prev_g);
  foundPointsY_g.copyTo(foundPointsY_prev_g);

  std::vector<AttentionWindow> wnds =
    findWindows(
        activationMap_g.getMat(cv::ACCESS_READ),
        averageX_g.getMat(cv::ACCESS_READ),
        averageY_g.getMat(cv::ACCESS_READ)
        );



  cv::Mat imShowCorn = cv::Mat(imCurr_g.size(),imCurr_g.type());
    if (!first){
    imShowCorn_g.copyTo(imShowCorn);
    }
  if (debug)
  {
    // ROS_INFO("out: %dx%d",outX_l.cols,outX_l.rows);
  }
  if (gui)
  {
    if (simpleDisplay)
      cv::imshow("corners",imShowCorn);
    else
      showFlow(
          foundPtsX_ord_g,
          foundPtsY_ord_g,
          foundPtsX_ord_flow_g,
          foundPtsY_ord_flow_g,
          enableBlankBG,
          activationMap_g,
          averageX_g,
          averageY_g,
          wnds);
  }

   imCurr_g.copyTo(imPrev_g);
   prevframe = currframe;

  first = false;
  return outvec;

}

std::vector<AttentionWindow> SparseOptFlowOcl::findWindows(
    cv::Mat activation,
    cv::Mat flowX,
    cv::Mat flowY,
    int minWindowSize,
    int maxWindowSize)
{
  std::vector<AttentionWindow> windows;
  cv::SimpleBlobDetector::Params params;
  params.minThreshold = 200;
  params.maxThreshold = 255;
  params.thresholdStep = 5;
  params.filterByArea = true;
  params.minArea = 1;
  params.maxArea = 20;
  params.filterByCircularity = false;
  params.filterByConvexity = false;
  params.filterByColor = false;
  params.filterByInertia = false;
  cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat activation_conv;
  cv::Mat showpts = cv::Mat(activation.size(),CV_8UC1);
  showpts = cv::Scalar(255,255,255);
  activation.convertTo(activation_conv,CV_8UC1);
  activation_conv = showpts - activation_conv;
  detector->detect(activation_conv,keypoints);
//  cv::drawKeypoints(showpts,keypoints,showpts,cv::Scalar(0,255,0),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
 // cv::imshow("cv_act",activation_conv);
//  cv::imshow("cv_fl_blob",showpts);
//  if (keypoints.size() > 0){
//    ROS_INFO("keypoints: %d", keypoints.size());
//    cv::waitKey(0);
//  }
//  return windows;

  int d = cellSize - cellOverlay;
  for (int i = 0; i<keypoints.size(); i++){
          AttentionWindow wnd;
          double sz = keypoints[i].size;
          if ((sz > maxWindowSize) || (sz < minWindowSize))
            continue;
          int x  = keypoints[i].pt.x;
          int y  = keypoints[i].pt.y;
          int dirX = flowX.at<short>(y,x);
          int dirY = flowY.at<short>(y,x);
          int Act = (int)keypoints[i].response;
          wnd.sizeInCells = cv::Size((int)sz,(int)sz);
          int wndx = d*(x-sz*0.5+0.5)+windowExtendedShell+dirX;
          int wndy = d*(y-sz*0.5+0.5)+windowExtendedShell+dirY;
          int wndw = d*(sz)+windowExtendedShell;
          int wndh = d*(sz)+windowExtendedShell;
          wndx = ((wndx)>=0?wndx:0);
          wndy = ((wndy)>=0?wndy:0);
          wndw = ((wndx+wndw)<=imCurr_g.cols?wndw:imCurr_g.cols-wndx);
          wndh = ((wndy+wndh)<=imCurr_g.rows?wndh:imCurr_g.rows-wndy);
          if ((wndw<=0) || (wndh<=0))
            continue;
          wnd.rect = cv::Rect2i(wndx,wndy,wndw,wndh);
          wnd.direction = std::make_pair((float)dirX,(float)dirY);
          wnd.Activation = Act;
          windows.push_back(wnd);
  }
  return windows;

}


void SparseOptFlowOcl::showFlow(
    const cv::UMat posx,
    const cv::UMat posy,
    const cv::UMat flowx,
    const cv::UMat flowy,
    bool blankBG,
    const cv::UMat actMap,
    const cv::UMat avgX,
    const cv::UMat avgY,
    const std::vector<AttentionWindow> wnds)
{
  cv::Mat out;

  drawWindows(wnds);
  drawOpticalFlow(posx.getMat(cv::ACCESS_READ), posy.getMat(cv::ACCESS_READ), flowx.getMat(cv::ACCESS_READ), flowy.getMat(cv::ACCESS_READ), blankBG, out );
  drawActivation(actMap.getMat(cv::ACCESS_READ),avgX.getMat(cv::ACCESS_READ),avgY.getMat(cv::ACCESS_READ));

  imView = ResizeToFitRectangle(imView,monitorSize);
  cv::imshow("cv_Main", imView);
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
    imView = cv::Mat(imView.size(),CV_8UC3); 
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
      imView.at<cv::Vec3b>(posy.at<ushort>(0,i),posx.at<ushort>(0,i)) = cv::Vec3b(255,0,255);
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
  for (int j = 0; j < actMap.rows; j++) {
    for (int i = 0; i < actMap.cols; i++) {
      cv::circle(
          imView,
          cv::Point2i(i*d+m,j*d+m),
          abs(actMap.at<short>(j,i)),
          cv::Scalar(0,0,255),
          2);
//      if (activationmap.at<short>(j,i) > 20){
//        cv::waitKey(0);
//      cv::circle(
//          imView,
//          cv::Point2i(i*d+m,j*d+m),
//          abs(activationmap.at<short>(j,i)),
//          cv::Scalar(0,0,0),
//          2);
//      }
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

void SparseOptFlowOcl::drawWindows(std::vector<AttentionWindow> wnds)
{
  if (wnds.size() == 0)
    return;


  for (int i=0; i<std::min(maxWindowNumber,(int)wnds.size());i++){
    cv::Rect rect = wnds[i].rect;
    if (std::max(rect.width,rect.height) > baseWindowSize){
      baseWindowSize = std::max(rect.width,rect.height);
      cv::copyMakeBorder(
          imShowWindows,imShowWindows,
          0,
          baseWindowSize*maxWindowNumber - rect.width,
          0,
          baseWindowSize - imShowWindows.rows,
          cv::BORDER_CONSTANT);
      imShowWindows = cv::Mat(cv::Size(baseWindowSize*maxWindowNumber,baseWindowSize),CV_8UC3);
    }
    cv::Point2i corner1 = cv::Point2i(rect.x,rect.y);
    cv::Point2i corner2 = cv::Point2i(rect.x+rect.width,rect.y+rect.height);

    ROS_INFO("x:%d, y:0, w:%d, h:%d; W:%d, H:%d",baseWindowSize*i, rect.width, rect.height, imShowWindows.cols, imShowWindows.rows);
    cv::rectangle(imView, corner1,corner2, cv::Scalar(255,0,0), 2);
    /* char wndname[10]; */
    /* std::sprintf(wndname,"cv_fl_%d",i); */
    imView(rect).copyTo(imShowWindows(cv::Rect(baseWindowSize*i,0,rect.width,rect.height)));
  }
  cv::imshow("cl_fl_windows", imShowWindows);
}

