#include "../include/drone_detector/SparseOptFlowOcl.h"
#include <ostream>
#include <dirent.h>
#include "ros/package.h"
#include <X11/Xlib.h>

#define maxCornersPerBlock 96
#define invalidFlow -5555
#define enableBlankBG false
#define maxPassedPoints 4000

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
  mainContext->create(cv::ocl::Device::TYPE_DGPU);
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
  if (device.isNVidia())
    cl_int ErrCode = clGetDeviceInfo(
        did,
        CL_DEVICE_WARP_SIZE_NV,
        sizeof(cl_uint),
        &EfficientWGSize,
        NULL);
  else if (device.isIntel())
    cl_int ErrCode = clGetDeviceInfo(
        did,
        CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        sizeof(cl_uint),
        &EfficientWGSize,
        NULL);
  else {
    ROS_INFO("Only NVIDIA and Intel cards are supported");
    return;
  }

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
  program = new cv::ocl::ProgramSource(kernelSource);

  cv::String ErrMsg;
  k_CornerPoints = cv::ocl::Kernel("CornerPoints", *program,
      cv::format( "-D maxCornersPerBlock=%d -D invalidFlowVal=%d "
        "-D samplePointSize=%d -D outputFlowFieldSize=%d -D outputFlowFieldOverlay=%d "
        "-D firstStepBlockSize=%d -D surroundRadius=%d "
        ,maxCornersPerBlock,invalidFlow,samplePointSize,cellSize,cellOverlay,viableSD,surroundRadius));
    if (k_CornerPoints.empty()){
      ROS_INFO("kernel CornerPoints failed to initialize!");
      return;
    }
  k_OptFlow = cv::ocl::Kernel("OptFlowReduced", *program,
      cv::format( "-D maxCornersPerBlock=%d -D invalidFlowVal=%d "
        "-D samplePointSize=%d -D outputFlowFieldSize=%d -D outputFlowFieldOverlay=%d "
        "-D firstStepBlockSize=%d -D surroundRadius=%d "
        ,maxCornersPerBlock,invalidFlow,samplePointSize,cellSize,cellOverlay,viableSD,surroundRadius));
    if (k_OptFlow.empty()){
      ROS_INFO("kernel OptFlowReduced failed to initialize!");
      return;
    }
  k_BordersSurround = cv::ocl::Kernel("BordersSurround", *program,
      cv::format( "-D maxCornersPerBlock=%d -D invalidFlowVal=%d "
        "-D samplePointSize=%d -D outputFlowFieldSize=%d -D outputFlowFieldOverlay=%d "
        "-D firstStepBlockSize=%d -D surroundRadius=%d "
        ,maxCornersPerBlock,invalidFlow,samplePointSize,cellSize,cellOverlay,viableSD,surroundRadius));
    if (k_BordersSurround.empty()){
      ROS_INFO("kernel BordersSurround failed to initialize!");
      return;
    }
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

  imPrev_g = imPrev.getUMat(cv::ACCESS_READ);
  imCurr_g = imCurr.getUMat(cv::ACCESS_READ);

  std::size_t gridA[3] = {(imPrev.cols)/(blockszX),(imPrev.rows)/(blockszY),1};
  std::size_t blockA[3] = {viableSD,viableSD,1};
  std::size_t globalA[3] = {gridA[0]*blockA[0],gridA[1]*blockA[1],1};
  std::size_t one[3] = {1,1,1};
  
  int d = cellSize-cellOverlay;
  std::size_t gridC[3] = {imCurr.cols/d,imCurr.rows/d,1};
  std::size_t blockC[3] = {1,1,1};
  std::size_t globalC[3] = {gridC[0]*blockC[0],gridC[1]*blockC[1],1};

  if (first){
    ROS_INFO("%dx%d",gridA[0],gridA[1]);

    foundPointsX_g      = cv::UMat(cv::Size(maxCornersPerBlock,gridA[1]*gridA[0]),CV_16UC1);
    foundPointsY_g      = cv::UMat(cv::Size(maxCornersPerBlock,gridA[1]*gridA[0]),CV_16UC1);
    foundPointsX_prev_g = cv::UMat(cv::Size(maxCornersPerBlock,gridA[1]*gridA[0]),CV_16UC1);
    foundPointsY_prev_g = cv::UMat(cv::Size(maxCornersPerBlock,gridA[1]*gridA[0]),CV_16UC1);
    foundPointsX_prev_g = cv::Scalar(0);
    foundPointsY_prev_g = cv::Scalar(0);
    foundPointsX_g = cv::Scalar(0);
    foundPointsY_g = cv::Scalar(0);
    activationMap_g = cv::UMat(cv::Size(gridC[0],gridC[1]),CV_16SC1);
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
      cv::ocl::KernelArg::ReadOnly(imCurr_g),
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
    std::size_t globalB[3] = {gridB[0]*blockB[0],1,1};

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

    
    int i;
    i = k_OptFlow.set(0, cv::ocl::KernelArg::PtrReadOnly(imCurr_g));
    i = k_OptFlow.set(i, cv::ocl::KernelArg::ReadOnly(imPrev_g));
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



    k_BordersSurround.args(
        cv::ocl::KernelArg::PtrWriteOnly(activationMap_g),
        cv::ocl::KernelArg::PtrReadOnly(activationMap_prev_g),
        cv::ocl::KernelArg::PtrWriteOnly(averageX_g),
        cv::ocl::KernelArg::WriteOnlyNoSize(averageY_g),
        cellFlowX_g,
        cellFlowY_g,
        cellFlowNum_g,
        gridC_width_g
        );

    begin = clock();

    k_BordersSurround.run(2,globalC,blockC,true);

    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    ROS_INFO("BordersSurround took %f seconds",elapsed_secs);
    
  }     

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

  foundPointsX_g.copyTo(foundPointsX_prev_g);
  foundPointsY_g.copyTo(foundPointsY_prev_g);
  activationMap_g.copyTo(activationMap_prev_g);



  cv::Mat imShowCorn = cv::Mat(imCurr.size(),imCurr.type());
    if (!first){
    imShowCorn_g.copyTo(imShowCorn);
    }
  if (debug)
  {
    // ROS_INFO("out: %dx%d",outX_l.cols,outX_l.rows);
  }
  if (gui)
  {
    if (false)
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
          averageY_g);
  }

  imPrev = imCurr.clone();

  first = false;
  return outvec;

}

void SparseOptFlowOcl::showFlow(
    const cv::UMat posx,
    const cv::UMat posy,
    const cv::UMat flowx,
    const cv::UMat flowy,
    bool blankBG,
    const cv::UMat actMap,
    const cv::UMat avgX,
    const cv::UMat avgY)
{
  cv::Mat out;

  drawOpticalFlow(posx.getMat(cv::ACCESS_READ), posy.getMat(cv::ACCESS_READ), flowx.getMat(cv::ACCESS_READ), flowy.getMat(cv::ACCESS_READ), blankBG, out );
  drawActivation(actMap.getMat(cv::ACCESS_READ),avgX.getMat(cv::ACCESS_READ),avgY.getMat(cv::ACCESS_READ));

  imView = ResizeToFitRectangle(imView,monitorSize);
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

