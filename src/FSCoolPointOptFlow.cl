#define i1__at(x,y) input_1[(imgSrcOffset + x)+(y)*imgSrcStep]
#define arraySize 50
#define MinValThreshold (samplePointSize2*20)//*1*prevFoundNum[currLine])
#define MaxAbsDiffThreshold (samplePointSize2*10)
#define FastThresh 30
#define CornerArraySize 10
#define maxNumOfBlocks 2000
#define shiftRadius 1
#define maxDistMultiplier 1.5
#define threadsPerCornerPoint 32
#define distanceWeight (samplePointSize2*0.05)
#define excludedPoint -1
#define minPointsThreshold 4

__kernel void CornerPoints(
    __global uchar* input_1,
    int imgSrcStep,
    int imgSrcOffset,
    int imgSrcHeight,
    int imgSrcWidth,
    __global uchar* output_view,
    int showCornStep,
    int showCornOffset,
    __global ushort* foundPointsX,
    __global ushort* foundPointsY,
    int foundPointsStep,
    int foundPointsOffset,
    int foundPointsHeight,
    int foundPointsWidth,
    __global ushort* foundPtsX_ord,
    __global ushort* foundPtsY_ord,
    __global int* numFoundBlock,
    __global int* foundPtsSize
    )
{
  int blockX = get_group_id(0);
  int blockY = get_group_id(1);
  int blockNumX = get_num_groups(0);
  int blockNumY = get_num_groups(1);
  int threadX = get_local_id(0);
  int threadY = get_local_id(1);
  int blockSize = get_local_size(0);



  int repetitions = 1; //ceil(ScanDiameter/(float)threadDiameter);

 // int maxij = blockSize*repetitions-3;
  numFoundBlock[blockY*blockNumX+blockX] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int m=0; m<repetitions; m++)
    for (int n=0; n<repetitions; n++)
    {
      int i = n*blockSize + threadX;
      int j = m*blockSize + threadY;
      int x = blockX*blockSize+i;
      int y = blockY*blockSize+i;
      //if ((i>=3)&&(i<=maxij)&&(j>=3)&&(j<=maxij))
      if ((x>=3)&&(x<=imgSrcStep-3)&&(y>=3)&&(y<=imgSrcStep-3))
      {
        uchar I[17];
        I[0] = i1__at(blockX*(blockSize)+i,blockY*(blockSize)+j);
        I[1] = i1__at(blockX*(blockSize)+i,blockY*(blockSize)+(j-3));
        I[9] = i1__at(blockX*(blockSize)+i,blockY*(blockSize)+(j+3));

        if (((I[1]>(I[0]-FastThresh)) && (I[1]<(I[0]+FastThresh)))
            &&
            ((I[9]>(I[0]-FastThresh)) && (I[9]<(I[0]+FastThresh))))
          return;
        I[5] = i1__at(blockX*(blockSize)+(i+3),blockY*(blockSize)+j);
        I[13] = i1__at(blockX*(blockSize)+(i-3),blockY*(blockSize)+j);
        char l=
          (I[1]>(I[0]+FastThresh))+(I[9]>(I[0]+FastThresh))+
          (I[5]>(I[0]+FastThresh))+(I[13]>(I[0]+FastThresh));
        char h=
          (I[1]<(I[0]-FastThresh))+(I[9]<(I[0]-FastThresh))+
          (I[5]<(I[0]-FastThresh))+(I[13]<(I[0]-FastThresh));

//          foundPtsSize[0] =blockNumY;
//          return;

        if ( (l>3) || (h>3) )
        {
          char sg = 1;
          if (l >=3)
            sg = -1;
          I[2] = i1__at(blockX*(blockSize)+(i+1),blockY*(blockSize)+(j-3));
          I[3] = i1__at(blockX*(blockSize)+(i+2),blockY*(blockSize)+(j-2));
          I[4] = i1__at(blockX*(blockSize)+(i+3),blockY*(blockSize)+(j-1));
          I[6] = i1__at(blockX*(blockSize)+(i+3),blockY*(blockSize)+(j+1));
          I[7] = i1__at(blockX*(blockSize)+(i+2),blockY*(blockSize)+(j+2));
          I[8] = i1__at(blockX*(blockSize)+(i+1),blockY*(blockSize)+(j+3));
          I[10] = i1__at(blockX*(blockSize)+(i-1),blockY*(blockSize)+(j+3));
          I[11] = i1__at(blockX*(blockSize)+(i-2),blockY*(blockSize)+(j+2));
          I[12] = i1__at(blockX*(blockSize)+(i-3),blockY*(blockSize)+(j+1));
          I[14] = i1__at(blockX*(blockSize)+(i-3),blockY*(blockSize)+(j-1));
          I[15] = i1__at(blockX*(blockSize)+(i-2),blockY*(blockSize)+(j-2));
          I[16] = i1__at(blockX*(blockSize)+(i-1),blockY*(blockSize)+(j-3));
          int cadj = 0;
          int cadj_b = 0;
          for (int pix = 1; pix < 16; pix++) {
            if (sg == 1) {
              if (I[pix]<(I[0]-FastThresh)) {
                cadj++; 
                if (cadj == 12) {
                  break;
                }
              }
              else {
                if (cadj == pix-1) {
                  cadj_b = cadj; 
                }
                cadj = 0;
              }
            }
            else {
              if (I[pix]>(I[0]+FastThresh)) {
                cadj++; 
                if (cadj == 12) {
                  break;
                }
              }
              else {
                if (cadj == pix-1) {
                  cadj_b = cadj; 
                }
                cadj = 0;
              }
            }
          }
          if ( (cadj + cadj_b) >= 12) {
            int indexLocal;
            int indexGlobal;
            indexLocal = atomic_inc(&(numFoundBlock[blockY*blockNumX+blockX]));
            indexGlobal = atomic_inc(&(foundPtsSize[0]));
            if (indexLocal <= maxCornersPerBlock)
            {
              output_view[
              (blockY*(blockSize)+j)*showCornStep +
                (showCornOffset+blockX*(blockSize)+i) ] =
                30;
              foundPointsX[
                (blockY*blockNumX+blockX)*foundPointsStep + indexLocal]=
                  blockX*(blockSize)+i;
              foundPointsY[
                (blockY*blockNumX+blockX)*foundPointsStep + indexLocal]=
                  blockY*(blockSize)+j;
            }
            foundPtsX_ord[indexGlobal] =
              blockX*(blockSize)+i;
            foundPtsY_ord[indexGlobal] =
              blockY*(blockSize)+j;
            barrier(CLK_LOCAL_MEM_FENCE);
            if (indexLocal >= maxCornersPerBlock){
              numFoundBlock[blockY*blockNumX+blockX] = maxCornersPerBlock;
            }
          }


        }
      }
    }
  return;
}


__kernel void OptFlowReduced(
    __global unsigned char* input_1,
    __global unsigned char* input_2,
    int imgSrcStep,
    int imgSrcOffset,
    int imgSrcHeight,
    int imgSrcWidth,
    __global ushort* foundPtsX,
    __global ushort* foundPtsY,
    __global short* prevFoundBlockX,
    __global short* prevFoundBlockY,
    int prevFoundBlockStep,
    int prevFoundBlockOffset,
    __constant int* prevFoundNum,
    __global signed char* output_view,
    int showCornWidth,
    int showCornOffset,
    int prevBlockWidth,
    int prevBlockHeight,
    __global ushort* outputPosOrdX,
    __global ushort* outputPosOrdY,
    __global int* outputFlowBlockX,
    __global int* outputFlowBlockY,
    __global int* outputFlowBlockNum,
    int outputFlowBlockWidth
    )
{
  int block = get_group_id(0);
  int blockNum = get_num_groups(0);
  int threadX = get_local_id(0);
  int threadY = get_local_id(1);
  int threadNumX = get_local_size(0);
  int threadNumY = get_local_size(1);

  int corner = -samplePointSize/2;
  int posX = foundPtsX[block]; 
  int posY = foundPtsY[block]; 

  __local int abssum[arraySize*arraySize];
  __local int Xpositions[arraySize*arraySize];
  __local int Ypositions[arraySize*arraySize];
  __local int Indices[arraySize*arraySize];

  int samplePointSize2 = samplePointSize*samplePointSize;

  int currBlockX = posX/firstStepBlockSize;
  int currBlockY = posY/firstStepBlockSize;
  int currLine = (currBlockY)*prevBlockWidth + (currBlockX);
  int d = outputFlowFieldSize - outputFlowFieldOverlay;

//  float maxdist2 = (firstStepBlockSize*firstStepBlockSize*maxDistMultiplier*maxDistMultiplier);

  barrier(CLK_LOCAL_MEM_FENCE);
  //First gather up the previous corner points that are in the vicinity
  int blockShiftX = -shiftRadius;
  int blockShiftY = -shiftRadius;
  int colNum = 0;
  int lineNum;
  int pointsHeld = 0;

  bool watchdog = false;
  int counter = 0;
  
  while ((blockShiftY <= shiftRadius)) {
    if (( blockShiftX > shiftRadius) || ( blockShiftY > shiftRadius) )
      break;

    counter++;
    if (counter == 720) {
      watchdog = true;
      break;
    }

    if ((currBlockX + blockShiftX) < 0) {
      blockShiftX++;
      continue;
    }
    if ((currBlockX + blockShiftX) >= prevBlockWidth) {
      blockShiftY++;
      blockShiftX = -shiftRadius;
      continue;
    }
    if ((currBlockY + blockShiftY) < 0) {
      blockShiftY++;
      continue;
    }
    if ((currBlockY + blockShiftY) >= prevBlockHeight) {
      break;
    }

    lineNum = (currBlockY + blockShiftY)*prevBlockWidth + (currBlockX + blockShiftX);
    if (prevFoundNum[lineNum] > 0) {
      int consideredX;
      int consideredY;
      consideredX = prevFoundBlockX[lineNum*prevFoundBlockStep+colNum];
      consideredY = prevFoundBlockY[lineNum*prevFoundBlockStep+colNum];
      if (pointsHeld > 0)
        if ((consideredX == Xpositions[pointsHeld-1]) && (consideredY == Ypositions[pointsHeld-1])){
          watchdog = true;
          break;
        }


      if (
          (consideredX >= (-corner))
          &&(consideredY >= (-corner))
          &&(consideredX<(imgSrcWidth+corner))
          &&(consideredY<(imgSrcHeight+corner))
         ){
      //  int dx = posX - consideredX;
      //  int dy = posY - consideredY;
      //  float dist2 = dx*dx+dy*dy;
      //  if (dist2 <= maxdist2)
        {
          Xpositions[pointsHeld] = consideredX;
          Ypositions[pointsHeld] = consideredY;
          Indices[pointsHeld] = lineNum*prevFoundBlockStep+colNum;
          pointsHeld++;
        }
      }
      colNum++;
    }
    if (colNum > (prevFoundNum[lineNum]-1)) {
      colNum = 0;
      if (blockShiftX < shiftRadius) {
        blockShiftX++;
      }
      else { 
        blockShiftX = -shiftRadius;
        blockShiftY++;
      }
    }
  }

  //add current position in the previous image to consideration
  Xpositions[pointsHeld] = posX;
  Ypositions[pointsHeld] = posY;
  pointsHeld++;


  int repetitionsOverCorners = ceil(pointsHeld/(float)threadNumX);
  int repetitionsOverPixels = ceil(samplePointSize2/(float)threadNumY);
  barrier(CLK_LOCAL_MEM_FENCE);
  //Next, Check each of them for match
  for (int n = 0; n < repetitionsOverCorners; n++) {
    int indexLocal = n*threadNumX + threadX;
      abssum[indexLocal] = 0;
      barrier(CLK_LOCAL_MEM_FENCE);
    if (indexLocal < pointsHeld) {
      int posX_prev = Xpositions[indexLocal];
      int posY_prev = Ypositions[indexLocal];
      if (threadY == 0) {
        int dx = Xpositions[indexLocal] - posX;
        int dy = Ypositions[indexLocal] - posY;
        int distPenalty = distanceWeight*(dx*dx+dy*dy);// (int)(distanceWeight*sqrt((float)(dx*dx+dy*dy)));
        atomic_add(&abssum[indexLocal],distPenalty);
      }
      for (int m=0;m<repetitionsOverPixels;m++) {
        int indexPixel = n*threadNumY+threadY;
        int i = indexPixel % samplePointSize;
        int j = indexPixel / samplePointSize;
        atomic_add(&(abssum[indexLocal]),
            abs(
              input_1[
              (imgSrcOffset + posX + i + corner)+
              (posY + j + corner)*imgSrcWidth]
              -
              input_2[
              (imgSrcOffset + posX_prev + i + corner)+
              (posY_prev + j + corner)*imgSrcWidth]
                )
              );
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (pointsHeld == 0)
    return;


  int resX, resY;

  if ((threadX == 0) && (threadY  == 0) )
  {

    int minval = abssum[0];
    int minI = 0; 

    for (int i=1;i<pointsHeld;i++)
    {
      if (minval > abssum[i])
      {
        minval = abssum[i];
        minI = i;
      }
    }

    resX = Xpositions[minI];
    resY = Ypositions[minI];

    //    if ((resX > imgSrcWidth) || (resX < 0))
    //      return;
    //    if ((resY > imgSrcHeight) || (resY < 0))
    //      return;
    if ( (minI < (pointsHeld-1)) && ((abssum[pointsHeld-1] - minval) <= MaxAbsDiffThreshold))  //if the difference is small, then it is considered to be noise in a uniformly colored area
      {
     //   resX = Xpositions[pointsHeld-1];
     //   resY = Ypositions[pointsHeld-1];
      resY = invalidFlowVal;
      resX = invalidFlowVal;
      output_view[(posY)*showCornWidth+ (posX)+showCornOffset ] = 100;
      }
    else if ( ((minval) >= MinValThreshold) && (true))  //if the value is great, then it is considered to be too noisy, blurred or with too great a shift
    {
      resY = invalidFlowVal;
      resX = invalidFlowVal;
      output_view[(posY)*showCornWidth+ (posX)+showCornOffset ] = 100;
    }
    else
      output_view[(resY)*showCornWidth+ (resX)+showCornOffset ] = 255;

    if (resX != invalidFlowVal) {
    //  prevFoundBlockX[Indices[minI]] = excludedPoint;
    //  prevFoundBlockY[Indices[minI]] = excludedPoint;
      
      int Ix = posX / d;
      int Iy = posY / d;
      bool edgeX = ((posX%d)<outputFlowFieldOverlay);
      bool edgeY = ((posY%d)<outputFlowFieldOverlay);
      if (Ix==0)
        edgeX = false;
      if (Iy==0)
        edgeY = false;
      
      if (edgeX && edgeY) {
        for (int i = -1; i < 0; i++) {
          for (int j = -1; j < 0; j++) {
            atomic_add(&outputFlowBlockX[(Iy+j)*outputFlowBlockWidth+(Ix+i)],(posX - resX));
            atomic_add(&outputFlowBlockY[(Iy+j)*outputFlowBlockWidth+(Ix+i)],(posY - resY));
            atomic_inc(&outputFlowBlockNum[(Iy+j)*outputFlowBlockWidth+(Ix+i)]);
          }
        }
      }
      else if (edgeX) {
        for (int i = -1; i < 0; i++) {
          atomic_add(&outputFlowBlockX[(Iy)*outputFlowBlockWidth+(Ix+i)],(posX - resX));
          atomic_add(&outputFlowBlockY[(Iy)*outputFlowBlockWidth+(Ix+i)],(posY - resY));
          atomic_inc(&outputFlowBlockNum[(Iy)*outputFlowBlockWidth+(Ix+i)]);
        }
      }
      else if (edgeY){
        for (int j = -1; j < 0; j++) {
          atomic_add(&outputFlowBlockX[(Iy+j)*outputFlowBlockWidth+(Ix)],(posX - resX));
          atomic_add(&outputFlowBlockY[(Iy+j)*outputFlowBlockWidth+(Ix)],(posY - resY));
          atomic_inc(&outputFlowBlockNum[(Iy+j)*outputFlowBlockWidth+(Ix)]);
        }
      }
      else {
        atomic_add(&outputFlowBlockX[(Iy)*outputFlowBlockWidth+(Ix)],(posX - resX));
        atomic_add(&outputFlowBlockY[(Iy)*outputFlowBlockWidth+(Ix)],(posY - resY));
        atomic_inc(&outputFlowBlockNum[(Iy)*outputFlowBlockWidth+(Ix)]);
      }

    }

    outputPosOrdX[block] = resX;
    outputPosOrdY[block] = resY;
  }

  return;
}


__kernel void BordersSurround(
    __global short* outA,
    __global short* outB,
    __global short* outC,
    int outStep,
    int outOffset,
    __global int* inX,
    __global int* inY,
    __global int* inNum,
    int inStep
    )
{
  int blockX = get_group_id(0);
  int blockY = get_group_id(1);
  int blockNumX = get_num_groups(0);
  int threadX = get_local_id(0);
  int threadY = get_local_id(1);
  int threadNumX = get_local_size(0);
  int threadNumY = get_local_size(1);

  int currIndexOutput = blockY*(outStep/sizeof(ushort))+blockX+outOffset/sizeof(ushort);
  int currIndexCenter = blockY*inStep+blockX;
  int currIndexSurr;

  if (inNum[currIndexCenter] < minPointsThreshold) {
    outA[currIndexOutput] = 0;
    outB[currIndexOutput] = 0;
    outC[currIndexOutput] = 0;
    return;
  }

  float avgOutX = 0;
  float avgOutY = 0;
  int cntOut = 0;
  float avgInX = inX[currIndexCenter]/(float)inNum[currIndexCenter];
  float avgInY = inY[currIndexCenter]/(float)inNum[currIndexCenter];

  for (int i = -surroundRadius; i <= surroundRadius; i++) {
    for (int j = -surroundRadius; j <= surroundRadius; j++) {
      if ((i!=0)||(j!=0)){
        currIndexSurr = (blockY+j)*inStep+(blockX+i);
        if (inNum[currIndexSurr] != 0){
          avgOutX += inX[currIndexSurr]; 
          avgOutY += inY[currIndexSurr]; 
          cntOut += inNum[currIndexSurr];
        }
      }
    }
  }

  if (cntOut == 0) {
    avgOutX = 0;
    avgOutY = 0;
  }
  else {
    avgOutX = avgOutX/cntOut;
    avgOutY = avgOutY/cntOut;
  }


  float dx = avgOutX - avgInX;
  float dy = avgOutY - avgInY;
  outA[currIndexOutput] = (short)sqrt(dx*dx+dy*dy);//*(cntOut/inNum[currIndexCenter]);
  outB[currIndexOutput] = (short)avgInX;
  outC[currIndexOutput] = (short)avgInY;
//  int blockShiftX = -shiftRadius;
//  int blockShiftY = -shiftRadius;
//  int colNum = 0;
//  int lineNum;
//  int pointsHeld = 0;
//  
//  while ((blockShiftY <= shiftRadius)) {
//    if (( blockShiftX > shiftRadius) || ( blockShiftY > shiftRadius) )
//      break;
//
//
//    if ((currBlockX + blockShiftX) < 0) {
//      blockShiftX++;
//      continue;
//    }
//    if ((currBlockX + blockShiftX) >= prevBlockWidth) {
//      blockShiftY++;
//      blockShiftX = -shiftRadius;
//      continue;
//    }
//    if ((currBlockY + blockShiftY) < 0) {
//      blockShiftY++;
//      continue;
//    }
//    if ((currBlockY + blockShiftY) >= prevBlockHeight) {
//      break;
//    }
//
//    lineNum = (currBlockY + blockShiftY)*prevBlockWidth + (currBlockX + blockShiftX);
//    if (prevFoundNum[lineNum] > 0) {
//      int consideredX;
//      int consideredY;
//      consideredX = prevFoundBlockX[lineNum*prevFoundBlockWidth+colNum];
//      consideredY = prevFoundBlockY[lineNum*prevFoundBlockWidth+colNum];
//      if (pointsHeld > 0)
//        if ((consideredX == Xpositions[pointsHeld-1]) && (consideredY == Ypositions[pointsHeld-1])){
//          watchdog = true;
//          break;
//        }
//
//
//      if (
//          (consideredX >= (-corner))
//          &&(consideredY >= (-corner))
//          &&(consideredX<(imgSrcWidth+corner))
//          &&(consideredY<(imgSrcHeight+corner))
//          &&(consideredX != excludedPoint)
//         ){
//      //  int dx = posX - consideredX;
//      //  int dy = posY - consideredY;
//      //  float dist2 = dx*dx+dy*dy;
//      //  if (dist2 <= maxdist2)
//        {
//          Xpositions[pointsHeld] = consideredX;
//          Ypositions[pointsHeld] = consideredY;
//          Indices[pointsHeld] = lineNum*prevFoundBlockWidth+colNum;
//          pointsHeld++;
//        }
//      }
//      colNum++;
//    }
//    if (colNum > (prevFoundNum[lineNum]-1)) {
//      colNum = 0;
//      if (blockShiftX < shiftRadius) {
//        blockShiftX++;
//      }
//      else { 
//        blockShiftX = -shiftRadius;
//        blockShiftY++;
//      }
//    }
//  }
}

__kernel void BordersGlobal_C1_D0(
    )
{
}

__kernel void BordersHeading_C1_D0(
    )
{
}

__kernel void Tester(__global uchar* input,int step,int offset)
{
  input[get_global_id(0)+offset+step*get_global_id(1)] = (get_global_id(0)+get_global_id(1))%256;
}


