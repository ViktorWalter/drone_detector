#define i1__at(x,y) input_1[(imgSrcOffset + x)+(y)*imgSrcWidth]
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

__kernel void CornerPoints_C1_D0(
    __global unsigned char* input_1,
    int imgSrcWidth,
    int imgSrcOffset,
    int imgSrcTrueWidth,
    __global signed char* output_view,
    int showCornWidth,
    int showCornOffset,
    __global int* foundPointsX,
    __global int* foundPointsY,
    __global int* numFoundBlock,
    __global ushort* foundPtsX_ord,
    __global ushort* foundPtsY_ord,
    __global int* foundPtsSize,
    int foundPtsOrdOffset,
    int maxCornersPerBlock
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
      if ((x>=3)&&(x<=imgSrcTrueWidth-3)&&(y>=3)&&(y<=imgSrcTrueWidth-3))
      {
        uchar I[17];
        I[0] = i1__at(blockX*(blockSize)+i,blockY*(blockSize)+j);
        I[1] = i1__at(blockX*(blockSize)+i,blockY*(blockSize)+(j-3));
        I[9] = i1__at(blockX*(blockSize)+i,blockY*(blockSize)+(j+3));

        if (((I[1]>(I[0]-FastThresh)) && (I[1]<(I[0]+FastThresh))) && (I[9]>(I[0]-FastThresh)) && (I[9]<(I[0]+FastThresh)))
          return;
        I[5] = i1__at(blockX*(blockSize)+(i+3),blockY*(blockSize)+j);
        I[13] = i1__at(blockX*(blockSize)+(i-3),blockY*(blockSize)+j);
        char l= (I[1]>(I[0]+FastThresh))+(I[9]>(I[0]+FastThresh))+(I[5]>(I[0]+FastThresh))+(I[13]>(I[0]+FastThresh));
        char h= (I[1]<(I[0]-FastThresh))+(I[9]<(I[0]-FastThresh))+(I[5]<(I[0]-FastThresh))+(I[13]<(I[0]+FastThresh));
        if ( (l>3) || (h>3) )
        {
          char sg = 1;
          if (l >3)
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
              output_view[(blockY*(blockSize)+j)*showCornWidth + (showCornOffset+blockX*(blockSize)+i) ] = 30;
              foundPointsX[
                (blockY*blockNumX+blockX)*maxCornersPerBlock + indexLocal]=
                  blockX*(blockSize)+i;
              foundPointsY[
                (blockY*blockNumX+blockX)*maxCornersPerBlock + indexLocal]=
                  blockY*(blockSize)+j;
            }
            foundPtsX_ord[foundPtsOrdOffset + indexGlobal] =
              blockX*(blockSize)+i;
            foundPtsY_ord[foundPtsOrdOffset + indexGlobal] =
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
//
//  barrier(CLK_GLOBAL_MEM_FENCE);
//  if ((blockX == 0) && (blockY == 0))
//    if ((threadX == 0) && (threadY == 0))
//    {
//      for (int i = 0; i < blockNumX*blockNumY; i++) {
//        int numFound =
//          ((numFoundBlock[i]>maxCornersPerBlock) ? maxCornersPerBlock : numFoundBlock[i]);
//        for (int j = 0; j < numFound; j++) {
//          foundPtsX_ord[foundPtsOrdOffset + foundPtsSize[0]] =
//            foundPointsX[i*foundPointsWidth+foundPointsOffset+j];
//          foundPtsY_ord[foundPtsOrdOffset + foundPtsSize[0]] =
//            foundPointsY[i*foundPointsWidth+foundPointsOffset+j];
//          atomic_inc(&foundPtsSize[0]);
//        }
//
//      }
//
//    }
//
//  return;
}


__kernel void OptFlowReduced_C1_D0(
    __global unsigned char* input_1,
    __global unsigned char* input_2,
    int imgSrcWidth,
    int imgSrcOffset,
    int imgSrcTrueWidth,
    int imgSrcTrueHeight,
    __global ushort* foundPtsX,
    __global ushort* foundPtsY,
    __global int* prevFoundBlockX,
    __global int* prevFoundBlockY,
    __constant int* prevFoundNum,
    int maxCornersPerBlock,
    __global signed char* output_view,
    int showCornWidth,
    int showCornOffset,
    int samplePointSize,
    int prevBlockSize,
    int prevBlockWidth,
    int prevBlockHeight,
    __global ushort* outputOffsetX,
    __global ushort* outputOffsetY,
    int invalidFlowVal

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

  int currBlockX = posX/prevBlockSize;
  int currBlockY = posY/prevBlockSize;
  int currLine = (currBlockY)*prevBlockWidth + (currBlockX);

//  float maxdist2 = (prevBlockSize*prevBlockSize*maxDistMultiplier*maxDistMultiplier);

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
      consideredX = prevFoundBlockX[lineNum*maxCornersPerBlock+colNum];
      consideredY = prevFoundBlockY[lineNum*maxCornersPerBlock+colNum];
      if (pointsHeld > 0)
        if ((consideredX == Xpositions[pointsHeld-1]) && (consideredY == Ypositions[pointsHeld-1])){
          watchdog = true;
          break;
        }


      if (
          (consideredX >= (-corner))
          &&(consideredY >= (-corner))
          &&(consideredX<(imgSrcTrueWidth+corner))
          &&(consideredY<(imgSrcTrueHeight+corner))
          &&(consideredX != excludedPoint)
         ){
      //  int dx = posX - consideredX;
      //  int dy = posY - consideredY;
      //  float dist2 = dx*dx+dy*dy;
      //  if (dist2 <= maxdist2)
        {
          Xpositions[pointsHeld] = consideredX;
          Ypositions[pointsHeld] = consideredY;
          Indices[pointsHeld] = lineNum*maxCornersPerBlock+colNum;
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

    //    if ((resX > imgSrcTrueWidth) || (resX < 0))
    //      return;
    //    if ((resY > imgSrcTrueHeight) || (resY < 0))
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

    if (watchdog)
    {
      resX = 8000;
      resY = 8000;
    }
    
    if (resX != invalidFlowVal) {
      prevFoundBlockX[Indices[minI]] = excludedPoint;
      prevFoundBlockY[Indices[minI]] = excludedPoint;
    }

    outputOffsetX[block] = resX;
    outputOffsetY[block] = resY;

  }


  return;
}



__kernel void Histogram_C1_D0(__constant signed char* inputX,
    __global signed char* inputY,
    int width,
    int offset,
    int scanRadius,
    int ScanDiameter,
    __global signed char* valueX,
    __global signed char* valueY,
    int TestDepth,
    __global signed char* outVecX,
    __global signed char* outVecY
    )
{

  int threadX = get_local_id(0);
  int threadY = get_local_id(1);

  int totalblockSize = get_local_size(0)*get_local_size(1);
  int HistSize = ScanDiameter;

  __local int HistogramX[arraySize];
  __local int HistogramY[arraySize];
  __local int HistIndexX[arraySize];
  __local int HistIndexY[arraySize];

  int imageIndex = (threadY*width+threadX+offset);
  int threadIndex = (threadY*get_local_size(0) + threadX);

  for (int i=0; ; i++)
  {
    int HistIndex = threadIndex + (i*totalblockSize);
    if (HistIndex > HistSize)
    {
      break;
    }

    HistogramX[HistIndex] = 0;
    HistogramY[HistIndex] = 0;
    HistIndexX[HistIndex] = HistIndex - scanRadius;
    HistIndexY[HistIndex] = HistIndex - scanRadius;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  int HistLocX=(inputX[imageIndex])+scanRadius;
  int HistLocY=(inputY[imageIndex])+scanRadius;

  atomic_add(&(HistogramX[HistLocX]),1);
  atomic_add(&(HistogramY[HistLocY]),1);

  barrier(CLK_LOCAL_MEM_FENCE);

  bool swapped;
  if (threadIndex == 0)
  {
    do
    {
      swapped = false;
      for (int i=1;i<HistSize;i++)
      {
        if (HistogramX[i] > HistogramX[i-1])
        {
          HistogramX[i] = atomic_xchg(&(HistogramX[i-1]),HistogramX[i]);
          HistIndexX[i] = atomic_xchg(&(HistIndexX[i-1]),HistIndexX[i]);
          swapped = true;
        }
      }
    } while (swapped == true);
  }

  if (threadIndex == 1)
  {
    do
    {
      swapped = false;
      for (int i=1;i<HistSize;i++)
      {
        if (HistogramY[i] > HistogramY[i-1])
        {
          HistogramY[i] = atomic_xchg(&(HistogramY[i-1]),HistogramY[i]);
          HistIndexY[i] = atomic_xchg(&(HistIndexY[i-1]),HistIndexY[i]);
          swapped = true;
        }
      }
    } while (swapped == true);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  *valueX = HistIndexX[0];
  *valueY = HistIndexY[0];


  if (threadIndex == 0)
  {
    int outIndex = 0;
    for (int i=0; i<TestDepth; i++)
    {
      for (int j=0; j<TestDepth; j++)
      {
        outVecX[outIndex] = HistIndexX[i];
        outVecY[outIndex] = HistIndexY[j];
        outIndex++;
      }
    }
  }



}



