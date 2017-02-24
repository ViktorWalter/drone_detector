#define i1__at(x,y) input_1[(imgSrcOffset + x)+(y)*imgSrcWidth]
#define arraySize 50
#define MinValThreshold (ScanDiameter*ScanDiameter*0.5)
#define MaxAbsDiffThreshold (ScanDiameter*ScanDiameter*0.5)
#define FastThresh 30
#define CornerArraySize 10
#define maxNumOfBlocks 2000
#define errorShift -255


__kernel void CornerPoints_C1_D0(
    __global unsigned char* input_1,
    int imgSrcWidth,
    int imgSrcOffset,
    __global signed char* output_view,
    int showCornWidth,
    int showCornOffset,
    __global ushort* foundPointsX,
    __global ushort* foundPointsY,
    __global int* numFoundBlock,
    int foundPointsWidth,
    int foundPointsOffset,
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

  int maxij = blockSize*repetitions-3;
  numFoundBlock[blockY*blockNumX+blockX] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int m=0; m<repetitions; m++)
    for (int n=0; n<repetitions; n++)
    {
      int i = n*blockSize + threadX;
      int j = m*blockSize + threadY;

      if ((i>=3)&&(i<=maxij)&&(j>=3)&&(j<=maxij))
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
            //            if (indexLocal <= maxCornersPerBlock)
            {
              output_view[(blockY*(blockSize)+j)*showCornWidth + (showCornOffset+blockX*(blockSize)+i) ] = 30;
              foundPointsX[
                (blockY*blockNumX+blockX)*foundPointsWidth + foundPointsOffset + indexLocal] =
                  blockX*(blockSize)+i;
              foundPointsY[
                (blockY*blockNumX+blockX)*foundPointsWidth + foundPointsOffset + indexLocal] =
                  blockY*(blockSize)+j;
              foundPtsX_ord[foundPtsOrdOffset + indexGlobal] =
                blockX*(blockSize)+i;
              foundPtsY_ord[foundPtsOrdOffset + indexGlobal] =
                blockY*(blockSize)+j;
            }

          }
        }
      }
    }
  return;

  barrier(CLK_GLOBAL_MEM_FENCE);
  if ((blockX == 0) && (blockY == 0))
    if ((threadX == 0) && (threadY == 0))
    {
      for (int i = 0; i < blockNumX*blockNumY; i++) {
        int numFound =
          ((numFoundBlock[i]>maxCornersPerBlock) ? maxCornersPerBlock : numFoundBlock[i]);
        for (int j = 0; j < numFound; j++) {
          foundPtsX_ord[foundPtsOrdOffset + foundPtsSize[0]] =
            foundPointsX[i*foundPointsWidth+foundPointsOffset+j];
          foundPtsY_ord[foundPtsOrdOffset + foundPtsSize[0]] =
            foundPointsY[i*foundPointsWidth+foundPointsOffset+j];
          atomic_inc(&foundPtsSize[0]);
        }

      }

    }

  return;
}

__kernel void OptFlow_C1_D0(
    __global unsigned char* input_1,
    __global unsigned char* input_2,
    int imgSrcWidth,
    int imgSrcOffset,
    int imgSrcTrueWidth,
    int imgSrcTrueHeight,
    __global ushort* foundPtsX,
    __global ushort* foundPtsY,
    __global signed char* output_view,
    int showCornWidth,
    int showCornOffset,
    int ScanRadius,
    int samplePointSize
    )
{
  int block = get_group_id(0);
  int blockNum = get_num_groups(0);
  int threadX = get_local_id(0);
  int threadY = get_local_id(1);
  int threadDiameter = get_local_size(0);

  int ScanDiameter = ScanRadius*2+1;
  int corner = -samplePointSize/2;
  int posX = foundPtsX[block]; 
  int posY = foundPtsY[block]; 

  __local int abssum[arraySize][arraySize];

  int repetitions = ceil(ScanDiameter/(float)threadDiameter);

  barrier(CLK_LOCAL_MEM_FENCE);

  if ((posX >= (ScanRadius-corner))&&(posY >= (ScanRadius-corner))&&(posX<(imgSrcTrueWidth-(ScanRadius-corner)))&&(posY<(imgSrcTrueHeight-(ScanRadius-corner))))
  {
    for (int m=0; m<repetitions; m++)
    {
      for (int n=0; n<repetitions; n++)
      {
        int currXshift = n*threadDiameter + threadX;
        int currYshift = m*threadDiameter + threadY;

        if ((currXshift<ScanDiameter) && (currYshift<ScanDiameter))
        {
          abssum[currYshift][currXshift] = 0;

          for (int i=0;i<samplePointSize;i++)
          {
            for (int j=0;j<samplePointSize;j++)
            {
              atomic_add(&(abssum[currYshift][currXshift]),
                  abs(
                    input_1[
                    (imgSrcOffset + posX + i + corner)+
                    (posY + j + corner)*imgSrcWidth]
                    -
                    input_2[
                    (imgSrcOffset + posX + i + corner + currXshift - ScanRadius)+
                    (posY + j + corner + currYshift - ScanRadius)*imgSrcWidth]
                    )
                  );
            }

          }
        }
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      __local int minval[arraySize];
      __local signed char minX[arraySize];
      signed char minY;

      bool tester = false;
      if (threadY == 0)
      {
        for (int n=0; n<repetitions; n++)
        {
          int currXshift = n*threadDiameter + threadX;
          if (currXshift >= ScanDiameter)
            break;
          minval[currXshift] = abssum[currXshift][0];
          minX[currXshift] = -ScanRadius;
          for (int i=1;i<ScanDiameter;i++)
          {
            if (minval[currXshift] > abssum[currXshift][i])
            {
              minval[currXshift] = abssum[currXshift][i];
              minX[currXshift] = i-ScanRadius;
              tester = true;
            }
          }
        }
      }

      barrier(CLK_LOCAL_MEM_FENCE);
      int resX, resY;

      if ( (threadY == 0) && (threadX == 0))
      {

        int minvalFin = minval[0];
        minY = -ScanRadius;
        for (int i=1;i<ScanDiameter;i++)
        {
          if (minvalFin > minval[i])
          {
            minvalFin = minval[i];
            minY = i-ScanRadius;
            tester = true;
          }
        }
        resY = minY;
        resX = minX[minY+ScanRadius];
        //output_Y[blockY*imgDstWidth+imgDstOffset + blockX] = minY;
        //output_X[blockY*imgDstWidth+imgDstOffset + blockX] = minX[minY+scanRadius];

        if (((abssum[ScanRadius][ScanRadius] - minvalFin) <= MinValThreshold) && (false))  //if the difference is small, then it is considered to be noise in a uniformly colored area
        {
          resY = 0;
          resX = 0;
          // output_Y[blockY*imgDstWidth+imgSrcOffset+blockX] = 0;
          //output_X[blockY*imgDstWidth+imgSrcOffset+blockX] = 0;
        }

        output_view[(posY - resY)*showCornWidth+ (posX - resX)+showCornOffset ] = 255;

        if ((minvalFin) >= MaxAbsDiffThreshold)  //if the value is great, then it is considered to be too noisy, blurred or with too great a shift
        {
          resY = errorShift;
          resX = errorShift;
          output_view[(posY)*showCornWidth+ (posX)+showCornOffset ] = 100;
        }

      }

    }
  }
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
    __constant ushort* prevFoundBlockX,
    __constant ushort* prevFoundBlockY,
    int prevFoundBlockWidth,
    __constant int* prevFoundNum,
    __global signed char* output_view,
    int showCornWidth,
    int showCornOffset,
    int ScanRadius,
    int samplePointSize,
    int prevBlockSize,
    int prevBlockWidth
    )
{
  int block = get_group_id(0);
  int blockNum = get_num_groups(0);
  int threadI = get_local_id(0);
  int threadNum = get_local_size(0);

  int ScanDiameter = ScanRadius*2+1;
  int corner = -samplePointSize/2;
  int posX = foundPtsX[block]; 
  int posY = foundPtsY[block]; 

  __local int abssum[arraySize*arraySize];
  __local int Xpositions[arraySize*arraySize];
  __local int Ypositions[arraySize*arraySize];

  int currBlockX = posX/prevBlockSize;
  int currBlockY = posY/prevBlockSize;

  barrier(CLK_LOCAL_MEM_FENCE);

  int blockShiftX = -1;
  int blockShiftY = -1;
  int colNum = 0;
  int lineNum = (currBlockY + blockShiftY)*prevBlockWidth + (currBlockX + blockShiftX);
  int pointsHeld = 0;

  while (blockShiftY <= 1) {
    int consideredX = prevFoundBlockX[lineNum*prevFoundBlockWidth+colNum];
    int consideredY = prevFoundBlockY[lineNum*prevFoundBlockWidth+colNum];

    if ((consideredX >= (-corner))&&(consideredY >= (-corner))&&(consideredX<(imgSrcTrueWidth-(-corner)))&&(consideredY<(imgSrcTrueHeight-(-corner)))){
      Xpositions[pointsHeld] = consideredX;
      Ypositions[pointsHeld] = consideredY;
      pointsHeld++;
    }
    if (colNum == prevFoundNum[lineNum]) {
      colNum = 0;
      if (blockShiftX == 1) {
        blockShiftX = -1;
        blockShiftY++;
      }
      else {
        blockShiftX++;
      }
      lineNum = (currBlockY + blockShiftY)*prevBlockWidth + (currBlockX + blockShiftX);
    }
    else {
      colNum++;
    }
  }

  int repetitions = ceil(pointsHeld/(float)threadNum);
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int n = 0; n < repetitions; n++) {
    int indexLocal = n*threadNum + threadI;
    if (indexLocal < pointsHeld) {
      int posX_prev = Xpositions[indexLocal];
      int posY_prev = Ypositions[indexLocal];
      abssum[indexLocal] = 0;
      for (int i=0;i<samplePointSize;i++) {
        for (int j=0;j<samplePointSize;j++) {
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
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  int resX, resY;

  if (threadI == 0)
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

    if (((abssum[minI] - minval) <= MaxAbsDiffThreshold) && (false))  //if the difference is small, then it is considered to be noise in a uniformly colored area
    {
      resY = 0;
      resX = 0;
    }

    output_view[(resY)*showCornWidth+ (resX)+showCornOffset ] = 255;

    if ((minval) >= MinValThreshold)  //if the value is great, then it is considered to be too noisy, blurred or with too great a shift
    {
      resY = errorShift;
      resX = errorShift;
      output_view[(posY)*showCornWidth+ (posX)+showCornOffset ] = 100;
    }

  }


return;
/*

__local int minval[arraySize];
__local signed char minX[arraySize];
signed char minY;

bool tester = false;
if (threadY == 0)
{
  for (int n=0; n<repetitions; n++)
  {
    int currXshift = n*threadDiameter + threadX;
    if (currXshift >= ScanDiameter)
      break;
    minval[currXshift] = abssum[currXshift][0];
    minX[currXshift] = -ScanRadius;
    for (int i=1;i<ScanDiameter;i++)
    {
      if (minval[currXshift] > abssum[currXshift][i])
      {
        minval[currXshift] = abssum[currXshift][i];
        minX[currXshift] = i-ScanRadius;
        tester = true;
      }
    }
  }
}

barrier(CLK_LOCAL_MEM_FENCE);
*/
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



