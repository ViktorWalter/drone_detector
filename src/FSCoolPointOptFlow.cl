#define i1__at(x,y) input_1[(imgSrcOffset + x)+(y)*imgSrcWidth]
#define arraySize 50
#define MinValThreshold (ScanDiameter*ScanDiameter*0.1)
#define MaxAbsDiffThreshold (ScanDiameter*ScanDiameter*0.5)
#define FastThresh 30
#define CornerArraySize 10

__kernel void OptFlow_C1_D0(	__global unsigned char* input_1,
                                __global unsigned char* input_2,
                                int imgSrcWidth,
                                int imgSrcOffset,
                                int imgDstWidth,
                                int imgDstOffset,
                                __global signed char* output_X,
                                __global signed char* output_Y,
                                int showCornWidth,
                                __global signed char* output_view,
                                int blockSize,
                                int blockStep,
                                int scanRadius)
{
        int blockX = get_group_id(0);
        int blockY = get_group_id(1);
        int threadX = get_local_id(0);
        int threadY = get_local_id(1);

        if ((blockX) >= imgDstWidth)
            return;

        int ScanDiameter = scanRadius*2+1;
        __local int abssum[arraySize][arraySize];

        int threadDiameter = get_local_size(0);
        int repetitions = ceil(ScanDiameter/(float)threadDiameter);

        int maxij = threadDiameter*repetitions-3;
        for (int m=0; m<repetitions; m++)
            for (int n=0; n<repetitions; n++)
            {
                int i = n*threadDiameter + threadX;
                int j = m*threadDiameter + threadY;

                if ((i>=3)&&(i<=maxij)&&(j>=3)&&(j<=maxij))
                {
                                 uchar I[17];
                                 I[0] = i1__at(blockX*(blockSize+blockStep)+i,blockY*(blockSize+blockStep)+j);
                                 I[1] = i1__at(blockX*(blockSize+blockStep)+i,blockY*(blockSize+blockStep)+(j-3));
                                 I[9] = i1__at(blockX*(blockSize+blockStep)+i,blockY*(blockSize+blockStep)+(j+3));
                                 if (((I[1]>(I[0]-FastThresh)) && (I[1]<(I[0]+FastThresh))) && (I[9]>(I[0]-FastThresh)) && (I[9]<(I[0]+FastThresh)))
                                   continue;
                                 I[5] = i1__at(blockX*(blockSize+blockStep)+(i+3),blockY*(blockSize+blockStep)+j);
                                 I[13] = i1__at(blockX*(blockSize+blockStep)+(i-3),blockY*(blockSize+blockStep)+j);
                                 char l= (I[1]>(I[0]+FastThresh))+(I[9]>(I[0]+FastThresh))+(I[5]>(I[0]+FastThresh))+(I[13]>(I[0]+FastThresh));
                                 char h= (I[1]<(I[0]-FastThresh))+(I[9]<(I[0]-FastThresh))+(I[5]<(I[0]-FastThresh))+(I[13]<(I[0]+FastThresh));
                                 if ( (l<3) && (h<3) )
                                   continue;
                                 char sg = 1;
                                 if (l >3)
                                   sg = -1;
                                   
                                 I[2] = i1__at(blockX*(blockSize+blockStep)+(i+1),blockY*(blockSize+blockStep)+(j-3));
                                 I[3] = i1__at(blockX*(blockSize+blockStep)+(i+2),blockY*(blockSize+blockStep)+(j-2));
                                 I[4] = i1__at(blockX*(blockSize+blockStep)+(i+3),blockY*(blockSize+blockStep)+(j-1));
                                 I[6] = i1__at(blockX*(blockSize+blockStep)+(i+3),blockY*(blockSize+blockStep)+(j+1));
                                 I[7] = i1__at(blockX*(blockSize+blockStep)+(i+2),blockY*(blockSize+blockStep)+(j+2));
                                 I[8] = i1__at(blockX*(blockSize+blockStep)+(i+1),blockY*(blockSize+blockStep)+(j+3));
                                 I[10] = i1__at(blockX*(blockSize+blockStep)+(i-1),blockY*(blockSize+blockStep)+(j+3));
                                 I[11] = i1__at(blockX*(blockSize+blockStep)+(i-2),blockY*(blockSize+blockStep)+(j+2));
                                 I[12] = i1__at(blockX*(blockSize+blockStep)+(i-3),blockY*(blockSize+blockStep)+(j+1));
                                 I[14] = i1__at(blockX*(blockSize+blockStep)+(i-3),blockY*(blockSize+blockStep)+(j-1));
                                 I[15] = i1__at(blockX*(blockSize+blockStep)+(i-2),blockY*(blockSize+blockStep)+(j-2));
                                 I[16] = i1__at(blockX*(blockSize+blockStep)+(i-1),blockY*(blockSize+blockStep)+(j-3));
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
                                    output_view[((blockY*(blockSize+blockStep)) + j)*showCornWidth+ ((blockX*(blockSize+blockStep)) + i)] = 100;
                                 }
                                 else {
                                    output_view[((blockY*(blockSize+blockStep)) + j)*showCornWidth+ ((blockX*(blockSize+blockStep)) + i)] = 0;
                                 }
                }
             }
        
            return;

        for (int m=0; m<repetitions; m++)
            for (int n=0; n<repetitions; n++)
            {
                int currXshift = n*threadDiameter + threadX;
                int currYshift = m*threadDiameter + threadY;

                if ((currXshift<ScanDiameter) && (currYshift<ScanDiameter))
                {
                    abssum[currYshift][currXshift] = 0;

                    for (int i=0;i<blockSize;i++)
                    {
                        for (int j=0;j<blockSize;j++)
                        {
                            atomic_add(&(abssum[currYshift][currXshift]),
                              abs(
                                 input_1[((imgSrcOffset + blockX*(blockSize+blockStep)) + scanRadius + i)+
                                         ((blockY*(blockSize+blockStep)) + scanRadius + j)*imgSrcWidth]
                                 -
                                 input_2[((imgSrcOffset + blockX*(blockSize+blockStep)) + i + currXshift)+
                                         ((blockY*(blockSize+blockStep)) + j + currYshift)*imgSrcWidth]
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

        if (threadY == 0)
        {
            for (int n=0; n<repetitions; n++)
            {
                int currXshift = n*threadDiameter + threadX;
                if (currXshift > ScanDiameter)
                  break;
                minval[currXshift] = abssum[currXshift][0];
                minX[currXshift] = -scanRadius;
                for (int i=1;i<ScanDiameter;i++)
                {
                  if (minval[currXshift] > abssum[currXshift][i])
                  {
                      minval[currXshift] = abssum[currXshift][i];
                      minX[currXshift] = i-scanRadius;
                  }
                }
            }
         }

        barrier(CLK_LOCAL_MEM_FENCE);


        if ( (threadY == 0) && (threadX == 0))
        {

           int minvalFin = minval[0];
           minY = -scanRadius;
           for (int i=1;i<ScanDiameter;i++)
              {
                 if (minvalFin > minval[i])
                 {
                    minvalFin = minval[i];
                    minY = i-scanRadius;
                 }
              }
           output_Y[blockY*imgDstWidth+imgDstOffset + blockX] = minY;
           output_X[blockY*imgDstWidth+imgDstOffset + blockX] = minX[minY+scanRadius];

           if ((abssum[scanRadius][scanRadius] - minvalFin) <= MinValThreshold)  //if the difference is small, then it is considered to be noise in a uniformly colored area
           {
              output_Y[blockY*imgDstWidth+imgSrcOffset+blockX] = 0;
              output_X[blockY*imgDstWidth+imgSrcOffset+blockX] = 0;
           }
           if ((minvalFin) >= MaxAbsDiffThreshold)  //if the value is great, then it is considered to be too noisy, blurred or with too great a shift
           {
              output_Y[blockY*imgDstWidth+imgSrcOffset+blockX] = 0;
              output_X[blockY*imgDstWidth+imgSrcOffset+blockX] = 0;
           }

        }

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

