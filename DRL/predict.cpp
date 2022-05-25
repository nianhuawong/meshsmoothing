//
// File: predict.cpp
//
// MATLAB Coder version            : 5.0
// C/C++ source code generated on  : 24-May-2022 22:35:28
//

// Include Files
#include "predict.h"
#include "DeepLearningNetwork.h"
#include "evaluatePolicy.h"
#include "evaluatePolicy_rtwutil.h"
#include <cstring>

// Type Definitions
struct cell_wrap_3
{
  float f1[2];
};

// Function Definitions

//
// Arguments    : void
// Return Type  : int *
//

//
// Arguments    : b_policy_0 *obj
//                const double in[20]
//                float varargout_1[2]
// Return Type  : void
//
void DeepLearningNetwork_predict(b_policy_0 *obj, const double in[20], float
  varargout_1[2])
{
  float miniBatchT[20];
  cell_wrap_3 outputsMiniBatch[1];
  for (int i = 0; i < 10; i++) {
    int miniBatchT_tmp;
    miniBatchT_tmp = i << 1;
    miniBatchT[miniBatchT_tmp] = static_cast<float>(in[i]);
    miniBatchT[miniBatchT_tmp + 1] = static_cast<float>(in[i + 10]);
  }

  memcpy(obj->getInputDataPointer(), miniBatchT, obj->layers[0]->getOutputTensor
         (0)->getNumElements() * sizeof(float));
  obj->predict();
  memcpy(outputsMiniBatch[0].f1, obj->getLayerOutput(8, 0), obj->layers[8]
         ->getOutputTensor(0)->getNumElements() * sizeof(float));
  varargout_1[0] = outputsMiniBatch[0].f1[0];
  varargout_1[1] = outputsMiniBatch[0].f1[1];
}

//
// File trailer for predict.cpp
//
// [EOF]
//
