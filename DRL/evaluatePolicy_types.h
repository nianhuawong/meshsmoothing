//
// File: evaluatePolicy_types.h
//
// MATLAB Coder version            : 5.0
// C/C++ source code generated on  : 24-May-2022 22:35:28
//
#ifndef EVALUATEPOLICY_TYPES_H
#define EVALUATEPOLICY_TYPES_H

// Include Files
#include "rtwtypes.h"

// Type Definitions
#include "cnn_api.hpp"
#include "MWTargetNetworkImpl.hpp"

// Type Definitions
class b_policy_0
{
 public:
  void allocate();
  void postsetup();
  b_policy_0();
  void deallocate();
  void setSize();
  void setup();
  void predict();
  void cleanup();
  float *getLayerOutput(int layerIndex, int portIndex);
  float *getInputDataPointer();
  float *getOutputDataPointer();
  ~b_policy_0();
  int batchSize;
  int numLayers;
  MWTensor *inputTensor;
  MWTensor *outputTensor;
  MWCNNLayer *layers[9];
  float *inputData;
  float *outputData;
 private:
  MWTargetNetworkImpl *targetImpl;
};

#endif

//
// File trailer for evaluatePolicy_types.h
//
// [EOF]
//
