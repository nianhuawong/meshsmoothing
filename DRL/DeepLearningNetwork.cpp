//
// File: DeepLearningNetwork.cpp
//
// MATLAB Coder version            : 5.0
// C/C++ source code generated on  : 24-May-2022 22:35:28
//

// Include Files
#include "DeepLearningNetwork.h"
#include "evaluatePolicy.h"
#include "evaluatePolicy_rtwutil.h"
#include "predict.h"

// Type Definitions
#include "cnn_api.hpp"
#include "MWTanhLayer.hpp"
#include "MWTargetNetworkImpl.hpp"

// Function Definitions

//
// Arguments    : int numBufstoAllocate
//                MWCNNLayer *layers[9]
//                int numLayers
// Return Type  : void
//

//
// Arguments    : void
// Return Type  : void
//

//
// Arguments    : void
// Return Type  : void
//

//
// Arguments    : void
// Return Type  : void
//

//
// Arguments    : MWTargetNetworkImpl *targetImpl
//                MWTensor *b
//                int InputSize
//                int OutputSize
//                const char * c_a___codegen_lib_evaluatePolic
//                const char * d_a___codegen_lib_evaluatePolic
//                int c
// Return Type  : void
//

//
// Arguments    : MWTargetNetworkImpl *targetImpl
//                MWTensor *m_in
//                int height
//                int width
//                int channels
//                int withAvg
//                const char * b
//                int c
// Return Type  : void
//

//
// Arguments    : MWTargetNetworkImpl *targetImpl
//                MWTensor *b
//                int c
// Return Type  : void
//

//
// Arguments    : MWTargetNetworkImpl *targetImpl
//                MWTensor *b
//                int inPlaceOp
//                int c
// Return Type  : void
//

//
// Arguments    : MWTargetNetworkImpl *targetImpl
//                MWTensor *b
//                int c
// Return Type  : void
//

//
// Arguments    : void
// Return Type  : void
//

//
// Arguments    : void
// Return Type  : void
//

//
// Arguments    : void
// Return Type  : float *
//

//
// Arguments    : MWCNNLayer *layers[9]
//                int layerIdx
//                int portIdx
// Return Type  : float *
//

//
// Arguments    : int handle
// Return Type  : void
//

//
// Arguments    : void
// Return Type  : void
//

//
// Arguments    : void
// Return Type  : void
//

//
// Arguments    : void
// Return Type  : void
//

//
// Arguments    : void
// Return Type  : void
//

//
// Arguments    : int batchSize
// Return Type  : void
//

//
// Arguments    : int channels
// Return Type  : void
//

//
// Arguments    : float *data
// Return Type  : void
//

//
// Arguments    : int height
// Return Type  : void
//

//
// Arguments    : const char * name
// Return Type  : void
//

//
// Arguments    : int sequenceLength
// Return Type  : void
//

//
// Arguments    : int width
// Return Type  : void
//

//
// Arguments    : void
// Return Type  : void
//
void b_policy_0::allocate()
{
  this->targetImpl->allocate(2, this->layers, this->numLayers);
  for (int idx = 0; idx < 9; idx++) {
    this->layers[idx]->allocate();
  }

  this->inputTensor->setData(this->layers[0]->getLayerOutput(0));
  this->outputTensor->setData(this->layers[8]->getLayerOutput(0));
}

//
// Arguments    : void
// Return Type  : void
//
void b_policy_0::cleanup()
{
  this->deallocate();
  for (int idx = 0; idx < 9; idx++) {
    this->layers[idx]->cleanup();
  }

  if (this->targetImpl) {
    this->targetImpl->cleanup();
  }
}

//
// Arguments    : void
// Return Type  : void
//
void b_policy_0::deallocate()
{
  this->targetImpl->deallocate();
  for (int idx = 0; idx < 9; idx++) {
    this->layers[idx]->deallocate();
  }
}

//
// Arguments    : void
// Return Type  : float *
//
float *b_policy_0::getInputDataPointer()
{
  return this->inputTensor->getFloatData();
}

//
// Arguments    : int layerIndex
//                int portIndex
// Return Type  : float *
//
float *b_policy_0::getLayerOutput(int layerIndex, int portIndex)
{
  return this->targetImpl->getLayerOutput(this->layers, layerIndex, portIndex);
}

//
// Arguments    : void
// Return Type  : float *
//
float *b_policy_0::getOutputDataPointer()
{
  return this->outputTensor->getFloatData();
}

//
// Arguments    : void
// Return Type  : void
//
b_policy_0::b_policy_0()
{
  this->numLayers = 9;
  this->targetImpl = 0;
  this->layers[0] = new MWInputLayer;
  this->layers[0]->setName("state");
  this->layers[1] = new MWFCLayer;
  this->layers[1]->setName("ActorFC1");
  this->layers[2] = new MWReLULayer;
  this->layers[2]->setName("ActorRelu1");
  this->layers[3] = new MWFCLayer;
  this->layers[3]->setName("ActorFC2");
  this->layers[4] = new MWReLULayer;
  this->layers[4]->setName("ActorRelu2");
  this->layers[5] = new MWFCLayer;
  this->layers[5]->setName("ActorFC3");
  this->layers[6] = new MWTanhLayer;
  this->layers[6]->setName("ActorTanhLayer");
  this->layers[7] = new MWFCLayer;
  this->layers[7]->setName("actor");
  this->layers[8] = new MWOutputLayer;
  this->layers[8]->setName("RepresentationLoss");
  this->targetImpl = new MWTargetNetworkImpl;
  this->inputTensor = new MWTensor;
  this->inputTensor->setHeight(10);
  this->inputTensor->setWidth(2);
  this->inputTensor->setChannels(1);
  this->inputTensor->setBatchSize(1);
  this->inputTensor->setSequenceLength(1);
  this->outputTensor = new MWTensor;
}

//
// Arguments    : void
// Return Type  : void
//
b_policy_0::~b_policy_0()
{
  this->cleanup();
  for (int idx = 0; idx < 9; idx++) {
    delete this->layers[idx];
  }

  if (this->targetImpl) {
    delete this->targetImpl;
  }

  delete this->inputTensor;
  delete this->outputTensor;
}

//
// Arguments    : void
// Return Type  : void
//
void b_policy_0::postsetup()
{
  this->targetImpl->postSetup();
}

//
// Arguments    : void
// Return Type  : void
//
void b_policy_0::predict()
{
  for (int idx = 0; idx < 9; idx++) {
    this->layers[idx]->predict();
  }
}

//
// Arguments    : void
// Return Type  : void
//
void b_policy_0::setSize()
{
  for (int idx = 0; idx < 9; idx++) {
    this->layers[idx]->propagateSize();
  }

  this->allocate();
  this->postsetup();
}

//
// Arguments    : void
// Return Type  : void
//
void b_policy_0::setup()
{
  this->targetImpl->preSetup();
  (dynamic_cast<MWInputLayer *>(this->layers[0]))->createInputLayer
    (this->targetImpl, this->inputTensor, 10, 2, 1, 0, "", 0);
  (dynamic_cast<MWFCLayer *>(this->layers[1]))->createFCLayer(this->targetImpl,
    this->layers[0]->getOutputTensor(0), 20, 30,
    //"./codegen/lib/evaluatePolicy/cnn_policy_ActorFC1_w.bin",
    //"./codegen/lib/evaluatePolicy/cnn_policy_ActorFC1_b.bin", 1);
      "../DRL/cnn_policy_ActorFC1_w.bin",
      "../DRL/cnn_policy_ActorFC1_b.bin", 1);
  (dynamic_cast<MWReLULayer *>(this->layers[2]))->createReLULayer
    (this->targetImpl, this->layers[1]->getOutputTensor(0), 1, 1);
  (dynamic_cast<MWFCLayer *>(this->layers[3]))->createFCLayer(this->targetImpl,
    this->layers[2]->getOutputTensor(0), 30, 30,
    //"./codegen/lib/evaluatePolicy/cnn_policy_ActorFC2_w.bin",
    //"./codegen/lib/evaluatePolicy/cnn_policy_ActorFC2_b.bin", 0);
      "../DRL/cnn_policy_ActorFC2_w.bin",
      "../DRL/cnn_policy_ActorFC2_b.bin", 0);
  (dynamic_cast<MWReLULayer *>(this->layers[4]))->createReLULayer
    (this->targetImpl, this->layers[3]->getOutputTensor(0), 1, 0);
  (dynamic_cast<MWFCLayer *>(this->layers[5]))->createFCLayer(this->targetImpl,
    this->layers[4]->getOutputTensor(0), 30, 30,
    //"./codegen/lib/evaluatePolicy/cnn_policy_ActorFC3_w.bin",
    //"./codegen/lib/evaluatePolicy/cnn_policy_ActorFC3_b.bin", 1);
      "../DRL/cnn_policy_ActorFC3_w.bin",
      "../DRL/cnn_policy_ActorFC3_b.bin", 1);
  (dynamic_cast<MWTanhLayer *>(this->layers[6]))->createTanhLayer
    (this->targetImpl, this->layers[5]->getOutputTensor(0), 0);
  (dynamic_cast<MWFCLayer *>(this->layers[7]))->createFCLayer(this->targetImpl,
    this->layers[6]->getOutputTensor(0), 30, 2,
    //"./codegen/lib/evaluatePolicy/cnn_policy_actor_w.bin",
    //"./codegen/lib/evaluatePolicy/cnn_policy_actor_b.bin", 1);
      "../DRL/cnn_policy_actor_w.bin",
      "../DRL/cnn_policy_actor_b.bin", 1);
  (dynamic_cast<MWOutputLayer *>(this->layers[8]))->createOutputLayer
    (this->targetImpl, this->layers[7]->getOutputTensor(0), 1);
  this->setSize();
}

//
// Arguments    : b_policy_0 *obj
// Return Type  : void
//
void DeepLearningNetwork_setup(b_policy_0 *obj)
{
  obj->setup();
  obj->batchSize = 1;
}

//
// File trailer for DeepLearningNetwork.cpp
//
// [EOF]
//
