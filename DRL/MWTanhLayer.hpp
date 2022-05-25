/* Copyright 2018 The MathWorks, Inc. */

// Target Agnostic header for Keras' Tanh Layer
#ifndef TANH_LAYER_HPP
#define TANH_LAYER_HPP

#include "cnn_api.hpp"


/**
  * Codegen class for Keras Tanh Layer
**/
class MWTanhLayer : public MWCNNLayer
{
  public:
    MWTanhLayer();
    ~MWTanhLayer();

    /** Create a new Tanh Layer */
    void createTanhLayer(MWTargetNetworkImpl*, MWTensor*, int);
    void propagateSize();
};
#endif
