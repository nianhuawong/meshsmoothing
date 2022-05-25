/* Copyright 2019 The MathWorks, Inc. */

#ifndef TANH_LAYER_IMPL_HPP
#define TANH_LAYER_IMPL_HPP

#include "MWCNNLayerImpl.hpp"

/*
 *  Codegen class for Keras tanh Layer
 */
class MWTargetNetworkImpl;
class MWTanhLayerImpl : public MWCNNLayerImpl {
  public:
    MWTanhLayerImpl(MWCNNLayer*, MWTargetNetworkImpl*);
    ~MWTanhLayerImpl();

    void propagateSize();
    void predict();

  private:
    std::unique_ptr<mkldnn::eltwise_forward::desc> tanh_d;
    std::unique_ptr<mkldnn::eltwise_forward::primitive_desc> tanh_pd;
    std::unique_ptr<mkldnn::eltwise_forward::primitive> tanh;
};
#endif
