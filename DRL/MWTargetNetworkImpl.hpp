/* Copyright 2017-2018 The MathWorks, Inc. */

#ifndef CNN_NTWK_IMPL
#define CNN_NTWK_IMPL

#include <vector>
#include "cnn_api.hpp"
#include "mkldnn.hpp"
#define MW_TARGET_TYPE_MKLDNN 1

class MWTargetNetworkImpl
{
  public:
    MWTargetNetworkImpl() : numBufs(0) {}
    ~MWTargetNetworkImpl() {}
    void allocate(int, MWCNNLayer *layers[],int numLayers);
    void deallocate();
    void preSetup();
    void postSetup(){}
    void cleanup();

    std::vector<float *> memBuffer;
    float *getLayerOutput(MWCNNLayer *layers[], int layerIndex, int portIndex);
    float* getLayerActivation(MWTensor*);
    int numBufs;

    std::unique_ptr<mkldnn::engine>  eng;
};
#endif
