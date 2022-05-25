/* Copyright 2018-2019 The MathWorks, Inc. */

// Target Agnostic implementation for Keras's Tanh Layer
#include "MWTanhLayer.hpp"
#include "MWTanhLayerImpl.hpp"
#include "MWTargetNetworkImpl.hpp"

MWTanhLayer::MWTanhLayer()
{
}

MWTanhLayer::~MWTanhLayer()
{
}

void MWTanhLayer::createTanhLayer(MWTargetNetworkImpl* ntwk_impl,
                                  MWTensor* dataInput,
                                  int outbufIdx)
{
#if defined(MW_TARGET_TYPE_CUDNN) || defined(MW_TARGET_TYPE_MKLDNN) || defined(MW_TARGET_TYPE_ARMNEON)
    setInputTensor(dataInput);
    allocateOutputTensor(-1, -1, -1, -1, -1, NULL);

    getOutputTensor(0)->setopBufIndex(outbufIdx);

    m_impl = new MWTanhLayerImpl(this, ntwk_impl);
    
#else
    setInputTensor(dataInput);
    allocateOutputTensor(getInputTensor()->getHeight(), getInputTensor()->getWidth(), getInputTensor()->getChannels(), getInputTensor()->getBatchSize(), getInputTensor()->getSequenceLength(), NULL);

    m_impl = new MWTanhLayerImpl(this, ntwk_impl, outbufIdx);
#endif
}

void MWTanhLayer::propagateSize()
{
#if defined(MW_TARGET_TYPE_CUDNN) || defined(MW_TARGET_TYPE_MKLDNN) || defined(MW_TARGET_TYPE_ARMNEON)
    resizeOutputTensor(getInputTensor()->getHeight(),
                       getInputTensor()->getWidth(),
                       getInputTensor()->getChannels(),
                       getInputTensor()->getBatchSize(),
                       getInputTensor()->getSequenceLength());

    m_impl->propagateSize();
#endif
}
