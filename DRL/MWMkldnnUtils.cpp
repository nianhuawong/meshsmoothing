#include <cassert>
#include <cstring>
#include <stdio.h>
#include "MWCNNLayerImpl.hpp"
#include "MWTargetNetworkImpl.hpp"
#include "cnn_api.hpp"
#include "MWMkldnnUtils.hpp"
#include "mkldnn.hpp"
 void MWMkldnnUtils::configureReorder(MWCNNLayerImpl* layerImpl, MWTensor* 
srcTensor, mkldnn::memory::format_tag dstDataFormat, int index) { 
std::shared_ptr<mkldnn::memory> ipLayerMemory = 
MWMkldnnUtils::getLayerMemoryPrimitive(srcTensor, layerImpl->getTargetImpl()); 
mkldnn::memory::dims layerMemoryDimensions = {srcTensor->getBatchSize(), 
srcTensor->getChannels(), srcTensor->getHeight(), srcTensor->getWidth()}; auto 
layerMemoryDescriptor = mkldnn::memory::desc(layerMemoryDimensions, 
mkldnn::memory::data_type::f32, dstDataFormat); 
layerImpl->setReorderLayerMemory(std::make_shared<mkldnn::memory>( 
layerMemoryDescriptor, *(layerImpl->getTargetImpl())->eng), index); 
layerImpl->setReorderPrim( std::make_shared<mkldnn::reorder>(*ipLayerMemory, 
*layerImpl->getReorderLayerMemory(index)), index); } format_type 
MWMkldnnUtils::getformatType(mkldnn::memory::desc inputdesc, MWTensor* 
srcTensor ) { mkldnn::memory::dims layerMemoryDimensionsNC = 
{srcTensor->getBatchSize(), srcTensor->getChannels() * srcTensor->getHeight() * 
srcTensor->getWidth()}; mkldnn::memory::dims layerMemoryDimensions = 
{srcTensor->getBatchSize(), srcTensor->getChannels(), srcTensor->getHeight(), 
srcTensor->getWidth()}; auto layerMemoryDescriptorNC = 
mkldnn::memory::desc(layerMemoryDimensionsNC, mkldnn::memory::data_type::f32, 
mkldnn::memory::format_tag::nc); if (inputdesc == layerMemoryDescriptorNC){ 
return format_type::NC_FORMAT; } auto layerMemoryDescriptorNCHW = 
mkldnn::memory::desc(layerMemoryDimensions, mkldnn::memory::data_type::f32, 
mkldnn::memory::format_tag::nchw); if (inputdesc == layerMemoryDescriptorNCHW){ 
return format_type::NCHW_FORMAT; } auto layerMemoryDescriptorNCHW8C = 
mkldnn::memory::desc(layerMemoryDimensions, mkldnn::memory::data_type::f32, 
mkldnn::memory::format_tag::nChw8c); if (inputdesc == 
layerMemoryDescriptorNCHW8C){ return format_type::NCHW8C_FORMAT; } return 
format_type::UNKNOWN_FORMAT ; } bool 
MWMkldnnUtils::checkformatType(mkldnn::memory::desc inputdesc, MWTensor* 
srcTensor, mkldnn::memory::format_tag dstDataFormat) { mkldnn::memory::dims 
layerMemoryDimensions = {srcTensor->getBatchSize(), srcTensor->getChannels(), 
srcTensor->getHeight(), srcTensor->getWidth()};  if(dstDataFormat == 
mkldnn::memory::format_tag::nc){ layerMemoryDimensions = 
{srcTensor->getBatchSize(), srcTensor->getChannels() * srcTensor->getHeight() * 
srcTensor->getWidth()};  } auto layerMemoryDescriptor = 
mkldnn::memory::desc(layerMemoryDimensions, mkldnn::memory::data_type::f32, 
dstDataFormat); bool isformatMatched = (layerMemoryDescriptor == inputdesc); 
return isformatMatched; } std::shared_ptr<mkldnn::memory> 
MWMkldnnUtils::getLayerMemoryPrimitive( MWTensor* aTensor, MWTargetNetworkImpl* 
ntwkImpl) { auto owningLayer = aTensor->getOwner(); auto owningLayerImpl = 
owningLayer->getImpl(); if (owningLayerImpl == NULL) { 
assert(owningLayer->getNumInputs() == 1); return 
MWMkldnnUtils::getLayerMemoryPrimitive(owningLayer->getInputTensor(0), 
ntwkImpl); } else { auto layerMemory = 
owningLayerImpl->getLayerMemory(aTensor->getSourcePortIndex()); if 
(layerMemory) { return layerMemory; } else { assert(owningLayer->getNumInputs() 
== 1); return 
MWMkldnnUtils::getLayerMemoryPrimitive(owningLayer->getInputTensor(0), 
ntwkImpl); } } }