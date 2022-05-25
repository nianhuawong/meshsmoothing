#include "MWTargetNetworkImpl.hpp"
#include "cnn_api.hpp"
#include "MWCNNLayerImpl.hpp"
#include "MWMkldnnUtils.hpp"
#include <cstring>
 void MWTargetNetworkImpl::allocate(int numBufsToAlloc, MWCNNLayer* layers[], 
int numLayers) { numBufs = numBufsToAlloc; int maxBufSize = -1; for (int i = 0; 
i < numLayers; i++) { if ((layers[i]->getImpl() != NULL) && 
(dynamic_cast<MWOutputLayer*>(layers[i]) == NULL)) { maxBufSize = 
std::max(maxBufSize, 
(int)(layers[i]->getImpl()->getLayerMemory()->get_desc().get_size() / 4)); } } 
for (int i = 0; i < numBufs; i++) { memBuffer.push_back(new float[maxBufSize]); 
} } void MWTargetNetworkImpl::preSetup() { eng = 
std::unique_ptr<mkldnn::engine>(new mkldnn::engine(mkldnn::engine::kind::cpu, 
0)); } float* MWTargetNetworkImpl::getLayerOutput(MWCNNLayer* layers[], int 
layerIndex, int portIndex) { MWTensor* opTensor = 
layers[layerIndex]->getOutputTensor(portIndex); float* opData = 
getLayerActivation(opTensor); return opData; } float* 
MWTargetNetworkImpl::getLayerActivation(MWTensor* opTensor) { MWCNNLayerImpl* 
layerImpl = opTensor->getOwner()->getImpl(); if (layerImpl == NULL) {  return 
getLayerActivation(opTensor->getOwner()->getInputTensor()); } else { if 
(dynamic_cast<MWOutputLayerImpl*>(layerImpl)) { return 
opTensor->getData<float>(); } else { std::shared_ptr<mkldnn::memory> 
currentLayerMemory = MWMkldnnUtils::getLayerMemoryPrimitive(opTensor, this); 
auto currentLayerMemoryDesc = currentLayerMemory->get_desc(); format_type 
formatTag = MWMkldnnUtils::getformatType(currentLayerMemoryDesc, opTensor); if 
((formatTag != format_type::NC_FORMAT) && (formatTag != 
format_type::NCHW_FORMAT)){ int layerOutputSize = opTensor->getBatchSize() * 
opTensor->getChannels() * opTensor->getHeight() * opTensor->getWidth(); 
MWMkldnnUtils::configureReorder(layerImpl, opTensor, 
mkldnn::memory::format_tag::nchw); std::vector<mkldnn::primitive> 
activationsPipeline; std::vector<std::unordered_map<int, mkldnn::memory>> 
activationsPipeline_memory_args; 
activationsPipeline.push_back(*layerImpl->getReorderPrim()); auto s = 
mkldnn::stream(*eng); activationsPipeline_memory_args.push_back( 
{{MKLDNN_ARG_FROM, *MWMkldnnUtils::getLayerMemoryPrimitive(opTensor, this)}, 
{MKLDNN_ARG_TO, *layerImpl->getReorderLayerMemory()}}); 
assert(activationsPipeline.size() == activationsPipeline_memory_args.size()); 
for (int i = 0; i < activationsPipeline.size(); i++) { 
activationsPipeline[i].execute(s, activationsPipeline_memory_args[i]); } 
memcpy(opTensor->getData<float>(), 
layerImpl->getReorderLayerMemory()->get_data_handle(), layerOutputSize * 
sizeof(float)); } return opTensor->getData<float>(); } } } void 
MWTargetNetworkImpl::deallocate() { for (int i = 0; i < memBuffer.size(); i++) 
{ if (memBuffer[i] != nullptr) { delete[] memBuffer[i]; } } memBuffer.clear(); 
} void MWTargetNetworkImpl::cleanup() { }