#include "MWTanhLayer.hpp"
#include "MWTanhLayerImpl.hpp"
#include "MWTargetNetworkImpl.hpp"
#include "MWMkldnnUtils.hpp"
#include <stdarg.h>
#include <cassert>
 MWTanhLayerImpl::MWTanhLayerImpl(MWCNNLayer* layer, MWTargetNetworkImpl* 
ntwk_impl) : MWCNNLayerImpl(layer, ntwk_impl) { } 
MWTanhLayerImpl::~MWTanhLayerImpl() { } void MWTanhLayerImpl::propagateSize() { 
float aplha = 0.0; MWTanhLayer* tanhLayer = 
static_cast<MWTanhLayer*>(getLayer()); MWTensor* ipTensor = 
tanhLayer->getInputTensor(0); setLayerMemory(std::make_shared<mkldnn::memory>( 
MWMkldnnUtils::getLayerMemoryPrimitive(ipTensor, 
euppfEoiaoCTcVgRPVhA)->get_desc(), *euppfEoiaoCTcVgRPVhA->eng)); tanh_d = 
std::unique_ptr<mkldnn::eltwise_forward::desc>(new 
mkldnn::eltwise_forward::desc( mkldnn::prop_kind::forward_inference, 
mkldnn::algorithm::eltwise_tanh, 
MWMkldnnUtils::getLayerMemoryPrimitive(ipTensor, 
euppfEoiaoCTcVgRPVhA)->get_desc(), aplha)); tanh_pd = 
std::unique_ptr<mkldnn::eltwise_forward::primitive_desc>( new 
mkldnn::eltwise_forward::primitive_desc(*tanh_d, *euppfEoiaoCTcVgRPVhA->eng)); 
tanh = std::unique_ptr<mkldnn::eltwise_forward>(new 
mkldnn::eltwise_forward(*tanh_pd)); pipeline_memory_args.push_back( 
{{MKLDNN_ARG_FROM, *MWMkldnnUtils::getLayerMemoryPrimitive(ipTensor, 
euppfEoiaoCTcVgRPVhA)}, {MKLDNN_ARG_TO, *getLayerMemory()}}); 
pipeline.push_back(*tanh); } void MWTanhLayerImpl::predict() { auto s = 
mkldnn::stream(*euppfEoiaoCTcVgRPVhA->eng); assert(pipeline.size() == 
pipeline_memory_args.size()); for (int i = 0; i < pipeline.size(); i++) { 
pipeline[i].execute(s, pipeline_memory_args[i]); }
#if MW_TANH_TAP
 reorderedLayerOutputTap(0);
#endif
 }