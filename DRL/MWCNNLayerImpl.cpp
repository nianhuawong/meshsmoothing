#include <cassert>
#include <cstring>
#include <stdio.h>
#include "MWCNNLayerImpl.hpp"
#include "MWTargetNetworkImpl.hpp"
#include "cnn_api.hpp"
#include "MWMkldnnUtils.hpp"
#include "mkldnn.hpp"
 MWCNNLayerImpl::MWCNNLayerImpl(MWCNNLayer* layer, MWTargetNetworkImpl* 
ntwk_impl) : bMAyVFGSPDjmUbziYLAy(layer) , euppfEoiaoCTcVgRPVhA(ntwk_impl) { } 
std::shared_ptr<mkldnn::memory> MWCNNLayerImpl::getLayerMemory(int index) { if 
(layerMemory.size() >= 1) { std::map<int, 
std::shared_ptr<mkldnn::memory>>::iterator it = layerMemory.find(index); return 
it->second; } else { return nullptr; } } std::shared_ptr<mkldnn::memory> 
MWCNNLayerImpl::getReorderLayerMemory(int index) { 
if(reorderLayerMemory.size()>=1) { std::map<int, 
std::shared_ptr<mkldnn::memory> >::iterator it = 
reorderLayerMemory.find(index); return it->second; } else return nullptr; } 
std::shared_ptr<mkldnn::reorder> MWCNNLayerImpl::getReorderPrim(int index) { 
if(reorderPrim.size()>=1) { std::map<int, std::shared_ptr<mkldnn::reorder> 
>::iterator it = reorderPrim.find(index); return it->second; } else return 
nullptr; } void MWCNNLayerImpl::setLayerMemory(std::shared_ptr<mkldnn::memory> 
other, int index) { layerMemory[index] = other; } void 
MWCNNLayerImpl::setReorderPrim(std::shared_ptr<mkldnn::reorder> other, int 
index) { reorderPrim[index] = other;  } void 
MWCNNLayerImpl::setReorderLayerMemory(std::shared_ptr<mkldnn::memory> other, 
int index) { reorderLayerMemory[index] = other; } void 
MWCNNLayerImpl::allocateOutputData(int i) { MWTensor* opTensor = 
getLayer()->getOutputTensor(i); int fYaOQTeunPwVjnhhTECh = 
static_cast<int>(this->getLayerMemory()->get_desc().get_size() / 4); int 
outBufIndex = opTensor->getopBufIndex(); if (outBufIndex < 0) { 
opTensor->setData((float*)calloc(fYaOQTeunPwVjnhhTECh, sizeof(float))); } 
else { opTensor->setData(euppfEoiaoCTcVgRPVhA->memBuffer[outBufIndex]); } 
MWMkldnnUtils::getLayerMemoryPrimitive(opTensor, euppfEoiaoCTcVgRPVhA) 
->set_data_handle(getData<float>(i)); } void 
MWCNNLayerImpl::deallocateOutputData(int i) { float* data = 
getLayer()->getOutputTensor(i)->getData<float>(); if (data) { if 
(getLayer()->getOutputTensor(i)->getopBufIndex() < 0) { free(data); } 
getLayer()->getOutputTensor(i)->setData((float*)NULL); } return; } 
MWInputLayerImpl::MWInputLayerImpl(MWCNNLayer* layer, MWTargetNetworkImpl* 
ntwk_impl, int TxNFOfYScyqGlEFFxbAv, int vIWQzNvYZSuxmOTVDFhU, int QwUuNuQNtlPXrIwRNiSZ, int 
xkUNToJIgvoLoUQuzKRF, const char* avg_file_name) : MWCNNLayerImpl(layer, 
ntwk_impl) , bDTIjtxZiSHtjwzgEluE(xkUNToJIgvoLoUQuzKRF) { if 
(bDTIjtxZiSHtjwzgEluE) { loadAvg(avg_file_name, TxNFOfYScyqGlEFFxbAv * vIWQzNvYZSuxmOTVDFhU, 
QwUuNuQNtlPXrIwRNiSZ); } } MWInputLayerImpl::~MWInputLayerImpl() { }
#if MW_LAYERS_TAP
 int tap_count = 0; void mw_interm_tap(float* inp, int size, int count) { FILE* 
fp; int i; std::string fileName{"taps/mw_interm_tap_"};
#define TXT_FILE 1
#if TXT_FILE
 fileName = fileName + std::to_string(count) + ".txt"; fp = 
fopen(fileName.c_str(), "w"); if (fp == NULL) { std::string errmsg = 
std::string("Error opening text file .Create taps folder to open") + fileName; 
printf("Error! Unable to open file %s\n", fileName); throw 
std::runtime_error(errmsg.c_str()); } for (i = 0; i < size; i++) { fprintf(fp, 
"%f\n", inp[i]); }
#else
 fileName = fileName + std::to_string(count) + ".bin"; fp = 
fopen(fileName.c_str(), "wb"); if (fp == NULL) { std::string errmsg = 
std::string("Error opening binary file .Create taps folder to open") + 
fileName; printf("Error! Unable to open file %s\n", fileName); throw 
std::runtime_error(errmsg.c_str()); } fwrite(inp, 4, size, fp);
#endif
 fclose(fp); } void MWCNNLayerImpl::reorderedLayerOutputTap(int portIndex) { 
MWTensor* opTensor = getLayer()->getOutputTensor(portIndex); MWCNNLayerImpl* 
layerImpl = bMAyVFGSPDjmUbziYLAy->getImpl(); int layerOutputSize = 
opTensor->getBatchSize() * opTensor->getChannels() * opTensor->getHeight() * 
opTensor->getWidth(); float* layerData = (float*)calloc(layerOutputSize, 
sizeof(float)); std::shared_ptr<mkldnn::memory> currentLayerMemory = 
MWMkldnnUtils::getLayerMemoryPrimitive(opTensor, euppfEoiaoCTcVgRPVhA); bool 
isMemPrimitiveNC = 
MWMkldnnUtils::checkformatType(currentLayerMemory->get_desc(), opTensor, 
mkldnn::memory::format_tag::nc); bool isMemPrimitiveNCHW = 
MWMkldnnUtils::checkformatType(currentLayerMemory->get_desc(), opTensor, 
mkldnn::memory::format_tag::nchw);  auto s = 
mkldnn::stream(*euppfEoiaoCTcVgRPVhA->eng); if (!(isMemPrimitiveNC || 
isMemPrimitiveNCHW)) { MWMkldnnUtils::configureReorder(layerImpl, opTensor, 
mkldnn::memory::format_tag::nchw); pipeline.clear(); 
pipeline_memory_args.clear(); pipeline_memory_args.push_back({{MKLDNN_ARG_FROM, 
*MWMkldnnUtils::getLayerMemoryPrimitive( opTensor, euppfEoiaoCTcVgRPVhA)}, 
{MKLDNN_ARG_TO, *getReorderLayerMemory()}}); 
pipeline.push_back(*layerImpl->getReorderPrim()); assert(pipeline.size() == 
pipeline_memory_args.size()); pipeline[0].execute(s, pipeline_memory_args[0]); 
memcpy(layerData, 
(float*)layerImpl->getReorderLayerMemory()->get_data_handle(), layerOutputSize 
* sizeof(float)); } else { memcpy(layerData, opTensor->getData<float>(), 
layerOutputSize * sizeof(float)); } mw_interm_tap(layerData, layerOutputSize, 
tap_count++); if (layerData) { free(layerData); layerData = NULL; } }
#endif
 void MWInputLayerImpl::propagateSize() { MWTensor* ipTensor = 
getLayer()->getInputTensor(0); mkldnn::memory::dims layerMemoryDimensions = 
{ipTensor->getBatchSize(), ipTensor->getChannels(), ipTensor->getHeight(), 
ipTensor->getWidth()}; auto layerMemoryDescriptor = mkldnn::memory::desc( 
layerMemoryDimensions, mkldnn::memory::data_type::f32, 
mkldnn::memory::format_tag::nchw); setLayerMemory( 
std::make_shared<mkldnn::memory>(layerMemoryDescriptor, 
*euppfEoiaoCTcVgRPVhA->eng)); setLayerMemory( 
std::make_shared<mkldnn::memory>(layerMemoryDescriptor, 
*euppfEoiaoCTcVgRPVhA->eng)); return; } void MWInputLayerImpl::loadAvg(const 
char* RqCYCrGsNvzKYrRMXbsI, int channelSize, int numChannels) { FILE* 
SGsAudmgjmvcUXzzrUtf = MWCNNLayer::openBinaryFile(RqCYCrGsNvzKYrRMXbsI); if 
(SGsAudmgjmvcUXzzrUtf == NULL) { printf("Unable to open file\n"); } int eybNKlJCSDUvsznWynwK 
= channelSize * numChannels; LtEgcYoEYjkrWuohutgw = new std::vector<float>; 
LtEgcYoEYjkrWuohutgw->reserve(eybNKlJCSDUvsznWynwK); if (bDTIjtxZiSHtjwzgEluE == 1) { 
call_fread(LtEgcYoEYjkrWuohutgw->data(), sizeof(float), eybNKlJCSDUvsznWynwK, SGsAudmgjmvcUXzzrUtf, 
RqCYCrGsNvzKYrRMXbsI); } else { int channelOffset = 0; std::vector<float> 
OwenhowBxTAXHXmJpIKd(numChannels); call_fread(OwenhowBxTAXHXmJpIKd.data(), 
sizeof(float), numChannels, SGsAudmgjmvcUXzzrUtf, RqCYCrGsNvzKYrRMXbsI); for (int i = 
0; i < numChannels; i++) { std::fill_n(LtEgcYoEYjkrWuohutgw->begin() + channelOffset, 
channelSize, OwenhowBxTAXHXmJpIKd[i]); channelOffset = channelOffset + 
channelSize; } } fclose(SGsAudmgjmvcUXzzrUtf); return; } void 
MWInputLayerImpl::predict() { float* inp = getData<float>(); int i, btch; 
MWInputLayer* inpLayer = static_cast<MWInputLayer*>(getLayer()); MWTensor* 
opTensor = inpLayer->getOutputTensor(0); if (bDTIjtxZiSHtjwzgEluE) { for (btch 
= 0; btch < opTensor->getBatchSize(); btch++) { for (i = 0; i < 
opTensor->getChannels() * opTensor->getHeight() * opTensor->getWidth(); i++) { 
inp[i] = inp[i] - LtEgcYoEYjkrWuohutgw->data()[i]; } inp += opTensor->getChannels() * 
opTensor->getHeight() * opTensor->getWidth(); } }
#if MW_INPUT_TAP
 reorderedLayerOutputTap(0);
#endif
 return; } void MWInputLayerImpl::cleanup() { if (bDTIjtxZiSHtjwzgEluE) { if 
(LtEgcYoEYjkrWuohutgw) { free(LtEgcYoEYjkrWuohutgw); LtEgcYoEYjkrWuohutgw = NULL; } } return; } 
MWReLULayerImpl::MWReLULayerImpl(MWCNNLayer* layer, MWTargetNetworkImpl* 
ntwk_impl, int inPlace) : MWCNNLayerImpl(layer, ntwk_impl) , 
UEESbUvbMihFnquvuFij(inPlace) { } MWReLULayerImpl::~MWReLULayerImpl() { } void 
MWReLULayerImpl::propagateSize() { MWReLULayer* reluLayer = 
static_cast<MWReLULayer*>(getLayer()); MWTensor* ipTensor = 
reluLayer->getInputTensor(); const float negative_slope = 0.0; 
setLayerMemory(std::make_shared<mkldnn::memory>( 
MWMkldnnUtils::getLayerMemoryPrimitive(ipTensor, 
euppfEoiaoCTcVgRPVhA)->get_desc(), *euppfEoiaoCTcVgRPVhA->eng)); relu_d = 
std::unique_ptr<mkldnn::eltwise_forward::desc>(new 
mkldnn::eltwise_forward::desc( mkldnn::prop_kind::forward_inference, 
mkldnn::algorithm::eltwise_relu, 
MWMkldnnUtils::getLayerMemoryPrimitive(ipTensor, 
euppfEoiaoCTcVgRPVhA)->get_desc(), negative_slope)); relu_pd = 
std::unique_ptr<mkldnn::eltwise_forward::primitive_desc>( new 
mkldnn::eltwise_forward::primitive_desc(*relu_d, *euppfEoiaoCTcVgRPVhA->eng)); 
relu = std::unique_ptr<mkldnn::eltwise_forward>(new 
mkldnn::eltwise_forward(*relu_pd)); pipeline_memory_args.push_back( 
{{MKLDNN_ARG_FROM, *MWMkldnnUtils::getLayerMemoryPrimitive(ipTensor, 
euppfEoiaoCTcVgRPVhA)}, {MKLDNN_ARG_TO, *getLayerMemory()}}); 
pipeline.push_back(*relu); } void MWReLULayerImpl::predict() { auto s = 
mkldnn::stream(*euppfEoiaoCTcVgRPVhA->eng); assert(pipeline.size() == 
pipeline_memory_args.size()); for (int i = 0; i < pipeline.size(); i++) { 
pipeline[i].execute(s, pipeline_memory_args[i]); }
#if MW_RELU_TAP
 reorderedLayerOutputTap(0);
#endif
 return; } void MWReLULayerImpl::allocateOutputData(int i) { if 
(UEESbUvbMihFnquvuFij) { MWTensor* ipTensor = getLayer()->getInputTensor(); 
MWTensor* opTensor = getLayer()->getOutputTensor(i); 
MWMkldnnUtils::getLayerMemoryPrimitive(opTensor, euppfEoiaoCTcVgRPVhA) 
->set_data_handle(ipTensor->getData<float>()); 
opTensor->setData(ipTensor->getData<float>()); } else { 
MWCNNLayerImpl::allocateOutputData(i); } } void 
MWReLULayerImpl::deallocateOutputData(int i) { MWTensor* op = 
getLayer()->getOutputTensor(i); float* data = 
getLayer()->getOutputTensor(i)->getData<float>(); if (data) { if 
((op->getopBufIndex() < 0) && !UEESbUvbMihFnquvuFij) { free(data); } 
op->setData((float*)NULL); } } MWNormLayerImpl::MWNormLayerImpl(MWCNNLayer* 
layer, MWTargetNetworkImpl* ntwk_impl, unsigned KHClOltUSuqFVVErSxVb, 
double AFQBkxwYGKLsACiDKwRM, double AHqhysOOIgbDpWZoPUFT, double ) : 
MWCNNLayerImpl(layer, ntwk_impl) , AFQBkxwYGKLsACiDKwRM(AFQBkxwYGKLsACiDKwRM) , 
AHqhysOOIgbDpWZoPUFT(AHqhysOOIgbDpWZoPUFT) , 
KHClOltUSuqFVVErSxVb(KHClOltUSuqFVVErSxVb) { } 
MWNormLayerImpl::~MWNormLayerImpl() { } void MWNormLayerImpl::propagateSize() { 
MWNormLayer* normLayer = static_cast<MWNormLayer*>(getLayer()); MWTensor* 
ipTensor = normLayer->getInputTensor(); int n = ipTensor->getBatchSize(); int c 
= ipTensor->getChannels(); int h = ipTensor->getHeight(); int w = 
ipTensor->getWidth(); lrn_src_memory = 
MWMkldnnUtils::getLayerMemoryPrimitive(ipTensor, euppfEoiaoCTcVgRPVhA); auto 
tmp_src_md = mkldnn::memory::desc({n, c * h * w}, 
mkldnn::memory::data_type::f32, mkldnn::memory::format_tag::nc); if (tmp_src_md 
== MWMkldnnUtils::getLayerMemoryPrimitive(ipTensor, 
euppfEoiaoCTcVgRPVhA)->get_desc()) { isReordered = true; mkldnn::memory::dims 
layerMemoryDimensions = {n, c, h, w}; lrn_src_md = 
std::unique_ptr<mkldnn::memory::desc>( new 
mkldnn::memory::desc(layerMemoryDimensions, mkldnn::memory::data_type::f32, 
mkldnn::memory::format_tag::nchw)); lrn_src_memory = 
std::make_shared<mkldnn::memory>(*lrn_src_md, *euppfEoiaoCTcVgRPVhA->eng); } 
setLayerMemory( std::make_shared<mkldnn::memory>(lrn_src_memory->get_desc(), 
*euppfEoiaoCTcVgRPVhA->eng)); lrn_d = 
std::unique_ptr<mkldnn::lrn_forward::desc>(new mkldnn::lrn_forward::desc( 
mkldnn::prop_kind::forward_inference, mkldnn::algorithm::lrn_across_channels, 
lrn_src_memory->get_desc(), KHClOltUSuqFVVErSxVb, 
(float)(AFQBkxwYGKLsACiDKwRM), (float)AHqhysOOIgbDpWZoPUFT, 1.0f)); lrn_pd = 
std::unique_ptr<mkldnn::lrn_forward::primitive_desc>( new 
mkldnn::lrn_forward::primitive_desc(*lrn_d, *euppfEoiaoCTcVgRPVhA->eng)); lrn = 
std::unique_ptr<mkldnn::lrn_forward>(new mkldnn::lrn_forward(*lrn_pd)); 
pipeline_memory_args.push_back( {{MKLDNN_ARG_FROM, *lrn_src_memory}, 
{MKLDNN_ARG_TO, *getLayerMemory()}}); pipeline.push_back(*lrn); return; } void 
MWNormLayerImpl::predict() { auto s = 
mkldnn::stream(*euppfEoiaoCTcVgRPVhA->eng); assert(pipeline.size() == 
pipeline_memory_args.size()); for (int i = 0; i < pipeline.size(); i++) { 
pipeline[i].execute(s, pipeline_memory_args[i]); }
#if MW_NORM_TAP
 reorderedLayerOutputTap(0);
#endif
 return; } void MWNormLayerImpl::allocateOutputData(int i) { MWTensor* ipTensor 
= getLayer()->getInputTensor(); if (isReordered) { 
lrn_src_memory->set_data_handle(ipTensor->getData<float>()); } 
MWCNNLayerImpl::allocateOutputData(i); } 
MWMaxPoolingLayerImpl::MWMaxPoolingLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl, int HJHXkKmgFxxIOsIvRRnF, int JgLfgHrHMEMmMYTettJF, int 
JwxFdqOKggeawILBfGgg, int KCudOrFMfgCzUPMcdePX, int ECTnqgWHyHCHCLBZlffd, int 
DqxLTLaJwwgQqmrtCDuu, int GnxRkpzrPZimKtYYHSuG, int GsZlHFuhbvjLtRMDjXnW, 
bool MEmIeGILUZNEWEagSzRk, int fvTCtkwXgyScJYogJVFU) : MWCNNLayerImpl(layer, 
ntwk_impl) , BLjrjqvCcCommiXWQLjs(MEmIeGILUZNEWEagSzRk) , 
HJHXkKmgFxxIOsIvRRnF(HJHXkKmgFxxIOsIvRRnF) , JgLfgHrHMEMmMYTettJF(JgLfgHrHMEMmMYTettJF) , 
ECTnqgWHyHCHCLBZlffd(ECTnqgWHyHCHCLBZlffd) , 
DqxLTLaJwwgQqmrtCDuu(DqxLTLaJwwgQqmrtCDuu) , 
GnxRkpzrPZimKtYYHSuG(GnxRkpzrPZimKtYYHSuG) , 
GsZlHFuhbvjLtRMDjXnW(GsZlHFuhbvjLtRMDjXnW) , 
JwxFdqOKggeawILBfGgg(JwxFdqOKggeawILBfGgg) , KCudOrFMfgCzUPMcdePX(KCudOrFMfgCzUPMcdePX) 
, fvTCtkwXgyScJYogJVFU(fvTCtkwXgyScJYogJVFU) { } 
MWMaxPoolingLayerImpl::~MWMaxPoolingLayerImpl() { } float* 
MWMaxPoolingLayerImpl::getIndexData() { return UKtMXCCqdjeyaVHabkxg; } 
std::shared_ptr<mkldnn::pooling_forward::primitive_desc> 
MWMaxPoolingLayerImpl::getPoolPrimitiveDesc() { return pool_pd; } 
mkldnn::memory::dims MWMaxPoolingLayerImpl::getPoolKernel() { return 
HtQBsWTCGEkpylRklilw; } mkldnn::memory::dims 
MWMaxPoolingLayerImpl::getPoolStrides() { return IwKnaBoXVubIRYcxEJLH; } 
mkldnn::memory::dims MWMaxPoolingLayerImpl::getPoolPadTL() { return 
IbSWJNMuIiKbocfQKqXb; } mkldnn::memory::dims 
MWMaxPoolingLayerImpl::getPoolPadBR() { return IAlDgIFcchbwRGBSfVfA; } void 
createMaxpoolingIndicesPrimitive( mkldnn::memory::desc pool_dst_md, 
std::shared_ptr<mkldnn::pooling_forward::desc> pool_d, 
std::shared_ptr<mkldnn::pooling_forward::primitive_desc>& pool_pd, 
std::shared_ptr<mkldnn::memory> srcMemory, std::shared_ptr<mkldnn::memory>& 
layerMemory, std::shared_ptr<mkldnn::memory>& indices_Memory, 
std::shared_ptr<mkldnn::pooling_forward::primitive>& pool, mkldnn::memory::dims 
HtQBsWTCGEkpylRklilw, mkldnn::memory::dims IwKnaBoXVubIRYcxEJLH, 
mkldnn::memory::dims IbSWJNMuIiKbocfQKqXb, mkldnn::memory::dims 
IAlDgIFcchbwRGBSfVfA) { pool_d = 
std::make_shared<mkldnn::pooling_forward::desc>( 
mkldnn::prop_kind::forward_training, mkldnn::algorithm::pooling_max, 
srcMemory->get_desc(), pool_dst_md, IwKnaBoXVubIRYcxEJLH, HtQBsWTCGEkpylRklilw, 
IbSWJNMuIiKbocfQKqXb, IAlDgIFcchbwRGBSfVfA); pool_pd = 
std::make_shared<mkldnn::pooling_forward::primitive_desc>(*pool_d, 
srcMemory->get_engine()); layerMemory = 
std::make_shared<mkldnn::memory>(pool_pd->dst_desc(), srcMemory->get_engine()); 
indices_Memory = std::make_shared<mkldnn::memory>(pool_pd->workspace_desc(), 
srcMemory->get_engine()); pool = 
std::make_shared<mkldnn::pooling_forward>(*pool_pd); } void 
createMaxpoolingPrimitive(mkldnn::memory::desc pool_dst_md, 
std::shared_ptr<mkldnn::pooling_forward::desc> pool_d, 
std::shared_ptr<mkldnn::pooling_forward::primitive_desc>& pool_pd, 
std::shared_ptr<mkldnn::memory> srcMemory, std::shared_ptr<mkldnn::memory>& 
layerMemory, std::shared_ptr<mkldnn::pooling_forward::primitive>& pool, 
mkldnn::memory::dims HtQBsWTCGEkpylRklilw, mkldnn::memory::dims 
IwKnaBoXVubIRYcxEJLH, mkldnn::memory::dims IbSWJNMuIiKbocfQKqXb, 
mkldnn::memory::dims IAlDgIFcchbwRGBSfVfA) { pool_d = 
std::make_shared<mkldnn::pooling_forward::desc>( 
mkldnn::prop_kind::forward_inference, mkldnn::algorithm::pooling_max, 
srcMemory->get_desc(), pool_dst_md, IwKnaBoXVubIRYcxEJLH, HtQBsWTCGEkpylRklilw, 
IbSWJNMuIiKbocfQKqXb, IAlDgIFcchbwRGBSfVfA); pool_pd = 
std::make_shared<mkldnn::pooling_forward::primitive_desc>(*pool_d, 
srcMemory->get_engine()); layerMemory = 
std::make_shared<mkldnn::memory>(pool_pd->dst_desc(), srcMemory->get_engine()); 
pool = std::make_shared<mkldnn::pooling_forward>(*pool_pd); } void 
MWMaxPoolingLayerImpl::propagateSize() { MWMaxPoolingLayer* maxPoolLayer = 
static_cast<MWMaxPoolingLayer*>(getLayer()); MWTensor* ipTensor = 
maxPoolLayer->getInputTensor(); MWTensor* opTensor = 
maxPoolLayer->getOutputTensor(); int n = ipTensor->getBatchSize(); int c = 
ipTensor->getChannels(); int h = ipTensor->getHeight(); int w = 
ipTensor->getWidth(); mkldnn::memory::dims pool_usr_tz = {n, c, h, w}; 
mkldnn::memory::dims pool_dst_tz = {n, c, opTensor->getHeight(), 
opTensor->getWidth()}; HtQBsWTCGEkpylRklilw = {HJHXkKmgFxxIOsIvRRnF, 
JgLfgHrHMEMmMYTettJF}; IwKnaBoXVubIRYcxEJLH = {JwxFdqOKggeawILBfGgg, 
KCudOrFMfgCzUPMcdePX}; IbSWJNMuIiKbocfQKqXb = {ECTnqgWHyHCHCLBZlffd, 
GnxRkpzrPZimKtYYHSuG}; IAlDgIFcchbwRGBSfVfA = {DqxLTLaJwwgQqmrtCDuu, 
GsZlHFuhbvjLtRMDjXnW}; auto pool_dst_md = mkldnn::memory::desc({pool_dst_tz}, 
mkldnn::memory::data_type::f32, mkldnn::memory::format_tag::any); if 
(BLjrjqvCcCommiXWQLjs) { createMaxpoolingIndicesPrimitive( pool_dst_md, 
pool_d, pool_pd, MWMkldnnUtils::getLayerMemoryPrimitive(ipTensor, 
euppfEoiaoCTcVgRPVhA), layerMemory[0], layerMemory[1], pool, 
HtQBsWTCGEkpylRklilw, IwKnaBoXVubIRYcxEJLH, IbSWJNMuIiKbocfQKqXb, 
IAlDgIFcchbwRGBSfVfA); fjfzkUfcCOqjrkAVGfuc = 
static_cast<int>(getLayerMemory(1)->get_desc().get_size() / 4); 
pipeline_memory_args.push_back({{MKLDNN_ARG_FROM, 
*MWMkldnnUtils::getLayerMemoryPrimitive( ipTensor, euppfEoiaoCTcVgRPVhA)}, 
{MKLDNN_ARG_DST, *layerMemory[0]}, {MKLDNN_ARG_WORKSPACE, *layerMemory[1]}}); } 
else { createMaxpoolingPrimitive( pool_dst_md, pool_d, pool_pd, 
MWMkldnnUtils::getLayerMemoryPrimitive(ipTensor, euppfEoiaoCTcVgRPVhA), 
layerMemory[0], pool, HtQBsWTCGEkpylRklilw, IwKnaBoXVubIRYcxEJLH, 
IbSWJNMuIiKbocfQKqXb, IAlDgIFcchbwRGBSfVfA); 
pipeline_memory_args.push_back({{MKLDNN_ARG_FROM, 
*MWMkldnnUtils::getLayerMemoryPrimitive( ipTensor, euppfEoiaoCTcVgRPVhA)}, 
{MKLDNN_ARG_DST, *layerMemory[0]}}); } pipeline.push_back(*pool); return; } 
void MWMaxPoolingLayerImpl::predict() { auto s = 
mkldnn::stream(*euppfEoiaoCTcVgRPVhA->eng); assert(pipeline.size() == 
pipeline_memory_args.size()); for (int i = 0; i < pipeline.size(); i++) { 
pipeline[i].execute(s, pipeline_memory_args[i]); }
#if MW_POOL_TAP
 reorderedLayerOutputTap(0);
#endif
 return; } MWFCLayerImpl::~MWFCLayerImpl() { } 
MWFCLayerImpl::MWFCLayerImpl(MWCNNLayer* layer, MWTargetNetworkImpl* ntwk_impl, 
int WIxRBCJtmETvfxpuRuus, int oJUVMnJggjhEdQLWzIUC, const char* 
xHViLEwTujGGrPZZgmbF, const char* QVgVGfoCXYiYXzPhvVPX) : 
MWCNNLayerImpl(layer, ntwk_impl) { vIWQzNvYZSuxmOTVDFhU = 
(float*)calloc(WIxRBCJtmETvfxpuRuus * oJUVMnJggjhEdQLWzIUC, sizeof(float)); 
PmFfARVzoHVAYkfpuvqK = (float*)calloc(oJUVMnJggjhEdQLWzIUC, sizeof(float)); 
loadWeights(xHViLEwTujGGrPZZgmbF, WIxRBCJtmETvfxpuRuus * 
oJUVMnJggjhEdQLWzIUC); loadBias(QVgVGfoCXYiYXzPhvVPX, oJUVMnJggjhEdQLWzIUC); 
mkldnn::memory::dims fc_bias_tz = {oJUVMnJggjhEdQLWzIUC}; bias_md = 
std::make_shared<mkldnn::memory::desc>(fc_bias_tz, 
mkldnn::memory::data_type::f32, mkldnn::memory::format_tag::x); bias = 
std::make_shared<mkldnn::memory>(*bias_md, *euppfEoiaoCTcVgRPVhA->eng, 
PmFfARVzoHVAYkfpuvqK); } void MWFCLayerImpl::propagateSize() { MWFCLayer* fcLayer = 
static_cast<MWFCLayer*>(getLayer()); MWTensor* ipTensor = 
fcLayer->getInputTensor(); MWTensor* opTensor = fcLayer->getOutputTensor(); 
mkldnn::memory::dims fc_src_tz = {ipTensor->getBatchSize(), 
ipTensor->getChannels(), ipTensor->getHeight(), ipTensor->getWidth()}; 
mkldnn::memory::dims dim1 = {opTensor->getChannels(), ipTensor->getChannels()}; 
mkldnn::memory::dims dim2 = {opTensor->getChannels(), ipTensor->getChannels(), 
ipTensor->getHeight(), ipTensor->getWidth()}; fc_src_memory = 
MWMkldnnUtils::getLayerMemoryPrimitive(ipTensor, euppfEoiaoCTcVgRPVhA); bool 
isFormatNC = MWMkldnnUtils::checkformatType(fc_src_memory->get_desc(), 
ipTensor, mkldnn::memory::format_tag::nc); bool dimNCHWFlag = 
((ipTensor->getHeight() != 1) && (ipTensor->getWidth() != 1)) || !(isFormatNC); 
mkldnn::memory::dims fc_weights_tz = (dimNCHWFlag) ? dim2 : dim1; if 
(dimNCHWFlag) { weights_md = std::make_shared<mkldnn::memory::desc>( 
fc_weights_tz, mkldnn::memory::data_type::f32, 
mkldnn::memory::format_tag::oihw); } else { weights_md = 
std::make_shared<mkldnn::memory::desc>( fc_weights_tz, 
mkldnn::memory::data_type::f32, mkldnn::memory::format_tag::nc); } 
prepareWeights(); weights = std::make_shared<mkldnn::memory>(*weights_md, 
*euppfEoiaoCTcVgRPVhA->eng, vIWQzNvYZSuxmOTVDFhU); std::shared_ptr<mkldnn::memory::desc> 
fc_src_md = std::make_shared<mkldnn::memory::desc>(fc_src_memory->get_desc()); 
if (dimNCHWFlag) { fc_src_md = std::make_shared<mkldnn::memory::desc>( 
fc_src_tz, mkldnn::memory::data_type::f32, mkldnn::memory::format_tag::any); } 
auto fc_weights_md = mkldnn::memory::desc({fc_weights_tz}, 
mkldnn::memory::data_type::f32, mkldnn::memory::format_tag::any); auto 
fc_bias_md = mkldnn::memory::desc( {opTensor->getChannels()}, 
mkldnn::memory::data_type::f32, mkldnn::memory::format_tag::any); auto 
fc_dst_md = mkldnn::memory::desc({ipTensor->getBatchSize(), 
opTensor->getChannels()}, mkldnn::memory::data_type::f32, 
mkldnn::memory::format_tag::any); ip_d = 
std::unique_ptr<mkldnn::inner_product_forward::desc>( new 
mkldnn::inner_product_forward::desc(mkldnn::prop_kind::forward, *fc_src_md, 
fc_weights_md, fc_bias_md, fc_dst_md)); ip_pd = 
std::unique_ptr<mkldnn::inner_product_forward::primitive_desc>( new 
mkldnn::inner_product_forward::primitive_desc(*ip_d, 
*euppfEoiaoCTcVgRPVhA->eng)); if (dimNCHWFlag) { if 
(mkldnn::memory::desc(ip_pd->src_desc()) != fc_src_memory->get_desc()) { 
fc_src_memory = std::make_shared<mkldnn::memory>(ip_pd->src_desc(), 
*euppfEoiaoCTcVgRPVhA->eng); fc_reorder_src = 
std::unique_ptr<mkldnn::reorder>(new mkldnn::reorder( 
*MWMkldnnUtils::getLayerMemoryPrimitive(ipTensor, euppfEoiaoCTcVgRPVhA), 
*fc_src_memory)); pipeline_memory_args.push_back( {{MKLDNN_ARG_FROM, 
*MWMkldnnUtils::getLayerMemoryPrimitive(ipTensor, euppfEoiaoCTcVgRPVhA)}, 
{MKLDNN_ARG_TO, *fc_src_memory}}); pipeline.push_back(*fc_reorder_src); } } 
fc_weights_memory = weights; if (mkldnn::memory::desc(ip_pd->weights_desc()) != 
fc_weights_memory->get_desc()) { fc_weights_memory = 
std::make_shared<mkldnn::memory>(ip_pd->weights_desc(), 
*euppfEoiaoCTcVgRPVhA->eng); fc_reorder_weights = 
std::unique_ptr<mkldnn::reorder>(new mkldnn::reorder(*weights, 
*fc_weights_memory)); pipeline_weights.push_back(*fc_reorder_weights); auto s = 
mkldnn::stream(*euppfEoiaoCTcVgRPVhA->eng); 
pipeline_weights_memory_args.push_back( {{MKLDNN_ARG_FROM, *weights}, 
{MKLDNN_ARG_TO, *fc_weights_memory}}); assert(pipeline_weights.size() == 
pipeline_weights_memory_args.size()); for (int i = 0; i < 
pipeline_weights.size(); i++) { pipeline_weights[i].execute(s, 
pipeline_weights_memory_args[i]); } } 
setLayerMemory(std::make_shared<mkldnn::memory>(ip_pd->dst_desc(), 
*euppfEoiaoCTcVgRPVhA->eng)); ip = 
std::unique_ptr<mkldnn::inner_product_forward>(new 
mkldnn::inner_product_forward(*ip_pd)); 
pipeline_memory_args.push_back({{MKLDNN_ARG_SRC, *fc_src_memory}, 
{MKLDNN_ARG_WEIGHTS, *fc_weights_memory}, {MKLDNN_ARG_BIAS, *bias}, 
{MKLDNN_ARG_DST, *getLayerMemory()}}); pipeline.push_back(*ip); return; } void 
MWFCLayerImpl::predict() { auto s = mkldnn::stream(*euppfEoiaoCTcVgRPVhA->eng); 
assert(pipeline.size() == pipeline_memory_args.size()); for (int i = 0; i < 
pipeline.size(); i++) { pipeline[i].execute(s, pipeline_memory_args[i]); }
#if MW_FC_TAP
 reorderedLayerOutputTap(0);
#endif
 return; } void MWFCLayerImpl::loadWeights(const char* RqCYCrGsNvzKYrRMXbsI, int 
eybNKlJCSDUvsznWynwK) { FILE* SGsAudmgjmvcUXzzrUtf = 
MWCNNLayer::openBinaryFile(RqCYCrGsNvzKYrRMXbsI); call_fread(vIWQzNvYZSuxmOTVDFhU, 
sizeof(float), eybNKlJCSDUvsznWynwK, SGsAudmgjmvcUXzzrUtf, RqCYCrGsNvzKYrRMXbsI); 
fclose(SGsAudmgjmvcUXzzrUtf); return; } void MWFCLayerImpl::prepareWeights() { 
MWTensor* ipTensor = getLayer()->getInputTensor(); MWTensor* opTensor = 
getLayer()->getOutputTensor(); int AwZQzUhuWVLGrWgLHRuM = 
ipTensor->getHeight(); int AzTsxYcYjIEJsGQbeYHm = ipTensor->getWidth(); int 
CpMjJjtGOeWOzwxpAAQP = AwZQzUhuWVLGrWgLHRuM * AzTsxYcYjIEJsGQbeYHm 
* ipTensor->getChannels(); int eybNKlJCSDUvsznWynwK = CpMjJjtGOeWOzwxpAAQP * 
opTensor->getChannels(); if (AwZQzUhuWVLGrWgLHRuM != 1 && 
AzTsxYcYjIEJsGQbeYHm != 1) { float* veFyKKHbdqBIvQLYBqfF = 
(float*)malloc(sizeof(float) * AwZQzUhuWVLGrWgLHRuM * 
AzTsxYcYjIEJsGQbeYHm); for (int k = 0; k < eybNKlJCSDUvsznWynwK / 
AwZQzUhuWVLGrWgLHRuM / AzTsxYcYjIEJsGQbeYHm; k++) { for (int i = 0; i < 
AwZQzUhuWVLGrWgLHRuM * AzTsxYcYjIEJsGQbeYHm; i++) { veFyKKHbdqBIvQLYBqfF[i] = 
vIWQzNvYZSuxmOTVDFhU[k * AwZQzUhuWVLGrWgLHRuM * AzTsxYcYjIEJsGQbeYHm + i]; } for 
(int j = 0; j < AwZQzUhuWVLGrWgLHRuM; j++) for (int i = 0; i < 
AzTsxYcYjIEJsGQbeYHm; i++) { vIWQzNvYZSuxmOTVDFhU[k * AwZQzUhuWVLGrWgLHRuM * 
AzTsxYcYjIEJsGQbeYHm + j * AzTsxYcYjIEJsGQbeYHm + i] = veFyKKHbdqBIvQLYBqfF[j + i 
* AwZQzUhuWVLGrWgLHRuM]; } } free(veFyKKHbdqBIvQLYBqfF); } return; } void 
MWFCLayerImpl::loadBias(const char* RqCYCrGsNvzKYrRMXbsI, int eybNKlJCSDUvsznWynwK) { 
FILE* SGsAudmgjmvcUXzzrUtf = MWCNNLayer::openBinaryFile(RqCYCrGsNvzKYrRMXbsI); 
call_fread(PmFfARVzoHVAYkfpuvqK, sizeof(float), eybNKlJCSDUvsznWynwK, SGsAudmgjmvcUXzzrUtf, 
RqCYCrGsNvzKYrRMXbsI); fclose(SGsAudmgjmvcUXzzrUtf); return; } void 
MWFCLayerImpl::cleanup() { if (vIWQzNvYZSuxmOTVDFhU) { free(vIWQzNvYZSuxmOTVDFhU); vIWQzNvYZSuxmOTVDFhU 
= NULL; } if (PmFfARVzoHVAYkfpuvqK) { free(PmFfARVzoHVAYkfpuvqK); PmFfARVzoHVAYkfpuvqK = NULL; } 
return; } MWSoftmaxLayerImpl::MWSoftmaxLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl) : MWCNNLayerImpl(layer, ntwk_impl) { } 
MWSoftmaxLayerImpl::~MWSoftmaxLayerImpl() { } mkldnn::memory::desc 
MWSoftmaxLayerImpl::getlayerMemoryPrimDescriptor( 
std::shared_ptr<mkldnn::memory> ipLayerMemory, int batchSize, int channels, int 
height, int width, bool isNCLayout) { if (isNCLayout) { mkldnn::memory::dims 
layerMemoryDimensions = {batchSize, channels}; return 
(mkldnn::memory::desc(layerMemoryDimensions, mkldnn::memory::data_type::f32, 
mkldnn::memory::format_tag::nc)); } else { mkldnn::memory::dims 
layerMemoryDimensions = {batchSize, channels, height, width}; return 
(mkldnn::memory::desc(layerMemoryDimensions, mkldnn::memory::data_type::f32, 
mkldnn::memory::format_tag::nchw)); } } void 
MWSoftmaxLayerImpl::propagateSize() { MWTensor* ipTensor = 
getLayer()->getInputTensor(0); int batchSize = ipTensor->getBatchSize(); int 
channels = ipTensor->getChannels(); int height = ipTensor->getHeight(); int 
width = ipTensor->getWidth(); mkldnn::memory::format_tag outputDataFormat = 
mkldnn::memory::format_tag::nchw; auto prevMemoryDesc = 
MWMkldnnUtils::getLayerMemoryPrimitive(ipTensor, 
euppfEoiaoCTcVgRPVhA)->get_desc(); bool isInpMemPrimitiveNC = 
MWMkldnnUtils::checkformatType(prevMemoryDesc, ipTensor, 
mkldnn::memory::format_tag::nc); bool isInpMemPrimitiveNCHW = 
MWMkldnnUtils::checkformatType(prevMemoryDesc, ipTensor, 
mkldnn::memory::format_tag::nchw); if (!(isInpMemPrimitiveNC || 
isInpMemPrimitiveNCHW)) { MWMkldnnUtils::configureReorder(this, ipTensor, 
outputDataFormat); pipeline.push_back(*getReorderPrim()); 
pipeline_memory_args.push_back({{MKLDNN_ARG_FROM, 
*MWMkldnnUtils::getLayerMemoryPrimitive( ipTensor, euppfEoiaoCTcVgRPVhA)}, 
{MKLDNN_ARG_TO, *getReorderLayerMemory()}}); } else { setReorderLayerMemory( 
MWMkldnnUtils::getLayerMemoryPrimitive(ipTensor, euppfEoiaoCTcVgRPVhA)); } auto 
layerMemoryDescriptor = getlayerMemoryPrimDescriptor( getReorderLayerMemory(), 
batchSize, channels, height, width, isInpMemPrimitiveNC); softmax_d = 
std::unique_ptr<mkldnn::softmax_forward::desc>(new 
mkldnn::softmax_forward::desc( mkldnn::prop_kind::forward_inference, 
layerMemoryDescriptor, 1)); softmax_pd = 
std::unique_ptr<mkldnn::softmax_forward::primitive_desc>( new 
mkldnn::softmax_forward::primitive_desc(*softmax_d, 
*euppfEoiaoCTcVgRPVhA->eng)); 
setLayerMemory(std::make_shared<mkldnn::memory>(getReorderLayerMemory()->get_desc(), 
*euppfEoiaoCTcVgRPVhA->eng)); softmax = 
std::unique_ptr<mkldnn::softmax_forward>(new 
mkldnn::softmax_forward(*softmax_pd)); pipeline_memory_args.push_back( 
{{MKLDNN_ARG_FROM, *getReorderLayerMemory()}, {MKLDNN_ARG_TO, 
*getLayerMemory()}}); pipeline.push_back(*softmax); return; } void 
MWSoftmaxLayerImpl::predict() { auto s = 
mkldnn::stream(*euppfEoiaoCTcVgRPVhA->eng); assert(pipeline.size() == 
pipeline_memory_args.size()); for (int i = 0; i < pipeline.size(); i++) { 
pipeline[i].execute(s, pipeline_memory_args[i]); }
#if MW_SFMX_TAP
 reorderedLayerOutputTap(0);
#endif
 return; } MWAvgPoolingLayerImpl::MWAvgPoolingLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl, int HJHXkKmgFxxIOsIvRRnF, int JgLfgHrHMEMmMYTettJF, int 
JwxFdqOKggeawILBfGgg, int KCudOrFMfgCzUPMcdePX, int ECTnqgWHyHCHCLBZlffd, int 
DqxLTLaJwwgQqmrtCDuu, int GnxRkpzrPZimKtYYHSuG, int GsZlHFuhbvjLtRMDjXnW) : 
MWCNNLayerImpl(layer, ntwk_impl) , HJHXkKmgFxxIOsIvRRnF(HJHXkKmgFxxIOsIvRRnF) , 
JgLfgHrHMEMmMYTettJF(JgLfgHrHMEMmMYTettJF) , JwxFdqOKggeawILBfGgg(JwxFdqOKggeawILBfGgg) , 
KCudOrFMfgCzUPMcdePX(KCudOrFMfgCzUPMcdePX) , 
ECTnqgWHyHCHCLBZlffd(ECTnqgWHyHCHCLBZlffd) , 
DqxLTLaJwwgQqmrtCDuu(DqxLTLaJwwgQqmrtCDuu) , 
GnxRkpzrPZimKtYYHSuG(GnxRkpzrPZimKtYYHSuG) , 
GsZlHFuhbvjLtRMDjXnW(GsZlHFuhbvjLtRMDjXnW) { } 
MWAvgPoolingLayerImpl::~MWAvgPoolingLayerImpl() { } void 
MWAvgPoolingLayerImpl::propagateSize() { MWAvgPoolingLayer* avgpoolLayer = 
static_cast<MWAvgPoolingLayer*>(getLayer()); MWTensor* ipTensor = 
avgpoolLayer->getInputTensor(0); MWTensor* opTensor = 
avgpoolLayer->getOutputTensor(0); if ((HJHXkKmgFxxIOsIvRRnF == -1) && 
(JgLfgHrHMEMmMYTettJF == -1)) { HJHXkKmgFxxIOsIvRRnF = ipTensor->getHeight(); 
JgLfgHrHMEMmMYTettJF = ipTensor->getWidth(); } int n = ipTensor->getBatchSize(); 
int c = ipTensor->getChannels(); int h = ipTensor->getHeight(); int w = 
ipTensor->getWidth(); mkldnn::memory::dims pool_usr_tz = {n, c, h, w}; 
mkldnn::memory::dims pool_dst_tz = {n, c, opTensor->getHeight(), 
opTensor->getWidth()}; mkldnn::memory::dims pool_kernel = {HJHXkKmgFxxIOsIvRRnF, 
JgLfgHrHMEMmMYTettJF}; mkldnn::memory::dims IwKnaBoXVubIRYcxEJLH = 
{JwxFdqOKggeawILBfGgg, KCudOrFMfgCzUPMcdePX}; mkldnn::memory::dims 
IbSWJNMuIiKbocfQKqXb = {ECTnqgWHyHCHCLBZlffd, GnxRkpzrPZimKtYYHSuG}; 
mkldnn::memory::dims IAlDgIFcchbwRGBSfVfA = {DqxLTLaJwwgQqmrtCDuu, 
GsZlHFuhbvjLtRMDjXnW}; auto pool_dst_md = mkldnn::memory::desc({pool_dst_tz}, 
mkldnn::memory::data_type::f32, mkldnn::memory::format_tag::any); pool_d = 
std::unique_ptr<mkldnn::pooling_forward::desc>(new 
mkldnn::pooling_forward::desc( mkldnn::prop_kind::forward_inference, 
mkldnn::algorithm::pooling_avg_include_padding, 
MWMkldnnUtils::getLayerMemoryPrimitive(ipTensor, 
euppfEoiaoCTcVgRPVhA)->get_desc(), pool_dst_md, IwKnaBoXVubIRYcxEJLH, 
pool_kernel, IbSWJNMuIiKbocfQKqXb, IAlDgIFcchbwRGBSfVfA)); pool_pd = 
std::unique_ptr<mkldnn::pooling_forward::primitive_desc>( new 
mkldnn::pooling_forward::primitive_desc(*pool_d, *euppfEoiaoCTcVgRPVhA->eng)); 
setLayerMemory( std::make_shared<mkldnn::memory>(pool_pd->dst_desc(), 
*euppfEoiaoCTcVgRPVhA->eng)); pool = 
std::unique_ptr<mkldnn::pooling_forward>(new 
mkldnn::pooling_forward(*pool_pd)); pipeline.push_back(*pool); 
pipeline_memory_args.push_back( {{MKLDNN_ARG_FROM, 
*MWMkldnnUtils::getLayerMemoryPrimitive(ipTensor, euppfEoiaoCTcVgRPVhA)}, 
{MKLDNN_ARG_TO, *getLayerMemory()}}); } void MWAvgPoolingLayerImpl::predict() { 
auto s = mkldnn::stream(*euppfEoiaoCTcVgRPVhA->eng); assert(pipeline.size() == 
pipeline_memory_args.size()); for (int i = 0; i < pipeline.size(); i++) { 
pipeline[i].execute(s, pipeline_memory_args[i]); }
#if MW_POOL_TAP
 reorderedLayerOutputTap(0);
#endif
 return; } MWOutputLayerImpl::MWOutputLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl) : MWCNNLayerImpl(layer, ntwk_impl) , 
doReorderFlag(false) { } MWOutputLayerImpl::~MWOutputLayerImpl() { } void 
MWOutputLayerImpl::propagateSize() { MWOutputLayer* opLayer = 
static_cast<MWOutputLayer*>(getLayer()); MWTensor* ipTensor = 
opLayer->getInputTensor(0); doReorderFlag = false; mkldnn::memory::format_tag 
outputDataFormat = mkldnn::memory::format_tag::nchw; auto prevMemoryDesc = 
MWMkldnnUtils::getLayerMemoryPrimitive(ipTensor, 
euppfEoiaoCTcVgRPVhA)->get_desc(); format_type formatTag = 
MWMkldnnUtils::getformatType(prevMemoryDesc, ipTensor); if ((formatTag != 
format_type::NC_FORMAT) && (formatTag != format_type::NCHW_FORMAT)) { 
doReorderFlag = true; MWMkldnnUtils::configureReorder(this, ipTensor, 
outputDataFormat); pipeline_memory_args.push_back({{MKLDNN_ARG_FROM, 
*MWMkldnnUtils::getLayerMemoryPrimitive( ipTensor, euppfEoiaoCTcVgRPVhA)}, 
{MKLDNN_ARG_TO, *getReorderLayerMemory()}}); 
pipeline.push_back(*getReorderPrim()); } } void MWOutputLayerImpl::predict() { 
if (doReorderFlag) { auto s = mkldnn::stream(*euppfEoiaoCTcVgRPVhA->eng); 
assert(pipeline.size() == pipeline_memory_args.size()); for (int i = 0; i < 
pipeline.size(); i++) { pipeline[i].execute(s, pipeline_memory_args[i]); } } } 
void MWOutputLayerImpl::allocateOutputData(int i) { MWTensor* ipTensor = 
getLayer()->getInputTensor(0); MWTensor* opTensor = 
getLayer()->getOutputTensor(i); if (doReorderFlag) { 
opTensor->setData((float*)getReorderLayerMemory()->get_data_handle()); } else { 
opTensor->setData(ipTensor->getData<float>()); } }