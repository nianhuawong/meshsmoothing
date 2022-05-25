/* Copyright 2017-2018 The MathWorks, Inc. */

#ifndef _MKLDNN_UTILS_
#define _MKLDNN_UTILS_

#include <map>
#include <vector>
class MWTensor;
class MWCNNLayer;
class MWCNNLayerImpl;
class MWTargetNetworkImpl;

#include "mkldnn.hpp"

enum class format_type {NC_FORMAT, NCHW_FORMAT, NCHW8C_FORMAT, UNKNOWN_FORMAT};

class MWMkldnnUtils {

  public:
    MWMkldnnUtils() {
    }
    ~MWMkldnnUtils() {
    }

    static void configureReorder(MWCNNLayerImpl*,
                                 MWTensor* srcTensor,
                                 mkldnn::memory::format_tag dstDataFormat, int index=0);
    static std::shared_ptr<mkldnn::memory> getLayerMemoryPrimitive(MWTensor* aTensor,
                                                                   MWTargetNetworkImpl* ntwkImpl);
    static format_type getformatType(mkldnn::memory::desc inputdesc,
                             MWTensor* srcTensor);
    
    static bool checkformatType(mkldnn::memory::desc inputdesc,
                                MWTensor* srcTensor,
                                mkldnn::memory::format_tag dstDataFormat);
                                     
};

#endif
