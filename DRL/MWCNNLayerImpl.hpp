/* Copyright 2017-2019 The MathWorks, Inc. */

#ifndef CNN_API_IMPL
#define CNN_API_IMPL

#include <map>
#include <vector>
#include "cnn_api.hpp"
class MWTargetNetworkImpl;
#include "mkldnn.hpp"


/*If MW_LAYERS_TAP is enabled, it will tap layer wise output */
#define MW_LAYERS_TAP 0

#if MW_LAYERS_TAP
extern int tap_count;
extern void mw_interm_tap(float* inp, int size, int count);


#define MW_INPUT_TAP 1
#define MW_CONV_TAP 1
#define MW_RELU_TAP 1
#define MW_NORM_TAP 1
#define MW_POOL_TAP 1
#define MW_UNPOOL_TAP 1
#define MW_FC_TAP 1
#define MW_SFMX_TAP 1
#define MW_ADDITION_TAP 1
#define MW_CONCAT_TAP 1
#define MW_TRANSPOSEDCONV_TAP 1
#define MW_SCALING_TAP 1
#define MW_CROP2D_TAP 1
#define MW_YOLOREORG_TAP 1
#define MW_YOLOEXTRACT_TAP 1
#define MW_SIGMOID_TAP 1
#define MW_EXP_TAP 1
#define MW_YOLOSFMAX_TAP 1
#define MW_FLATTEN_CSTYLE_TAP 1
#define MW_TANH_TAP 1
#define MW_ZEROPAD_TAP 1
#define MW_FLATTEN_TAP 1
#define MW_SPLIT_LAYER 1
#define MW_AFFINE_TAP 1
#define MW_SSDMERGE_TAP 1
#define MW_ELU_TAP 1
#else
#define MW_INPUT_TAP 0
#define MW_CONV_TAP 0
#define MW_RELU_TAP 0
#define MW_NORM_TAP 0
#define MW_POOL_TAP 0
#define MW_UNPOOL_TAP 0
#define MW_FC_TAP 0
#define MW_SFMX_TAP 0
#define MW_ADDITION_TAP 0
#define MW_CONCAT_TAP 0
#define MW_TRANSPOSEDCONV_TAP 0
#define MW_SCALING_TAP 0
#define MW_CROP2D_TAP 0
#define MW_YOLOREORG_TAP 0
#define MW_YOLOEXTRACT_TAP 0
#define MW_SIGMOID_TAP 0
#define MW_EXP_TAP 0
#define MW_YOLOSFMAX_TAP 0
#define MW_FLATTEN_CSTYLE_TAP 0
#define MW_TANH_TAP 0
#define MW_ZEROPAD_TAP 0
#define MW_FLATTEN_TAP 0
#define MW_SPLIT_LAYER 0
#define MW_AFFINE_TAP 0
#define MW_SSDMERGE_TAP 0
#define MW_ELU_TAP 0
#endif    

class MWCNNLayerImpl {
  public:
    MWCNNLayerImpl(MWCNNLayer* layer, MWTargetNetworkImpl* ntwk_impl);
    virtual ~MWCNNLayerImpl() {
    }
    virtual void predict() {
    }
    virtual void cleanup() {
    }
    virtual void postSetup() {
    }
    virtual void propagateSize() = 0;
    virtual void allocate() {
    }
    virtual void deallocate() {
    }
    virtual void allocateOutputData(int);
    virtual void deallocateOutputData(int);
    template <typename T>
    T* getData(int index = 0);

  public:
    MWCNNLayer* getLayer() {
        return bMAyVFGSPDjmUbziYLAy;
    }
    std::shared_ptr<mkldnn::memory> getLayerMemory(int index = 0);
    void setLayerMemory(std::shared_ptr<mkldnn::memory> other, int index = 0);

    std::shared_ptr<mkldnn::memory> getReorderLayerMemory(int index = 0) ;

    std::shared_ptr<mkldnn::reorder> getReorderPrim(int index = 0);

    MWTargetNetworkImpl* getTargetImpl() {
        return euppfEoiaoCTcVgRPVhA;
    }
    void setReorderLayerMemory(std::shared_ptr<mkldnn::memory> other, int index = 0);
    void setReorderPrim(std::shared_ptr<mkldnn::reorder> other, int index = 0);
    void reorderedLayerOutputTap(int);

  protected:
    MWCNNLayer* bMAyVFGSPDjmUbziYLAy;
    MWTargetNetworkImpl* euppfEoiaoCTcVgRPVhA;

    /*mkldnn related config params*/
    std::map<int, std::shared_ptr<mkldnn::memory>> layerMemory;
    std::map<int, std::shared_ptr<mkldnn::memory>> reorderLayerMemory;
    std::map<int, std::shared_ptr<mkldnn::reorder>> reorderPrim;  // to handle the reorder

    std::vector<mkldnn::primitive> pipeline;

    std::vector<std::unordered_map<int, mkldnn::memory>> pipeline_memory_args;
    std::vector<std::unordered_map<int, mkldnn::memory>> pipeline_weights_memory_args;
    std::vector<mkldnn::primitive> pipeline_weights;

};


template <typename T>
T* MWCNNLayerImpl::getData(int index) {
    T* data = getLayer()->getOutputTensor(index)->getData<T>();
    assert(data);
    return data;
}

class MWInputLayerImpl : public MWCNNLayerImpl {
  private:
    int bDTIjtxZiSHtjwzgEluE;
    std::vector<float>* LtEgcYoEYjkrWuohutgw;

  public:
    MWInputLayerImpl(MWCNNLayer* layer,
                     MWTargetNetworkImpl* ntwk_impl,
                     int,
                     int,
                     int,
                     int,
                     const char* avg_file_name);
    ~MWInputLayerImpl();
    void predict();
    void cleanup();
    void propagateSize();

  private:
    void loadAvg(const char*, int, int);
};

// ReLULayer
class MWReLULayerImpl : public MWCNNLayerImpl {
  public:
    MWReLULayerImpl(MWCNNLayer*, MWTargetNetworkImpl*, int);
    ~MWReLULayerImpl();

    void predict();
    void deallocateOutputData(int);
    void allocateOutputData(int);
    void propagateSize();

  private:
    int UEESbUvbMihFnquvuFij;

    // ReLU Layer related config params
    std::unique_ptr<mkldnn::eltwise_forward::desc> relu_d;
    std::unique_ptr<mkldnn::eltwise_forward::primitive_desc> relu_pd;
    std::unique_ptr<mkldnn::eltwise_forward::primitive> relu;
};


// CrossChannelNormalizationLayer
class MWNormLayerImpl : public MWCNNLayerImpl {
  public:
    MWNormLayerImpl(MWCNNLayer*, MWTargetNetworkImpl*, unsigned, double, double, double);
    ~MWNormLayerImpl();

    void predict();
    void allocateOutputData(int);
    void propagateSize();

  private:
    double AFQBkxwYGKLsACiDKwRM;
    double AHqhysOOIgbDpWZoPUFT;
    unsigned KHClOltUSuqFVVErSxVb;
    std::unique_ptr<mkldnn::memory::desc> lrn_src_md;
    std::shared_ptr<mkldnn::memory> lrn_src_memory;
    bool isReordered;

    // NormLayer related config params
    std::unique_ptr<mkldnn::lrn_forward::desc> lrn_d;
    std::unique_ptr<mkldnn::lrn_forward::primitive_desc> lrn_pd;
    std::unique_ptr<mkldnn::lrn_forward::primitive> lrn;
};

class MWMaxPoolingLayerImpl : public MWCNNLayerImpl {
  public:
    // Create MaxPooling2DLayer with PoolSize = [ PoolH PoolW ]
    //                                Stride = [ StrideH StrideW ]
    //                               Padding = [ PaddingH_T PaddingH_B PaddingW_L PaddingW_R ]
    MWMaxPoolingLayerImpl(MWCNNLayer*,
                          MWTargetNetworkImpl*,
                          int,
                          int,
                          int,
                          int,
                          int,
                          int,
                          int,
                          int,
                          bool,
                          int);
    ~MWMaxPoolingLayerImpl();


  public:
    void predict();
    float* getIndexData();
    void propagateSize();

    std::shared_ptr<mkldnn::pooling_forward::primitive_desc> getPoolPrimitiveDesc();

    mkldnn::memory::dims getPoolKernel();
    mkldnn::memory::dims getPoolStrides();
    mkldnn::memory::dims getPoolPadTL();
    mkldnn::memory::dims getPoolPadBR();


  private:
    /*mkldnn configurations*/

    float* UKtMXCCqdjeyaVHabkxg; // indices output
    int fjfzkUfcCOqjrkAVGfuc;
    bool BLjrjqvCcCommiXWQLjs;

    int HJHXkKmgFxxIOsIvRRnF;
    int JgLfgHrHMEMmMYTettJF;
    int JwxFdqOKggeawILBfGgg;
    int KCudOrFMfgCzUPMcdePX;
    int ECTnqgWHyHCHCLBZlffd;
    int DqxLTLaJwwgQqmrtCDuu;
    int GnxRkpzrPZimKtYYHSuG;
    int GsZlHFuhbvjLtRMDjXnW;
    int fvTCtkwXgyScJYogJVFU;

    mkldnn::memory::dims HtQBsWTCGEkpylRklilw;
    mkldnn::memory::dims IwKnaBoXVubIRYcxEJLH;
    mkldnn::memory::dims IbSWJNMuIiKbocfQKqXb;
    mkldnn::memory::dims IAlDgIFcchbwRGBSfVfA;

    /*mkldnn primitives to handle the reorder for optimization purpose*/
    std::shared_ptr<mkldnn::pooling_forward::desc> pool_d;
    std::shared_ptr<mkldnn::pooling_forward::primitive_desc> pool_pd;
    std::shared_ptr<mkldnn::pooling_forward::primitive> pool;
};

// FullyConnectedLayer
class MWFCLayerImpl : public MWCNNLayerImpl {

  public:
    MWFCLayerImpl(MWCNNLayer*, MWTargetNetworkImpl*, int, int, const char*, const char*);
    ~MWFCLayerImpl();

    void predict();
    void propagateSize();
    void cleanup();

  private:
    void loadWeights(const char*, int);
    void loadBias(const char*, int);
    void prepareWeights();

    float* vIWQzNvYZSuxmOTVDFhU;
    float* PmFfARVzoHVAYkfpuvqK;

    std::shared_ptr<mkldnn::memory::desc> weights_md;
    std::shared_ptr<mkldnn::memory::desc> bias_md;

    std::shared_ptr<mkldnn::memory> weights;
    std::shared_ptr<mkldnn::memory> bias;

    std::shared_ptr<mkldnn::memory> fc_src_memory;
    std::shared_ptr<mkldnn::memory> fc_weights_memory;

    std::unique_ptr<mkldnn::reorder> fc_reorder_src;
    std::unique_ptr<mkldnn::reorder> fc_reorder_weights;

    std::unique_ptr<mkldnn::inner_product_forward::desc> ip_d;
    std::unique_ptr<mkldnn::inner_product_forward::primitive_desc> ip_pd;
    std::unique_ptr<mkldnn::inner_product_forward::primitive> ip;
};

// SoftmaxLayer
class MWSoftmaxLayerImpl : public MWCNNLayerImpl {
  public:
    MWSoftmaxLayerImpl(MWCNNLayer*, MWTargetNetworkImpl*);
    ~MWSoftmaxLayerImpl();

    void predict();
    void propagateSize();

    mkldnn::memory::desc getlayerMemoryPrimDescriptor(std::shared_ptr<mkldnn::memory>,
                                                      int,
                                                      int,
                                                      int,
                                                      int,
                                                      bool);

  private:
    std::unique_ptr<mkldnn::softmax_forward::desc> softmax_d;
    std::unique_ptr<mkldnn::softmax_forward::primitive_desc> softmax_pd;
    std::unique_ptr<mkldnn::softmax_forward::primitive> softmax;
};
// AvgPooling2DLayer
class MWAvgPoolingLayerImpl : public MWCNNLayerImpl {
  public:
    MWAvgPoolingLayerImpl(MWCNNLayer*,
                          MWTargetNetworkImpl*,
                          int,
                          int,
                          int,
                          int,
                          int,
                          int,
                          int,
                          int);
    ~MWAvgPoolingLayerImpl();

    void predict();
    void propagateSize();

  private:
    int HJHXkKmgFxxIOsIvRRnF;
    int JgLfgHrHMEMmMYTettJF;
    int JwxFdqOKggeawILBfGgg;
    int KCudOrFMfgCzUPMcdePX;
    int ECTnqgWHyHCHCLBZlffd;
    int DqxLTLaJwwgQqmrtCDuu;
    int GnxRkpzrPZimKtYYHSuG;
    int GsZlHFuhbvjLtRMDjXnW;

    /*mkldnn primitives to handle the reorder for optimization purpose*/
    std::unique_ptr<mkldnn::pooling_forward::desc> pool_d;
    std::unique_ptr<mkldnn::pooling_forward::primitive_desc> pool_pd;
    std::unique_ptr<mkldnn::pooling_forward::primitive> pool;
};

class MWOutputLayerImpl : public MWCNNLayerImpl {
  public:
    MWOutputLayerImpl(MWCNNLayer*, MWTargetNetworkImpl*);
    ~MWOutputLayerImpl();
    void propagateSize();
    void predict();
    void allocateOutputData(int);
    void deallocateOutputData(int /*i*/) {
    }

  private:
    bool doReorderFlag;
};

#endif
