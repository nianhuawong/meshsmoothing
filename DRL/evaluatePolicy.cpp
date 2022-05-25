//
// File: evaluatePolicy.cpp
//
// MATLAB Coder version            : 5.0
// C/C++ source code generated on  : 24-May-2022 22:35:28
//

// Include Files
#include "evaluatePolicy.h"
#include "DeepLearningNetwork.h"
#include "evaluatePolicy_data.h"
#include "evaluatePolicy_initialize.h"
#include "predict.h"

// Variable Definitions
static b_policy_0 policy;
static boolean_T policy_not_empty;

// Function Definitions

//
// Reinforcement Learning Toolbox
//  Generated on: 24-May-2022 22:34:41
// Arguments    : const double observation1[20]
//                float action1[2]
// Return Type  : void
//
void evaluatePolicy(const double observation1[20], float action1[2])
{
  if (!isInitialized_evaluatePolicy) {
    evaluatePolicy_initialize();
  }

  //  Local Functions
  if (!policy_not_empty) {
    DeepLearningNetwork_setup(&policy);
    policy_not_empty = true;
  }

  DeepLearningNetwork_predict(&policy, observation1, action1);
}

//
// Arguments    : void
// Return Type  : void
//
void localEvaluate_init()
{
  policy_not_empty = false;
}

//
// File trailer for evaluatePolicy.cpp
//
// [EOF]
//
