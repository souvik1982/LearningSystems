/* 
==================================================
 ActivationFunctionBase
 
 Base class for Activation Functions
 
Copyright (C) 20 February 2017 Souvik Das
ALL RIGHTS RESERVED
=================================================
*/

#include "math.h"

class ActivationFunctionBase
{
  public:
    // For now, these have default implementation as the logistic function
    // Make these virtual eventually
    float activate(float z)
    {
      float y = 1./(1.+exp(-z));
      return y;
    }
    float derivative(float y)
    {
      float der=y*(1.-y);
      return der;
    }
};
