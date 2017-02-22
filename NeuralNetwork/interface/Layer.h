/* 
==================================================
 Layer
 
 One layer of an Artificial Neural Network
 
Copyright (C) 20 February 2017 Souvik Das
ALL RIGHTS RESERVED
=================================================
*/

#pragma once

#include <vector>

#include "ActivationFunctionBase.h"
#include "CostFunctionBase.h"
#include "DescentMethodBase.h"

class Layer;

class Layer
{
  private:
  
    int hidinout_; // 0 = hidden layer, 1 = input layer, 2 = output layer
    
    std::vector<float> v_z_;                    // logits
    std::vector<float> v_y_;                    // activations
    std::vector<std::vector<float> > *v_v_w_;   // fan-out weights[thisLayer][nextLayer]; This is a pointer because multiple convolution layers may share the same weights
    std::vector<float> v_dEdz_;                 // derivative of error w.r.t. logit
    std::vector<std::vector<float> > v_v_dEdw_; //derivative of error w.r.t. fan out weights 
    Layer *nextLayer_;
    std::vector<Layer*> v_convolutionLayers_;
    
    ActivationFunctionBase *activationFunction_;
    CostFunctionBase *costFunction_;
    DescentMethodBase *descentMethod_;
    
  public:
  
    Layer(unsigned int neurons, int hidinout);
    void setNextLayer(Layer *nextLayer);
    void setConvolutionLayers(std::vector<Layer*> v_convolution_Layers);
    
    void setInputs(std::vector<float> *inputs);
    
    void computeActivations();                       // compute y from z. needs a given activationFunction_
    void computeForwardPropagation();                // compute v_z of next layer from v_y and v_v_w of this layer
    
    void computedEdz();                              // if output layer, depends on costFunction and activationFunction. else, needs dE/dz of next layer, weights, activationFunction of this layer
    void computedEdw();                              // needs dEdz of next layer, activation of this layer
    void computeNewWeights();                        // needs dE/dw of this layer's and convolved layer's fan-out weights. needs descent method
};
