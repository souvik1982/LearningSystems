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

#include "ActivationFunction.h"
#include "CostFunction.h"
#include "DescentMethod.h"

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
    
    ActivationFunction *activationFunction_=0;
    CostFunction *costFunction_=0;
    DescentMethod *descentMethod_=0;
    
  public:
  
    // if nextLayer = 0, this is an output layer
    Layer(unsigned int neurons, int hidinout, Layer *nextLayer=0, Layer *convLayer=0);
    
    setInputs(std::vector<float> *inputs);
    
    computeActivations();                       // compute y from z. needs a given activationFunction_
    computeForwardPropagation();                // compute v_z of next layer from v_y and v_v_w of this layer
    
    computedEdz();                              // if output layer, depends on costFunction and activationFunction. else, needs dE/dz of next layer, weights, activationFunction of this layer
    computedEdw();                              // needs dEdz of next layer, activation of this layer
    computeNewWeights();                        // needs dE/dw of this layer's and convolved layer's fan-out weights. needs descent method
};
