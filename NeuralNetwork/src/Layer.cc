/* 
==================================================
 Layer
 
 One layer of an Artificial Neural Network
 
Copyright (C) 20 February 2017 Souvik Das
ALL RIGHTS RESERVED
=================================================
*/

#include <cassert>
#include <iostream>

#include "../interface/Layer.h"

Layer::Layer(unsigned int neurons, int hidinout)
{
  hidinout_=hidinout;
  
  v_z_.resize(neurons);
  v_y_.resize(neurons);
  v_v_w_->resize(neurons);
  v_dEdz_.resize(neurons);
  v_v_dEdw_.resize(neurons);
  
  nextLayer_=0;
  
  activationFunction_=0;
  costFunction_=0;
  descentMethod_=0;
}

void Layer::setNextLayer(Layer *nextLayer)
{
  if (hidinout_!=2)
  {
    if (nextLayer!=0) nextLayer_=nextLayer;
    else
    {
      std::cout<<"ERROR: The nextLayer = 0"<<std::endl;
      assert(0);
    }
  }
  else
  {
    std::cout<<"ERROR: This layer is an output layer but has been assigned a nextLayer"<<std::endl;
    assert(0);
  }
}

void Layer::setConvolutionLayers(std::vector<Layer*> v_convolutionalLayers)
{
  if (hidinout_!=2)
  {
    v_convolutionLayers_=v_convolutionalLayers;
  }
  else
  {
    std::cout<<"ERROR: This layer is an output layer but has been assigned convolutional layers"<<std::endl;
    assert(0);
  }
}

void Layer::setInputs(std::vector<float> *v_inputs)
{
  if (hidinout_==1)
  {
    v_y_=*v_inputs;
  }
  else
  {
    std::cout<<"ERROR: This is not an input layer but has been assigned inputs"<<std::endl;
    assert(0);
  }
}
  
