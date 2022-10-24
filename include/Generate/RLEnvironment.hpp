#pragma once

//--------------------------------------------------------------
// LibTorch library
//--------------------------------------------------------------
#include <torch/torch.h>
//--------------------------------------------------------------
// User Defined library
//--------------------------------------------------------------
#include "Generate/Generate.hpp"
//--------------------------------------------------------------
// template<typename INPUT, typename COST_OUTPUT, typename ...Args>
class RLEnvironment : protected Generate{
    //--------------------------------------------------------------
    public:
        //--------------------------------------------------------------
        RLEnvironment(void) = delete;
        //--------------------------
        // RLEnvironment(const INPUT& input) : m_input(input){
        //     //--------------------------
        // }// end RLEnvironment(const INPUT& input) : m_input(input)
        //--------------------------
        // RLEnvironment(INPUT&& input) : m_input(std::move(input)){
        //     //--------------------------
        // }// end RLEnvironment(const INPUT& input) : m_input(input)
        //--------------------------
        RLEnvironment(  const double& radius = 1, 
                        const size_t& generated_points = 60000, 
                        const std::tuple<double, double>& center = {0,0}, 
                        const size_t& batch_size = 20U);
        //--------------------------
        // template<typename Functions>
        // void set_cost_function(Functions& function){
        //     //--------------------------
        //     // m_CostFunction = function(std::forward(Args...));
        //     m_CostFunction = object_bind(&function, this);
        //     //--------------------------
        // }// end void set_cost_function(Functions&& function);
        //--------------------------------------------------------------
    protected:
        //--------------------------------------------------------------
        /*template<typename R, typename C, typename... Args>
        std::function<R(Args...)> objectBind(R (C::* func)(Args...), C& instance) {
            return [=](Args... args){ return (instance.*func)(args...); };
        }// end std::function<R(Args...)> objectBind(R (C::* func)(Args...), C& instance) 
        From: https://stackoverflow.com/questions/14803112/short-way-to-stdbind-member-function-to-object-instance-without-binding-param */
        //--------------------------------------------------------------
        // template<typename C>
        // std::function<COST_OUTPUT(Args...)> object_bind(COST_OUTPUT (C::* func)(Args...), C& instance){
        //     return [=](Args... args){ return (instance.*func)(args...); };
        // }// end std::function<COST_OUTPUT(Args...)> objectBind(R (C::* func)(Args...), C& instance)
        //--------------------------
        double internal_reward_function(const torch::Tensor& real_value, const torch::Tensor& predicted_value, const long double& tolerance = 5E-2) const;
        //--------------------------
        /**
         *  @brief This is a way to step through the problem
         *
         *  @tparam predicted_value: A torch Tensor represeting the actions .
         *  @return A tuple of New states as torch tensor and the reward as a double
         */
        std::tuple<torch::Tensor, double, bool> internal_step_function(const torch::Tensor& actions, const long double& tolerance = 5E-2);
        //--------------------------------------------------------------
    private:
        //--------------------------------------------------------------
        // INPUT m_input;
        std::vector<std::tuple<torch::Tensor, torch::Tensor>>::iterator m_iter, m_iter_end;
        std::vector<double> m_output_data;
        // std::function<COST_OUTPUT(Args...)> m_CostFunction = nullptr;
        //--------------------------
    //--------------------------------------------------------------
};// end class RLEnvironment
//--------------------------------------------------------------