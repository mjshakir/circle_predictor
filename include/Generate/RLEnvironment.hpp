#pragma once

//--------------------------------------------------------------
// LibTorch library
//--------------------------------------------------------------
#include <torch/torch.h>
//--------------------------------------------------------------
// template<typename INPUT, typename COST_OUTPUT, typename ...Args>
template<typename COST_OUTPUT, typename C, typename... Args>
class RLEnvironment{
    //--------------------------------------------------------------
    public:
        //--------------------------------------------------------------
        RLEnvironment(void) = delete;
        //--------------------------
        template<typename Dataset>
        RLEnvironment(Dataset&& data_loader){
            //--------------------------
            for (const auto& batch : *data_loader){
                //--------------------------
                auto input_data = batch.data, target_data = batch.target;
                //--------------------------
                _full_data.push_back({input_data, target_data});
                //--------------------------
            } //for (const auto& batch : *data_loader)
            //--------------------------
            m_iter = _full_data.begin();
            m_iter_end = _full_data.end();
            //--------------------------
        }// end RLEnvironment(Dataset&& data_loader)
        //--------------------------
        template<typename Functions>
        void set_reward_function(Functions& function){
            //--------------------------
            m_CostFunction = object_bind(&function, this);
            //--------------------------
        }// end void set_reward_function(Functions&& function);
        //--------------------------------------------------------------
    protected:
        //--------------------------------------------------------------
        /*template<typename R, typename C, typename... Args>
        std::function<R(Args...)> objectBind(R (C::* func)(Args...), C& instance) {
            return [=](Args... args){ return (instance.*func)(args...); };
        }// end std::function<R(Args...)> objectBind(R (C::* func)(Args...), C& instance) 
        From: https://stackoverflow.com/questions/14803112/short-way-to-stdbind-member-function-to-object-instance-without-binding-param 
        and https://stackoverflow.com/questions/70355767/binding-a-class-method-to-a-method-of-another-class */
        //--------------------------------------------------------------
        // template<typename C>
        // std::function<COST_OUTPUT(Args...)> object_bind(COST_OUTPUT (C::* func)(Args...), C& instance){
        //     return [=](Args... args){ return (instance.*func)(args...); };
        // }// end std::function<COST_OUTPUT(Args...)> objectBind(R (C::* func)(Args...), C& instance)
        //--------------------------
        std::function<COST_OUTPUT(Args...)> objectBind(COST_OUTPUT (C::* func)(Args...), C& instance) {
            return [=](Args... args){ return (instance.*func)(args...); };
        }// end std::function<COST_OUTPUT(Args...)> objectBind(COST_OUTPUT (C::* func)(Args...), C& instance)
        //--------------------------
        std::tuple<torch::Tensor, COST_OUTPUT, bool> internal_step(Args... args){
            //--------------------------
            auto [_input, _target] = *m_iter;
            //--------------------------
            auto _reward = m_CostFunction(args...);
            //--------------------------
            if(m_iter == m_iter_end){
                //--------------------------
                return {_input, _reward, true};
                //--------------------------
            }// if(m_iter.end())
            //--------------------------
            ++m_iter;
            //--------------------------
            return {_input, _reward, false};
            //--------------------------
        }// end void internal_step(const ACTION& actions)
        //--------------------------------------------------------------
    private:
        //--------------------------------------------------------------
        Dataset m_data_loader;
        //--------------------------
        std::vector<std::tuple<torch::Tensor, torch::Tensor>>::iterator m_iter, m_iter_end;
        //--------------------------
        std::function<COST_OUTPUT(Args...)> m_CostFunction = nullptr;
        //--------------------------
    //--------------------------------------------------------------
};// end class RLEnvironment
//--------------------------------------------------------------