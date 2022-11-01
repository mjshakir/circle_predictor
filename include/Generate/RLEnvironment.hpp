#pragma once
//--------------------------------------------------------------
/* From: https://stackoverflow.com/questions/14803112/short-way-to-stdbind-member-function-to-object-instance-without-binding-param 
    and https://stackoverflow.com/questions/70355767/binding-a-class-method-to-a-method-of-another-class */
//--------------------------------------------------------------
// LibTorch library
//--------------------------------------------------------------
#include <torch/torch.h>
//--------------------------------------------------------------
// template<typename COST_OUTPUT, typename C, typename... Args>
template<typename Dataset, typename COST_OUTPUT, typename... Args>

class RLEnvironment{
    //--------------------------------------------------------------
    public:
        //--------------------------------------------------------------
        RLEnvironment(void) = delete;
        //--------------------------
        RLEnvironment(Dataset&& data_loader) : m_data_loader(std::move{data_loader}){
            //----------------------------
        }// end RLEnvironment(Dataset&& data_loader)
        //--------------------------
        // template<typename Functions>
        // void set_reward_function(Functions& function){
        //     //--------------------------
        //     m_CostFunction = object_bind(&function, this);
        //     //--------------------------
        // }// end void set_reward_function(Functions&& function);
        //--------------------------
        template<typename FUNCTION>
        void set_reward_function(COST_OUTPUT (FUNCTION::*fun) (), FUNCTION *t){
            //--------------------------
            m_CostFunction = [t, fun](Args... args){ return (t->*fun) (args...); };
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
        // std::function<COST_OUTPUT(Args...)> objectBind(COST_OUTPUT (C::* func)(Args...), C& instance) {
        //     return [=](Args... args){ return (instance.*func)(args...); };
        // }// end std::function<COST_OUTPUT(Args...)> objectBind(COST_OUTPUT (C::* func)(Args...), C& instance)
        //--------------------------
        std::tuple<torch::Tensor, COST_OUTPUT, bool> internal_step(Args... args){
            //--------------------------
            auto input = m_data_loader->data;
            //--------------------------
            auto _reward = m_CostFunction(args...);
            //--------------------------
            if(m_data_loader == m_data_loader->end()){
                //--------------------------
                return {_input, _reward, true};
                //--------------------------
            }// if(m_iter.end())
            //--------------------------
            ++m_data_loader;
            //--------------------------
            return {_input, _reward, false};
            //--------------------------
        }// end void internal_step(const ACTION& actions)
        //--------------------------------------------------------------
    private:
        //--------------------------------------------------------------
        Dataset m_data_loader;
        //--------------------------
        std::function<COST_OUTPUT(Args...)> m_CostFunction = nullptr; 
    //--------------------------------------------------------------
};// end class RLEnvironment
//--------------------------------------------------------------