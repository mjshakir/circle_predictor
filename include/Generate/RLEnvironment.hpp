#pragma once
//--------------------------------------------------------------
/* From: https://stackoverflow.com/questions/14803112/short-way-to-stdbind-member-function-to-object-instance-without-binding-param 
    and https://stackoverflow.com/questions/70355767/binding-a-class-method-to-a-method-of-another-class 
    and https://stackoverflow.com/questions/28746744/passing-capturing-lambda-as-function-pointer */
//--------------------------------------------------------------
// LibTorch library
//--------------------------------------------------------------
#include <torch/torch.h>
//--------------------------------------------------------------
// template<typename COST_OUTPUT, typename C, typename... Args>
template<typename T, typename COST_OUTPUT, typename... Args>

class RLEnvironment{
    //--------------------------------------------------------------
    public:
        //--------------------------------------------------------------
        RLEnvironment(void) = delete;
        //--------------------------
        using CostFunction = std::function<COST_OUTPUT(Args...)>;
        RLEnvironment(std::vector<T>& data, CostFunction costFunction) :    m_data_iter(data.begin()), 
                                                                            m_data_iter_end(data.end()), 
                                                                            m_CostFunction(std::move(costFunction)){
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
        // template<typename FUNCTION>
        // void set_reward_function(COST_OUTPUT (FUNCTION::*fun) (Args...), FUNCTION *t){
        //     //--------------------------
        //     m_CostFunction = [t, fun](Args... args){ return (t->*fun) (args...); };
        //     //--------------------------
        // }// end void set_reward_function(Functions&& function);
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
            T _input = *m_data_iter;
            //--------------------------
            auto _reward = m_CostFunction(args...);
            //--------------------------
            if(m_data_iter == m_data_iter_end){
                //--------------------------
                return {torch::tensor(_input), _reward, true};
                //--------------------------
            }// if(m_iter.end())
            //--------------------------
            ++m_data_iter;
            //--------------------------
            return {torch::tensor(_input), _reward, false};
            //--------------------------
        }// end void internal_step(const ACTION& actions)
        //--------------------------------------------------------------
    private:
        //--------------------------------------------------------------
        typename std::vector<T>::iterator m_data_iter, m_data_iter_end;
        //--------------------------
        // std::function<COST_OUTPUT(Args...)> m_CostFunction = nullptr;
        //-------------------------- 
        // using CostFunction = std::function<COST_OUTPUT(Args...)>;
        // CostFunction m_CostFunction(Args...);
        CostFunction m_CostFunction;
    //--------------------------------------------------------------
};// end class RLEnvironment
//--------------------------------------------------------------