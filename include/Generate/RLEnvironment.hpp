//--------------------------------------------------------------
// LibTorch library
//--------------------------------------------------------------
#include <torch/torch.h>
//--------------------------------------------------------------
// User Defined library
//--------------------------------------------------------------
#include "Generate/Generate.hpp"
//--------------------------------------------------------------
template<typename INPUT, typename COST_OUTPUT, typename ...Args>
class RLEnvironment{
    //--------------------------------------------------------------
    public:
        //--------------------------
        RLEnvironment(void) = delete;
        //--------------------------
        RLEnvironment(const INPUT& input) : m_input(input){
            //--------------------------
        }// end RLEnvironment(const INPUT& input) : m_input(input)
        //--------------------------
        RLEnvironment(INPUT&& input) : m_input(std::move(input)){
            //--------------------------
        }// end RLEnvironment(const INPUT& input) : m_input(input)
        //--------------------------
        template<typename Functions>
        void set_cost_function(Functions& function){
            //--------------------------
            // m_CostFunction = function(std::forward(Args...));
            m_CostFunction = object_bind(&function, this);
            //--------------------------
        }// end void set_cost_function(Functions&& function);
        //--------------------------------------------------------------
    protected:
        //--------------------------
        /*template<typename R, typename C, typename... Args>
        std::function<R(Args...)> objectBind(R (C::* func)(Args...), C& instance) {
            return [=](Args... args){ return (instance.*func)(args...); };
        }// end std::function<R(Args...)> objectBind(R (C::* func)(Args...), C& instance) 
        From: https://stackoverflow.com/questions/14803112/short-way-to-stdbind-member-function-to-object-instance-without-binding-param */
        //--------------------------
        template<typename C>
        std::function<COST_OUTPUT(Args...)> object_bind(R (C::* func)(Args...), C& instance){
            return [=](Args... args){ return (instance.*func)(args...); };
        }// end std::function<COST_OUTPUT(Args...)> objectBind(R (C::* func)(Args...), C& instance)
        //--------------------------
    private:
        //--------------------------
        INPUT m_input;
        std::function<COST_OUTPUT(Args...)> m_CostFunction = nullptr;
        //--------------------------
    //--------------------------------------------------------------
};// end class RLEnvironment
//--------------------------------------------------------------