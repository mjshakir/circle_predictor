//--------------------------------------------------------------
// LibTorch library
//--------------------------------------------------------------
#include <torch/torch.h>
//--------------------------------------------------------------
// User Defined library
//--------------------------------------------------------------
#include "Generate/Generate.hpp"
//--------------------------------------------------------------
template<typename INPUT, typename ...Args>
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
        template<typename Functions>
        void set_cost_function(Functions& function){
            //--------------------------
            m_CostFunction = std::bind(&function, this, std::placeholders::_1, std::placeholders::_2);;
            //--------------------------
        }// end void set_cost_function(Functions&& function);
        //--------------------------------------------------------------
    protected:
        //--------------------------
        
        //--------------------------
    private:
        //--------------------------
        INPUT m_input;
        std::function<double(Args...)> m_CostFunction = nullptr;
        //--------------------------
    //--------------------------------------------------------------
};// end class RLEnvironment
//--------------------------------------------------------------