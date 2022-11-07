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
template<typename T, typename COST_OUTPUT, typename... Args>
//--------------------------------------------------------------
class RLEnvironment{
    //--------------------------------------------------------------
    public:
        //--------------------------------------------------------------
        RLEnvironment(void) = delete;
        //--------------------------
        RLEnvironment(  std::vector<T>&& data, 
                        std::function<COST_OUTPUT(Args&...)> costFunction,
                        const double& egreedy = 0.9,
                        const double& egreedy_final = 0.02,
                        const double& egreedy_decay = 500.) :   m_data(std::move(data)),
                                                                m_data_iter (m_data.begin()), 
                                                                m_CostFunction(std::move(costFunction)),
                                                                m_egreedy(egreedy),
                                                                m_egreedy_final(egreedy_final),
                                                                m_egreedy_decay(egreedy_decay){
            //----------------------------
        }// end RLEnvironment(Dataset&& data_loader)
        //--------------------------
        std::tuple<torch::Tensor, COST_OUTPUT, double, bool> step(Args... args){
            //----------------------------
            return internal_step(args...);
            //----------------------------
        }// std::tuple<torch::Tensor, COST_OUTPUT, double, bool> step(Args... args)
        //--------------------------------------------------------------
    protected:
        //--------------------------------------------------------------
        std::tuple<torch::Tensor, COST_OUTPUT, double, bool> internal_step(Args... args){
            //--------------------------
            if (m_data_iter == m_data.begin()){
                //--------------------------
                auto input = *m_data_iter;
                auto epsilon = calculate_epsilon();
                //--------------------------
                ++m_data_iter;
                //--------------------------
                return {*m_data_iter, NULL, epsilon, false};
                //--------------------------
            }// end auto _reward = m_CostFunction(args...)
            //--------------------------
            auto _reward = m_CostFunction(args...);
            //--------------------------
             if(m_data_iter == m_data.end()-1){
                //--------------------------
                return {*m_data_iter, _reward, calculate_epsilon(), true};
                //--------------------------
            }// if(m_data_iter == m_data.end())
            //--------------------------
            if(m_data_iter != m_data.end()-1){
                //--------------------------
                ++m_data_iter;
                //--------------------------
            }// if(m_data_iter == m_data.end())
            //--------------------------
            return {*m_data_iter, _reward, calculate_epsilon(), false};
            //--------------------------
        }// end std::tuple<torch::Tensor, COST_OUTPUT, double, bool> internal_step(Args... args))
        //--------------------------
        constexpr double calculate_epsilon(void){
            //--------------------------
            return m_egreedy_final + (m_egreedy - m_egreedy_final) * exp(-1. * std::distance(m_data.begin(), m_data_iter) / m_egreedy_decay );
            //--------------------------
        }// end double calculate_epsilon()
        //--------------------------------------------------------------
    private:
        //--------------------------------------------------------------
        std::vector<T> m_data;
        typename std::vector<T>::iterator m_data_iter;
        //--------------------------
        std::function<COST_OUTPUT(Args&...)> m_CostFunction;
        //--------------------------
        double m_egreedy, m_egreedy_final, m_egreedy_decay;
    //--------------------------------------------------------------
};// end class RLEnvironment
//--------------------------------------------------------------