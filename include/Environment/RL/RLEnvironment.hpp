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
// User definition
//--------------------------------------------------------------
#define OUT
//--------------------------------------------------------------
namespace Environment {
    //--------------------------------------------------------------
    template<typename T, typename COST_OUTPUT, typename... Args>
    //--------------------------------------------------------------
    class RLEnvironment {
        //--------------------------------------------------------------
        public:
            //--------------------------------------------------------------
            RLEnvironment(void) = delete;
            //--------------------------------------------------------------
            /**
             * @brief Construct to create a training environment for reinforcement learning 
             * 
             * @param data          [in] : Data Vector of the template type T
             * @param costFunction  [in] : Cost function that return the reward. needs to define the function return and parameters
             * @param egreedy       [in] : The starting egreedy                                           @default: 0.9
             * @param egreedy_final [in] : The egreedy number where it will change                        @default: 0.02
             * @param egreedy_decay [in] : The egreedy exponential (e^x) decay factor                     @default: 500.
             * @param batch         [in] : The batch number (needs to be less then half of the data size) @default: 1ul   
             */
            RLEnvironment(  std::vector<T>&& data, 
                            std::function<COST_OUTPUT(const Args&...)> costFunction,
                            const double& egreedy = 0.9,
                            const double& egreedy_final = 0.02,
                            const double& egreedy_decay = 500.) :   m_data(std::move(data)),
                                                                    m_data_iter (m_data.begin()), 
                                                                    m_CostFunction(std::move(costFunction)),
                                                                    m_egreedy(egreedy),
                                                                    m_egreedy_final(egreedy_final),
                                                                    m_egreedy_decay(egreedy_decay){
                //----------------------------
                if(egreedy_decay == 0.){
                    //----------------------------
                    throw std::runtime_error("Egreedy Decay Cannot Be Zero. egreedy_decay Value: [" + std::to_string(egreedy_decay) + "]");
                    //----------------------------
                }// end if(egreedy_decay == 0.)
                //----------------------------
            }// end RLEnvironment(Dataset&& data_loader)
            //--------------------------------------------------------------m_batch
            /**
             * @brief 
             * 
             * @param args 
             * @return std::tuple<torch::Tensor, COST_OUTPUT, double, bool> 
             */
            virtual std::tuple<torch::Tensor, COST_OUTPUT, double, bool> step(const Args&... args){
                //----------------------------
                return internal_step(args...);
                //----------------------------
            }// std::tuple<torch::Tensor, COST_OUTPUT, double, bool> step(const Args&... args)
            //--------------------------------------------------------------
            /**
             * @brief 
             * 
             * @param epsilon 
             * @param done 
             * @param args 
             * @return std::tuple<torch::Tensor, COST_OUTPUT> 
             */
            virtual std::tuple<torch::Tensor, COST_OUTPUT> step(OUT double& epsilon, OUT bool& done, const Args&... args){
                //----------------------------
                return internal_step(epsilon, done, args...);
                //----------------------------
            }// std::tuple<torch::Tensor, COST_OUTPUT> step(OUT double& epsilon, OUT bool& done, const Args&... args))
            //--------------------------------------------------------------
            /**
             * @brief Get the first object
             * 
             * @return std::tuple<torch::Tensor, double> 
             */
            virtual std::tuple<torch::Tensor, double> get_first(void){
                //----------------------------
                return get_first_internal();
                //----------------------------
            }// end torch::Tensor get_first(void)
            //--------------------------------------------------------------
            /**
             * @brief Get the first object
             * 
             * @param epsilon 
             * @return torch::Tensor 
             */
            virtual torch::Tensor get_first(OUT double& epsilon){
                //----------------------------
                return get_first_internal(epsilon);
                //----------------------------
            }// end torch::Tensor get_first(void)
            //--------------------------------------------------------------
            /**
             * @brief 
             * 
             */
            virtual void reset(void){
                //--------------------------
                reset_iterator();
                //--------------------------
            }// end void reset(void)
            //--------------------------------------------------------------
        protected:
            //--------------------------------------------------------------
            std::tuple<torch::Tensor, COST_OUTPUT, double, bool> internal_step(const Args&... args){
                //--------------------------
                if (m_data_iter == m_data.end()){
                    //--------------------------
                    throw std::out_of_range("End Of The Data Iterator");
                    //--------------------------
                }// end if (m_data_iter == m_data.end())
                //--------------------------
                if (m_data_iter == m_data.begin()){
                    //--------------------------
                    auto input = *m_data_iter;
                    auto epsilon = calculate_epsilon();
                    //--------------------------
                    ++m_data_iter;
                    //--------------------------
                    return {input, torch::tensor(NULL), epsilon, false};
                    //--------------------------
                }// end if (m_data_iter == m_data.begin())
                //--------------------------
                if(m_data_iter == m_data.end()-1){
                    //--------------------------
                    return {*m_data_iter, m_CostFunction(args...), calculate_epsilon(), true};
                    //--------------------------
                }// if(m_data_iter == m_data.end())
                //--------------------------
                if(m_data_iter != m_data.end()-1){
                    //--------------------------
                    ++m_data_iter;
                    //--------------------------
                }// if(m_data_iter == m_data.end())
                //--------------------------
                return {*m_data_iter, m_CostFunction(args...), calculate_epsilon(), false};
                //--------------------------
            }// end std::tuple<torch::Tensor, COST_OUTPUT, double, bool> internal_step(Args... args))
            //--------------------------------------------------------------
            std::tuple<torch::Tensor, COST_OUTPUT> internal_step(OUT double& epsilon, OUT bool& done, const Args&... args){
                //--------------------------
                if (m_data_iter == m_data.end()){
                    //--------------------------
                    throw std::out_of_range("End Of The Data Iterator");
                    //--------------------------
                }// end if (m_data_iter == m_data.end())
                //--------------------------
                if (m_data_iter == m_data.begin()){
                    //--------------------------
                    auto input = *m_data_iter;
                    epsilon = calculate_epsilon();
                    done = false;
                    //--------------------------
                    ++m_data_iter;
                    //--------------------------
                    return {input, torch::tensor(0)};
                    //--------------------------
                }// end if (m_data_iter == m_data.begin())
                //--------------------------
                if(m_data_iter == m_data.end()-1){
                    //--------------------------
                    epsilon = calculate_epsilon();
                    done = true;
                    //--------------------------
                    return {*m_data_iter, m_CostFunction(args...)};
                    //--------------------------
                }// if(m_data_iter == m_data.end())
                //--------------------------
                if(m_data_iter != m_data.end()-1){
                    //--------------------------
                    ++m_data_iter;
                    //--------------------------
                }// if(m_data_iter == m_data.end())
                //--------------------------
                epsilon = calculate_epsilon();
                done = false;
                //--------------------------
                return {*m_data_iter, m_CostFunction(args...)};
                //--------------------------
            }// end std::tuple<torch::Tensor, COST_OUTPUT> internal_step(double& epsilon, bool& done, Args... args)
            //--------------------------------------------------------------
            std::tuple<torch::Tensor, double> get_first_internal(void){
                //--------------------------
                if (m_data_iter == m_data.end()){
                    //--------------------------
                    throw std::out_of_range("End Of The Data Iterator");
                    //--------------------------
                }// end if (m_data_iter == m_data.end() or std::next(m_data_iter, batch) == m_data.end())
                //--------------------------
                if (m_data_iter == m_data.begin()){
                    //--------------------------
                    auto input = *m_data_iter;
                    //--------------------------
                    auto epsilon = calculate_epsilon();
                    //--------------------------
                    ++m_data_iter;
                    //--------------------------
                    return {input, epsilon};
                    //--------------------------
                }// end if (m_data_iter == m_data.begin())
                //--------------------------
                return {torch::tensor(0), 0};
                //--------------------------
            }// end torch::Tensor get_first_internal(void)
            //--------------------------------------------------------------
            torch::Tensor get_first_internal(OUT double& epsilon){
                //--------------------------
                if (m_data_iter == m_data.end()){
                    //--------------------------
                    throw std::out_of_range("End Of The Data Iterator");
                    //--------------------------
                }// end if (m_data_iter == m_data.end() or std::next(m_data_iter, batch) == m_data.end())
                //--------------------------
                if (m_data_iter == m_data.begin()){
                    //--------------------------
                    auto input = *m_data_iter;
                    //--------------------------
                    epsilon = calculate_epsilon();
                    //--------------------------
                    ++m_data_iter;
                    //--------------------------
                    return input;
                    //--------------------------
                }// end if (m_data_iter == m_data.begin())
                //--------------------------
                epsilon = 0.;
                //--------------------------
                return torch::tensor(0);
                //--------------------------
            }// end torch::Tensor get_first_internal(void)
            //--------------------------------------------------------------
            virtual void reset_iterator(void){
                //--------------------------
                m_data_iter = m_data.begin();
                //--------------------------
            }// end void rest_iterator(void)
            //--------------------------------------------------------------
            constexpr double calculate_epsilon(void){
                //--------------------------
                return m_egreedy_final + (m_egreedy - m_egreedy_final) * std::exp(-1. * std::distance(m_data.begin(), m_data_iter) / m_egreedy_decay);
                //--------------------------
            }// end double calculate_epsilon()
            //--------------------------------------------------------------
            std::vector<T>& get_data(void){
                //--------------------------
                return m_data;
                //--------------------------
            }// end std::vector<T>& get_data(void)
            //--------------------------------------------------------------
            std::function<COST_OUTPUT(const Args&...)>& get_cost_function(void){
                //--------------------------
                return m_CostFunction;
                //--------------------------
            }// end std::function<COST_OUTPUT(const Args&...)>& get_cost_function(void)
            //--------------------------------------------------------------
            typename std::vector<T>::iterator& get_iterator(void){
                //--------------------------
                return m_data_iter;
                //--------------------------
            }// end typename std::vector<T>::iterator& get_iterator(void)
            //--------------------------------------------------------------
        private:
            //--------------------------------------------------------------
            std::vector<T> m_data;
            typename std::vector<T>::iterator m_data_iter;
            //--------------------------
            std::function<COST_OUTPUT(const Args&...)> m_CostFunction;
            //--------------------------
            double m_egreedy, m_egreedy_final, m_egreedy_decay;
        //--------------------------------------------------------------
    };// end class RLEnvironment
    //--------------------------------------------------------------
}// end namespace Environment
//--------------------------------------------------------------