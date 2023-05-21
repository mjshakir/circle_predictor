#pragma once
//--------------------------------------------------------------
// User Defined library
//--------------------------------------------------------------
#include "Environment/RL/RLEnvironment.hpp"
//--------------------------------------------------------------
namespace RL {
    //--------------------------------------------------------------
    namespace Environment {
        //--------------------------------------------------------------
        template<typename T, typename COST_OUTPUT, typename... Args>
        //--------------------------------------------------------------
        class RLEnvironmentLoader : public RLEnvironment<T, COST_OUTPUT, Args...>{
            //--------------------------------------------------------------
            public:
                //--------------------------------------------------------------
                RLEnvironmentLoader(void) = delete;
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
                RLEnvironmentLoader(std::vector<T>&& data, 
                                    std::function<COST_OUTPUT(const Args&...)> costFunction,
                                    const double& egreedy = 0.9,
                                    const double& egreedy_final = 0.02,
                                    const double& egreedy_decay = 500.,
                                    const size_t& batch = 1ul) : RLEnvironment<T, COST_OUTPUT, Args...>(std::move(data), std::move(costFunction), 
                                                                                                        egreedy, egreedy_final, egreedy_decay),
                                                                m_data(this->get_data()),
                                                                m_data_iter (this->get_iterator()), 
                                                                m_CostFunction(this->get_cost_function()),
                                                                m_batch(batch){
                    //----------------------------
                    if(batch >= m_data.size()/2){
                        //--------------------------
                        throw std::out_of_range("Batch Size: [" + std::to_string(batch) + "] Must Be Less Then The data Size: [" + std::to_string(m_data.size()/2) + "]");
                        //--------------------------
                    }// end if(batch >= m_data.size()/2)
                    //----------------------------
                }// end RLEnvironmentLoader(Dataset&& data_loader)
                //--------------------------------------------------------------
                /**
                 * @brief 
                 * 
                 * @param args 
                 * @return std::tuple<torch::Tensor, COST_OUTPUT, double, bool> 
                 */
                virtual std::tuple<torch::Tensor, COST_OUTPUT, double, bool> step(const Args&... args) override {
                    //----------------------------
                    return internal_step(m_batch, args...);
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
                virtual std::tuple<torch::Tensor, COST_OUTPUT> step(OUT double& epsilon, OUT bool& done, const Args&... args) override {
                    //----------------------------
                    return internal_step(epsilon, done, m_batch, args...);
                    //----------------------------
                }// std::tuple<torch::Tensor, COST_OUTPUT> step(OUT double& epsilon, OUT bool& done, const Args&... args))
                //--------------------------------------------------------------
                /**
                 * @brief 
                 * 
                 * @param epsilon 
                 * @param done 
                 * @param batch 
                 * @param args 
                 * @return std::tuple<torch::Tensor, COST_OUTPUT> 
                 */
                std::tuple<torch::Tensor, COST_OUTPUT> step(OUT double& epsilon, OUT bool& done, const size_t& batch, const Args&... args){
                    //----------------------------
                    return internal_step(epsilon, done, batch, args...);
                    //----------------------------
                }// std::tuple<torch::Tensor, COST_OUTPUT> step(OUT double& epsilon, OUT bool& done, const size_t& batch, const Args&... args)
                //--------------------------------------------------------------
                /**
                 * @brief Get the first object
                 * 
                 * @return std::tuple<torch::Tensor, double> 
                 */
                virtual std::tuple<torch::Tensor, double> get_first(void) override {
                    //----------------------------
                    return get_first_internal(m_batch);
                    //----------------------------
                }// end torch::Tensor get_first(void)
                //--------------------------------------------------------------
                /**
                 * @brief Get the first object
                 * 
                 * @param epsilon 
                 * @return torch::Tensor 
                 */
                virtual torch::Tensor get_first(OUT double& epsilon) override {
                    //----------------------------
                    return get_first_internal(epsilon, m_batch);
                    //----------------------------
                }// end torch::Tensor get_first(void)
                //--------------------------------------------------------------
                /**
                 * @brief Get the first object
                 * 
                 * @param epsilon 
                 * @param batch 
                 * @return torch::Tensor 
                 */
                torch::Tensor get_first(OUT double& epsilon, const size_t& batch){
                    //----------------------------
                    return get_first_internal(epsilon, batch);
                    //----------------------------
                }// end torch::Tensor get_first(void)
                //--------------------------------------------------------------
            protected:
                //--------------------------------------------------------------
                std::tuple<torch::Tensor, COST_OUTPUT, double, bool> internal_step(const size_t& batch, const Args&... args){
                    //--------------------------------------------------------------
                    if (m_data_iter == m_data.end() or std::next(m_data_iter, batch) == m_data.end()){
                        //--------------------------
                        throw std::out_of_range("End Of The Data Iterator");
                        //--------------------------
                    }// end if (m_data_iter == m_data.end() or std::next(m_data_iter, batch) == m_data.end())
                    //--------------------------
                    torch::Tensor _data;
                    //--------------------------------------------------------------
                    if (m_data_iter == m_data.begin() and std::next(m_data_iter, batch) != m_data.end()-1){
                        //--------------------------
                        auto epsilon = this->calculate_epsilon();
                        //--------------------------
                        _data = *m_data_iter;
                        //--------------------------
                        for(size_t i = 1; i < batch; ++i){
                            //--------------------------
                            _data = torch::cat({_data, *++m_data_iter});
                            //--------------------------
                        }// end for(size_t i = 0; i < batch; ++i)
                        //--------------------------
                        return {_data, torch::tensor(NULL), epsilon, false};
                        //--------------------------
                    }// end if (m_data_iter == m_data.begin())
                    //--------------------------------------------------------------
                    if(std::next(m_data_iter, batch) == m_data.end()-1){
                        //--------------------------
                        _data = *m_data_iter;
                        //--------------------------
                        for (size_t i = 1; i < batch; ++i){
                            //--------------------------
                            _data = torch::cat({_data, *++m_data_iter});
                            //--------------------------
                        }// end for (size_t i = 1; i < batch; i++)
                        //--------------------------
                        return {_data, m_CostFunction(args...), this->calculate_epsilon(), true};
                        //--------------------------
                    }// if(m_data_iter == m_data.end())
                    //--------------------------------------------------------------
                    if(m_data_iter != m_data.end()-1 and std::next(m_data_iter, batch) != m_data.end()-1){
                        //--------------------------
                        _data = *++m_data_iter;
                        //--------------------------
                        for(size_t i = 1; i < batch; ++i){
                            //--------------------------
                            _data = torch::cat({_data, *++m_data_iter});
                            //--------------------------
                        }// end for(size_t i = 0; i < batch; ++i)
                        //--------------------------
                    }// end if(m_data_iter != m_data.end()-1 && std::next(m_data_iter, batch) != m_data.end()-1)
                    //--------------------------
                    return {_data, m_CostFunction(args...), this->calculate_epsilon(), false};
                    //--------------------------
                }// end std::tuple<torch::Tensor, COST_OUTPUT, double, bool> internal_step(const size_t& batch, Args... args)
                //--------------------------------------------------------------
                std::tuple<torch::Tensor, COST_OUTPUT> internal_step(OUT double& epsilon, OUT bool& done, const size_t& batch, const Args&... args){
                    //--------------------------------------------------------------
                    if (m_data_iter == m_data.end() or std::next(m_data_iter, batch) == m_data.end()){
                        //--------------------------
                        throw std::out_of_range("End Of The Data Iterator");
                        //--------------------------
                    }// end if (m_data_iter == m_data.end() or std::next(m_data_iter, batch) == m_data.end())
                    //--------------------------
                    torch::Tensor _data;
                    //--------------------------------------------------------------
                    if (m_data_iter == m_data.begin() and std::next(m_data_iter, batch) != m_data.end()-1){
                        //--------------------------
                        epsilon = this->calculate_epsilon();
                        done = false;
                        //--------------------------
                        _data = *m_data_iter;
                        //--------------------------
                        for(size_t i = 1; i < batch; ++i){
                            //--------------------------
                            _data = torch::cat({_data, *++m_data_iter});
                            //--------------------------
                        }// end for(size_t i = 0; i < batch; ++i)
                        //--------------------------
                        return {_data, torch::tensor(0)};
                        //--------------------------
                    }// end if (m_data_iter == m_data.begin())
                    //--------------------------------------------------------------
                    if(std::next(m_data_iter, batch) == m_data.end()-1){
                        //--------------------------
                        _data = *m_data_iter;
                        //--------------------------
                        for (size_t i = 1; i < batch; ++i){
                            //--------------------------
                            _data = torch::cat({_data, *++m_data_iter});
                            //--------------------------
                        }// end for (size_t i = 1; i < batch; i++)
                        //--------------------------
                        epsilon = this->calculate_epsilon();
                        done = true;
                        //--------------------------
                        return {_data, m_CostFunction(args...)};
                        //--------------------------
                    }// if(m_data_iter == m_data.end())
                    //--------------------------------------------------------------
                    if(m_data_iter != m_data.end()-1 and std::next(m_data_iter, batch) != m_data.end()-1){
                        //--------------------------
                        _data = *++m_data_iter;
                        //--------------------------
                        for(size_t i = 1; i < batch; ++i){
                            //--------------------------
                            _data = torch::cat({_data, *++m_data_iter});
                            //--------------------------
                        }// end for(size_t i = 0; i < batch; ++i)
                        //--------------------------
                    }// end if(m_data_iter != m_data.end()-1 && std::next(m_data_iter, batch) != m_data.end()-1)
                    //--------------------------
                    epsilon = this->calculate_epsilon();
                    done = false;
                    //--------------------------
                    return {_data, m_CostFunction(args...)};
                    //--------------------------
                }// end std::tuple<torch::Tensor, COST_OUTPUT> internal_step(double& epsilon, bool& done, const size_t& batch, Args... args)
                //--------------------------------------------------------------
                std::tuple<torch::Tensor, double> get_first_internal(const size_t& batch){
                    //--------------------------
                    if (m_data_iter == m_data.end() or std::next(m_data_iter, batch) == m_data.end()){
                        //--------------------------
                        throw std::out_of_range("End Of The Data Iterator");
                        //--------------------------
                    }// end if (m_data_iter == m_data.end() or std::next(m_data_iter, batch) == m_data.end())
                    //--------------------------
                    if (m_data_iter == m_data.begin() and std::next(m_data_iter, batch) != m_data.end()-1){
                        //--------------------------
                        torch::Tensor _data;
                        //--------------------------
                        auto epsilon = this->calculate_epsilon();
                        //--------------------------
                        _data = *m_data_iter;
                        //--------------------------
                        for(size_t i = 1; i < batch; ++i){
                            //--------------------------
                            _data = torch::cat({_data, *++m_data_iter});
                            //--------------------------
                        }// end for(size_t i = 0; i < batch; ++i)
                        //--------------------------
                        return {_data, epsilon};
                        //--------------------------
                    }// end if (m_data_iter == m_data.begin() and std::next(m_data_iter, batch) != m_data.end()-1)
                    //--------------------------
                    return {torch::tensor(0), 0};
                    //--------------------------
                }// end torch::Tensor get_first_internal(void)
                //--------------------------------------------------------------
                torch::Tensor get_first_internal(OUT double& epsilon, const size_t& batch){
                    //--------------------------
                    if (m_data_iter == m_data.end() or std::next(m_data_iter, batch) == m_data.end()){
                        //--------------------------
                        throw std::out_of_range("End Of The Data Iterator");
                        //--------------------------
                    }// end if (m_data_iter == m_data.end() or std::next(m_data_iter, batch) == m_data.end())
                    //--------------------------
                    if (m_data_iter == m_data.begin() and std::next(m_data_iter, batch) != m_data.end()-1){
                        //--------------------------
                        torch::Tensor _data;
                        //--------------------------
                        epsilon = this->calculate_epsilon();
                        //--------------------------
                        _data = *m_data_iter;
                        //--------------------------
                        for(size_t i = 1; i < batch; ++i){
                            //--------------------------
                            _data = torch::cat({_data, *++m_data_iter});
                            //--------------------------
                        }// end for(size_t i = 0; i < batch; ++i)
                        //--------------------------
                        return _data;
                        //--------------------------
                    }// end (m_data_iter == m_data.begin() and std::next(m_data_iter, batch) != m_data.end()-1)
                    //--------------------------
                    epsilon = 0.;
                    //--------------------------
                    return torch::tensor(0);
                    //--------------------------
                }// end torch::Tensor get_first_internal(void)
                //--------------------------------------------------------------
            private:
                //--------------------------------------------------------------
                std::vector<T>& m_data;
                typename std::vector<T>::iterator& m_data_iter;
                //--------------------------
                std::function<COST_OUTPUT(const Args&...)>& m_CostFunction;
                //--------------------------
                size_t m_batch;
            //--------------------------------------------------------------
        };// end class RLEnvironmentLoader
        //--------------------------------------------------------------
    }// end namespace Environment
    //--------------------------------------------------------------
}// end namespace RL
//--------------------------------------------------------------