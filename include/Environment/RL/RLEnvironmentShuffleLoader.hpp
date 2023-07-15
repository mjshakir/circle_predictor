#pragma once
//--------------------------------------------------------------
/* From: https://stackoverflow.com/questions/14803112/short-way-to-stdbind-member-function-to-object-instance-without-binding-param 
    and https://stackoverflow.com/questions/70355767/binding-a-class-method-to-a-method-of-another-class 
    and https://stackoverflow.com/questions/28746744/passing-capturing-lambda-as-function-pointer */
//--------------------------------------------------------------
// User Defined library
//--------------------------------------------------------------
#include "Environment/RL/RLEnvironmentShuffle.hpp"
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <random>
//-------------------
#include <algorithm>
#include <execution>
#include <mutex>
//--------------------------------------------------------------
namespace RL {
    //--------------------------------------------------------------
    namespace Environment {
        //--------------------------------------------------------------
        template<typename T, typename COST_OUTPUT, typename... Args>
        //--------------------------------------------------------------
        class RLEnvironmentShuffleLoader : public RLEnvironmentShuffle<T, COST_OUTPUT, Args...> {
            //--------------------------------------------------------------
            public:
                //--------------------------------------------------------------
                RLEnvironmentShuffleLoader(void) = delete;
                //--------------------------------------------------------------
                virtual ~RLEnvironmentShuffleLoader(void) = default;
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
                explicit RLEnvironmentShuffleLoader( std::vector<T>&& data, 
                                            std::function<COST_OUTPUT(const Args&...)> costFunction,
                                            const size_t& batch = 2ul,
                                            const double& egreedy = 0.9,
                                            const double& egreedy_final = 0.02,
                                            const double& egreedy_decay = 500.) : RLEnvironmentShuffle<T, COST_OUTPUT, Args...>(std::move(data),
                                                                                                                                std::move(costFunction), 
                                                                                                                                egreedy, 
                                                                                                                                egreedy_final, 
                                                                                                                                egreedy_decay),
                                                                                  m_data(this->get_data()),
                                                                                  m_data_iter(this->get_iterator()),
                                                                                  m_CostFunction(this->get_cost_function()),
                                                                                  m_batch(batch),
                                                                                  m_distribution(this->get_distribution()){
                    //--------------------------
                    if(batch >= m_data.size()/2){
                        //--------------------------
                        throw std::out_of_range("Batch Size: [" + std::to_string(batch) + "] Must Be Less Then The data Size: [" + std::to_string(m_data.size()/2) + "]");
                        //--------------------------
                    }// end if(batch >= m_data.size()/2)
                    //----------------------------        
                }// end RLEnvironmentShuffleLoader(Dataset&& data_loader)
                //--------------------------------------------------------------
                //Define copy constructor explicitly
                RLEnvironmentShuffleLoader(const RLEnvironmentShuffleLoader& other) :   RLEnvironmentShuffle<T, COST_OUTPUT, Args...>(other),
                                                                                        m_data(other.m_data),
                                                                                        m_data_iter(other.m_data_iter),
                                                                                        m_CostFunction(other.m_CostFunction),
                                                                                        m_batch(other.m_batch),
                                                                                        m_distribution(other.m_distribution) {
                    //--------------------------
                }// end RLEnvironmentLoaderAtomic(const RLEnvironmentLoaderAtomic& other)
                //--------------------------------------------------------------
                //Copy assignment operator
                RLEnvironmentShuffleLoader& operator=(const RLEnvironmentShuffleLoader& other) {
                    //--------------------------
                    // Check for self-assignment
                    if (this == &other) {
                        return *this;
                    }// end if (this == &other)
                    //--------------------------
                    // Perform a deep copy of the data
                    RLEnvironmentShuffle<T, COST_OUTPUT, Args...>::operator=(other);
                    m_data          = other.m_data;
                    m_data_iter     = other.m_data_iter;
                    m_CostFunction  = other.m_CostFunction;
                    m_batch         = other.m_batch;
                    //--------------------------
                    return *this;
                    //--------------------------
                }// end RLEnvironmentLoader& operator=(const RLEnvironmentLoader& other)
                //--------------------------------------------------------------
                /**
                 * @brief 
                 * 
                 * @param args 
                 * @return std::tuple<torch::Tensor, COST_OUTPUT, double, bool> 
                 */
                virtual std::tuple<torch::Tensor, COST_OUTPUT, double, bool> step(const Args&... args) override{
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
                virtual std::tuple<torch::Tensor, COST_OUTPUT> step(OUT double& epsilon, OUT bool& done, const Args&... args) override{
                    //----------------------------
                    return internal_step(epsilon, done, args...);
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
                virtual std::tuple<torch::Tensor, double> get_first(void) override{
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
                virtual torch::Tensor get_first(OUT double& epsilon) override{
                    //----------------------------
                    return get_first_internal(epsilon);
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
                using RLEnvironmentShuffle<T, COST_OUTPUT, Args...>::internal_step;  // Bring base class functions into scope
                //--------------------------
                using RLEnvironmentShuffle<T, COST_OUTPUT, Args...>::get_first_internal;  // Bring base class functions into scope
                //--------------------------------------------------------------
                std::tuple<torch::Tensor, COST_OUTPUT, double, bool> internal_step(const size_t& batch, const Args&... args){
                    //--------------------------------------------------------------
                    if (m_data_iter == m_data.end() or std::next(m_data_iter, batch) == m_data.end()){
                        //--------------------------
                        throw std::out_of_range("End Of The Data Iterator");
                        //--------------------------
                    }// end if (m_data_iter == m_data.end() or std::next(m_data_iter, batch) == m_data.end())
                    //--------------------------
                    if (m_data_iter == m_data.begin() and std::next(m_data_iter, batch) != m_data.end()-1){
                        //--------------------------
                        torch::Tensor _data;
                        double epsilon;
                        //--------------------------
                        std::tie(_data, epsilon) = get_first_internal(batch);
                        //--------------------------
                        return {_data, torch::tensor(NULL), epsilon, false};
                        //--------------------------
                    }// end if (m_data_iter == m_data.begin())
                    //--------------------------------------------------------------
                    std::vector<torch::Tensor> _data;
                    _data.reserve(batch);
                    //--------------------------
                    auto _data_end = std::next(m_data_iter, batch);
                    //--------------------------
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::discrete_distribution<> random_distribution(m_distribution.begin(), m_distribution.end());
                    //--------------------------------------------------------------
                    if(_data_end == m_data.end()-1){
                        //--------------------------
                        for(; m_data_iter != _data_end; ++m_data_iter) {
                            //--------------------------
                            auto _random_position = random_distribution(gen);
                            //--------------------------
                            _data.push_back(*std::next(m_data.begin(), _random_position));
                            //--------------------------
                            m_distribution.at(_random_position) = 0;
                            //--------------------------
                        }// end for(; m_data_iter != _data_end; ++m_data_iter)
                        //--------------------------
                        return {torch::cat(_data, 0), m_CostFunction(args...), this->calculate_epsilon(), true};
                        //--------------------------
                    }// if(m_data_iter == m_data.end())
                    //--------------------------------------------------------------
                    for(; m_data_iter != _data_end; ++m_data_iter) {
                        //--------------------------
                        auto _random_position = random_distribution(gen);
                        //--------------------------
                        _data.push_back(*std::next(m_data.begin(), _random_position));
                        //--------------------------
                        m_distribution.at(_random_position) = 0;
                        //--------------------------
                    }// end for(; m_data_iter != _data_end; ++m_data_iter)
                    //--------------------------
                    return {torch::cat(_data, 0), m_CostFunction(args...), this->calculate_epsilon(), false};
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
                    //--------------------------------------------------------------
                    if (m_data_iter == m_data.begin() and std::next(m_data_iter, batch) != m_data.end()-1){
                        //--------------------------
                        done = false;
                        //--------------------------
                        return {get_first_internal(epsilon, batch), torch::tensor(0)};
                    }// end if (m_data_iter == m_data.begin())
                    //--------------------------------------------------------------
                    std::vector<torch::Tensor> _data;
                    _data.reserve(batch);
                    //--------------------------
                    auto _data_end = std::next(m_data_iter, batch);
                    //--------------------------
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::discrete_distribution<> random_distribution(m_distribution.begin(), m_distribution.end());
                    //--------------------------------------------------------------
                    if(_data_end == m_data.end()-1){
                        //--------------------------
                        for(; m_data_iter != _data_end; ++m_data_iter) {
                            //--------------------------
                            auto _random_position = random_distribution(gen);
                            //--------------------------
                            _data.push_back(*std::next(m_data.begin(), _random_position));
                            //--------------------------
                            m_distribution.at(_random_position) = 0;
                            //--------------------------
                        }// end for(; m_data_iter != _data_end; ++m_data_iter)
                        //--------------------------
                        epsilon = this->calculate_epsilon();
                        done = true;
                        //--------------------------
                        return {torch::cat(_data, 0), m_CostFunction(args...)};
                        //--------------------------
                    }// if(m_data_iter == m_data.end())
                    //--------------------------------------------------------------
                    for(; m_data_iter != _data_end; ++m_data_iter) {
                        //--------------------------
                        auto _random_position = random_distribution(gen);
                        //--------------------------
                        _data.push_back(*std::next(m_data.begin(), _random_position));
                        //--------------------------
                        m_distribution.at(_random_position) = 0;
                        //--------------------------
                    }// end for(; m_data_iter != _data_end; ++m_data_iter)
                    //--------------------------
                    epsilon = this->calculate_epsilon();
                    done = false;
                    //--------------------------
                    return {torch::cat(_data, 0), m_CostFunction(args...)};
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
                        std::vector<torch::Tensor> _data;
                        _data.reserve(batch);
                        //--------------------------
                        std::random_device rd;
                        std::mt19937 gen(rd());
                        std::discrete_distribution<> random_distribution(m_distribution.begin(), m_distribution.end());
                        //--------------------------
                        auto epsilon = this->calculate_epsilon();
                        //--------------------------
                        auto _data_end = std::next(m_data_iter, batch);
                        //--------------------------
                        for(; m_data_iter != _data_end; ++m_data_iter) {
                            //--------------------------
                            auto _random_position = random_distribution(gen);
                            //--------------------------
                            _data.push_back(*std::next(m_data.begin(), _random_position));
                            //--------------------------
                            m_distribution.at(_random_position) = 0;
                            //--------------------------
                        }// end for(; m_data_iter != _data_end; ++m_data_iter)
                        //--------------------------
                        return {torch::cat(_data, 0), epsilon};
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
                        std::vector<torch::Tensor> _data;
                        _data.reserve(batch);
                        //--------------------------
                        std::random_device rd;
                        std::mt19937 gen(rd());
                        std::discrete_distribution<> random_distribution(m_distribution.begin(), m_distribution.end());
                        //--------------------------
                        auto epsilon = this->calculate_epsilon();
                        //--------------------------
                        auto _data_end = std::next(m_data_iter, batch);
                        //--------------------------
                        for(; m_data_iter != _data_end; ++m_data_iter) {
                            //--------------------------
                            auto _random_position = random_distribution(gen);
                            //--------------------------
                            _data.push_back(*std::next(m_data.begin(), _random_position));
                            //--------------------------
                            m_distribution.at(_random_position) = 0;
                            //--------------------------
                        }// end for(; m_data_iter != _data_end; ++m_data_iter)
                        //--------------------------
                        return torch::cat(_data, 0);
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
                //--------------------------
                std::vector<uint8_t>& m_distribution;
            //--------------------------------------------------------------
        };// end class RLEnvironmentShuffleLoader
        //--------------------------------------------------------------
    }// end namespace Environment
    //--------------------------------------------------------------
}// end namespace RL
//--------------------------------------------------------------