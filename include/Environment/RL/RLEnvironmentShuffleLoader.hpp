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
                explicit RLEnvironmentShuffleLoader(std::vector<T>&& data, 
                                                    std::function<COST_OUTPUT(const Args&...)>&& costFunction,
                                                    const size_t& batch = 2ul,
                                                    const double& egreedy = 0.9,
                                                    const double& egreedy_final = 0.02,
                                                    const double& egreedy_decay = 500.) : RLEnvironmentShuffle<T, COST_OUTPUT, Args...>(std::move(data),
                                                                                                                                        std::move(costFunction), 
                                                                                                                                        egreedy, 
                                                                                                                                        egreedy_final, 
                                                                                                                                        egreedy_decay),
                                                                                        m_batch(batch){
                    //--------------------------
                    if(batch >= this->get_data().size()/2){
                        //--------------------------
                        throw std::out_of_range("Batch Size: [" + std::to_string(batch) + "] Must Be Less Then The data Size: [" + std::to_string(this->get_data().size()/2) + "]");
                        //--------------------------
                    }// end if(batch >= this->get_data().size()/2)
                    //----------------------------        
                }// end RLEnvironmentShuffleLoader(Dataset&& data_loader)
                //--------------------------------------------------------------
                RLEnvironmentShuffleLoader(const RLEnvironmentShuffleLoader&)             = default;
                RLEnvironmentShuffleLoader& operator=(RLEnvironmentShuffleLoader&&)       = default;
                //----------------------------
                RLEnvironmentShuffleLoader(RLEnvironmentShuffleLoader&&)                  = default;
                RLEnvironmentShuffleLoader& operator=(const RLEnvironmentShuffleLoader&)  = default;
                //--------------------------------------------------------------
                /**
                 * @brief 
                 * 
                 * @param args 
                 * @return std::tuple<torch::Tensor, COST_OUTPUT, double, bool> 
                 */
                virtual std::tuple<torch::Tensor, COST_OUTPUT, double, bool> step(const Args&... args) override{
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
                virtual std::tuple<torch::Tensor, COST_OUTPUT> step(OUT double& epsilon, OUT bool& done, const Args&... args) override{
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
                virtual std::tuple<torch::Tensor, double> get_first(void) override{
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
                virtual torch::Tensor get_first(OUT double& epsilon) override{
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
                using RLEnvironmentShuffle<T, COST_OUTPUT, Args...>::internal_step;  // Bring base class functions into scope
                //--------------------------
                using RLEnvironmentShuffle<T, COST_OUTPUT, Args...>::get_first_internal;  // Bring base class functions into scope
                //--------------------------------------------------------------
                std::tuple<torch::Tensor, COST_OUTPUT, double, bool> internal_step(const size_t& batch, const Args&... args){
                    //--------------------------------------------------------------
                    if (this->get_iterator() == this->get_data().end() or std::next(this->get_iterator(), batch) == this->get_data().end()){
                        //--------------------------
                        throw std::out_of_range("End Of The Data Iterator");
                        //--------------------------
                    }// end if (this->get_iterator() == this->get_data().end() or std::next(this->get_iterator(), batch) == this->get_data().end())
                    //--------------------------
                    if (this->get_iterator() == this->get_data().begin() and std::next(this->get_iterator(), batch) != this->get_data().end()-1){
                        //--------------------------
                        torch::Tensor _data;
                        double epsilon;
                        //--------------------------
                        std::tie(_data, epsilon) = get_first_internal(batch);
                        //--------------------------
                        return {_data, torch::tensor(NULL), epsilon, false};
                        //--------------------------
                    }// end if (this->get_iterator() == this->get_data().begin())
                    //--------------------------------------------------------------
                    std::vector<torch::Tensor> _data;
                    _data.reserve(batch);
                    //--------------------------
                    auto _data_end = std::next(this->get_iterator(), batch);
                    //--------------------------------------------------------------
                    if(_data_end == this->get_data().end()-1){
                        //--------------------------
                        for(; this->get_iterator() != _data_end; ++this->get_iterator()) {
                            //--------------------------
                            auto _random_position = this->generate_random_position();
                            //--------------------------
                            _data.push_back(*std::next(this->get_data().begin(), _random_position));
                            //--------------------------
                            this->get_distribution().at(_random_position) = 0;
                            //--------------------------
                        }// end for(; this->get_iterator() != _data_end; ++this->get_iterator())
                        //--------------------------
                        return {torch::cat(_data, 0), this->cost_function(args...), this->calculate_epsilon(), true};
                        //--------------------------
                    }// if(this->get_iterator() == this->get_data().end())
                    //--------------------------------------------------------------
                    for(; this->get_iterator() != _data_end; ++this->get_iterator()) {
                        //--------------------------
                        auto _random_position = this->generate_random_position();
                        //--------------------------
                        _data.push_back(*std::next(this->get_data().begin(), _random_position));
                        //--------------------------
                        this->get_distribution().at(_random_position) = 0;
                        //--------------------------
                    }// end for(; this->get_iterator() != _data_end; ++this->get_iterator())
                    //--------------------------
                    return {torch::cat(_data, 0), this->cost_function(args...), this->calculate_epsilon(), false};
                    //--------------------------
                }// end std::tuple<torch::Tensor, COST_OUTPUT, double, bool> internal_step(const size_t& batch, Args... args)
                //--------------------------------------------------------------
                std::tuple<torch::Tensor, COST_OUTPUT> internal_step(OUT double& epsilon, OUT bool& done, const size_t& batch, const Args&... args){
                    //--------------------------------------------------------------
                    if (this->get_iterator() == this->get_data().end() or std::next(this->get_iterator(), batch) == this->get_data().end()){
                        //--------------------------
                        throw std::out_of_range("End Of The Data Iterator");
                        //--------------------------
                    }// end if (this->get_iterator() == this->get_data().end() or std::next(this->get_iterator(), batch) == this->get_data().end())
                    //--------------------------------------------------------------
                    if (this->get_iterator() == this->get_data().begin() and std::next(this->get_iterator(), batch) != this->get_data().end()-1){
                        //--------------------------
                        done = false;
                        //--------------------------
                        return {get_first_internal(epsilon, batch), torch::tensor(0)};
                        //--------------------------
                    }// end if (this->get_iterator() == this->get_data().begin())
                    //--------------------------------------------------------------
                    std::vector<torch::Tensor> _data;
                    _data.reserve(batch);
                    //--------------------------
                    auto _data_end = std::next(this->get_iterator(), batch);
                    //--------------------------
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::discrete_distribution<> random_distribution(this->get_distribution().begin(), this->get_distribution().end());
                    //--------------------------------------------------------------
                    if(_data_end == this->get_data().end()-1){
                        //--------------------------
                        for(; this->get_iterator() != _data_end; ++this->get_iterator()) {
                            //--------------------------
                            auto _random_position = this->generate_random_position();
                            //--------------------------
                            _data.push_back(*std::next(this->get_data().begin(), _random_position));
                            //--------------------------
                            this->get_distribution().at(_random_position) = 0;
                            //--------------------------
                        }// end for(; this->get_iterator() != _data_end; ++this->get_iterator())
                        //--------------------------
                        epsilon = this->calculate_epsilon();
                        done = true;
                        //--------------------------
                        return {torch::cat(_data, 0), this->cost_function(args...)};
                        //--------------------------
                    }// if(this->get_iterator() == this->get_data().end())
                    //--------------------------------------------------------------
                    for(; this->get_iterator() != _data_end; ++this->get_iterator()) {
                        //--------------------------
                        auto _random_position = this->generate_random_position();
                        //--------------------------
                        _data.push_back(*std::next(this->get_data().begin(), _random_position));
                        //--------------------------
                        this->get_distribution().at(_random_position) = 0;
                        //--------------------------
                    }// end for(; this->get_iterator() != _data_end; ++this->get_iterator())
                    //--------------------------
                    epsilon = this->calculate_epsilon();
                    done = false;
                    //--------------------------
                    return {torch::cat(_data, 0), this->cost_function(args...)};
                    //--------------------------
                }// end std::tuple<torch::Tensor, COST_OUTPUT> internal_step(double& epsilon, bool& done, const size_t& batch, Args... args)
                //--------------------------------------------------------------
                std::tuple<torch::Tensor, double> get_first_internal(const size_t& batch){
                    //--------------------------
                    if (this->get_iterator() == this->get_data().end() or std::next(this->get_iterator(), batch) == this->get_data().end()){
                        //--------------------------
                        throw std::out_of_range("End Of The Data Iterator");
                        //--------------------------
                    }// end if (this->get_iterator() == this->get_data().end() or std::next(this->get_iterator(), batch) == this->get_data().end())
                    //--------------------------
                    if (this->get_iterator() == this->get_data().begin() and std::next(this->get_iterator(), batch) != this->get_data().end()-1){
                        //--------------------------
                        std::vector<torch::Tensor> _data;
                        _data.reserve(batch);
                        //--------------------------
                        std::random_device rd;
                        std::mt19937 gen(rd());
                        std::discrete_distribution<> random_distribution(this->get_distribution().begin(), this->get_distribution().end());
                        //--------------------------
                        auto epsilon = this->calculate_epsilon();
                        //--------------------------
                        auto _data_end = std::next(this->get_iterator(), batch);
                        //--------------------------
                        for(; this->get_iterator() != _data_end; ++this->get_iterator()) {
                            //--------------------------
                            auto _random_position = this->generate_random_position();
                            //--------------------------
                            _data.push_back(*std::next(this->get_data().begin(), _random_position));
                            //--------------------------
                            this->get_distribution().at(_random_position) = 0;
                            //--------------------------
                        }// end for(; this->get_iterator() != _data_end; ++this->get_iterator())
                        //--------------------------
                        --this->get_iterator();
                        //--------------------------
                        return {torch::cat(_data, 0), epsilon};
                        //--------------------------
                    }// end if (this->get_iterator() == this->get_data().begin() and std::next(this->get_iterator(), batch) != this->get_data().end()-1)
                    //--------------------------
                    return {torch::tensor(0), 0};
                    //--------------------------
                }// end torch::Tensor get_first_internal(void)
                //--------------------------------------------------------------
                torch::Tensor get_first_internal(OUT double& epsilon, const size_t& batch){
                    //--------------------------
                    if (this->get_iterator() == this->get_data().end() or std::next(this->get_iterator(), batch) == this->get_data().end()){
                        //--------------------------
                        throw std::out_of_range("End Of The Data Iterator");
                        //--------------------------
                    }// end if (this->get_iterator() == this->get_data().end() or std::next(this->get_iterator(), batch) == this->get_data().end())
                    //--------------------------
                    if (this->get_iterator() == this->get_data().begin() and std::next(this->get_iterator(), batch) != this->get_data().end()-1){
                        //--------------------------
                        std::vector<torch::Tensor> _data;
                        _data.reserve(batch);
                        //--------------------------
                        std::random_device rd;
                        std::mt19937 gen(rd());
                        std::discrete_distribution<> random_distribution(this->get_distribution().begin(), this->get_distribution().end());
                        //--------------------------
                        epsilon = this->calculate_epsilon();
                        //--------------------------
                        auto _data_end = std::next(this->get_iterator(), batch);
                        //--------------------------
                        for(; this->get_iterator() != _data_end; ++this->get_iterator()) {
                            //--------------------------
                            auto _random_position = this->generate_random_position();
                            //--------------------------
                            _data.push_back(*std::next(this->get_data().begin(), _random_position));
                            //--------------------------
                            this->get_distribution().at(_random_position) = 0;
                            //--------------------------
                        }// end for(; this->get_iterator() != _data_end; ++this->get_iterator())
                        //--------------------------
                        --this->get_iterator();
                        //--------------------------
                        return torch::cat(_data, 0);
                        //--------------------------
                    }// end (this->get_iterator() == this->get_data().begin() and std::next(this->get_iterator(), batch) != this->get_data().end()-1)
                    //--------------------------
                    epsilon = 0.;
                    //--------------------------
                    return torch::tensor(0);
                    //--------------------------
                }// end torch::Tensor get_first_internal(void)
                //--------------------------------------------------------------
            private:
                //--------------------------------------------------------------
                size_t m_batch;
            //--------------------------------------------------------------
        };// end class RLEnvironmentShuffleLoader
        //--------------------------------------------------------------
    }// end namespace Environment
    //--------------------------------------------------------------
}// end namespace RL
//--------------------------------------------------------------