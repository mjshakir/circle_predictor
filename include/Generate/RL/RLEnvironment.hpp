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
// Standard library
//--------------------------------------------------------------
#include <random>
//-------------------
#include <algorithm>
#include <execution>
//--------------------------------------------------------------
// User definition
//--------------------------------------------------------------
#define OUT
//--------------------------------------------------------------
template<typename T, typename COST_OUTPUT, typename... Args>
//--------------------------------------------------------------
class RLEnvironment{
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
                        const double& egreedy_decay = 500.,
                        const size_t& batch = 1ul,
                        const bool& randomizer = false) :   m_data(std::move(data)),
                                                            m_data_iter (m_data.begin()), 
                                                            m_CostFunction(std::move(costFunction)),
                                                            m_egreedy(egreedy),
                                                            m_egreedy_final(egreedy_final),
                                                            m_egreedy_decay(egreedy_decay),
                                                            m_enable_batch((batch > 1) ? true : false),
                                                            m_batch(batch),
                                                            m_randomizer(randomizer){
            //----------------------------
            if(m_enable_batch and batch >= m_data.size()/2){
                //--------------------------
                throw std::out_of_range("Batch Size: [" + std::to_string(batch) + "] Must Be Less Then The data Size: [" + std::to_string(m_data.size()/2) + "]");
                //--------------------------
            }// end if(m_enable_batch and batch >= m_data.size()/2)
            //----------------------------
            if(egreedy_decay == 0.){
                //----------------------------
                throw std::runtime_error("Egreedy Decay Cannot Be Zero. egreedy_decay Value: [" + std::to_string(egreedy_decay) + "]");
                //----------------------------
            }// end if(egreedy_decay == 0.)
            //----------------------------
            if(randomizer){
                //----------------------------
                m_distribution.reserve(m_data.size());
                //----------------------------
                std::fill_n(std::execution::par, std::inserter(m_distribution, m_distribution.begin()), m_data.size(), 1u);
                //----------------------------
            }// end if(randomizer)
            //----------------------------
        }// end RLEnvironment(Dataset&& data_loader)
        //--------------------------------------------------------------m_batch
        /**
         * @brief 
         * 
         * @param args 
         * @return std::tuple<torch::Tensor, COST_OUTPUT, double, bool> 
         */
        std::tuple<torch::Tensor, COST_OUTPUT, double, bool> step(const Args&... args){
            //----------------------------
            return internal_step_control(args...);
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
        std::tuple<torch::Tensor, COST_OUTPUT> step(OUT double& epsilon, OUT bool& done, const Args&... args){
            //----------------------------
            return internal_step_control(epsilon, done, args...);
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
            return internal_step_control(epsilon, done, batch, args...);
            //----------------------------
        }// std::tuple<torch::Tensor, COST_OUTPUT> step(OUT double& epsilon, OUT bool& done, const size_t& batch, const Args&... args)
        //--------------------------------------------------------------
        /**
         * @brief Get the first object
         * 
         * @return std::tuple<torch::Tensor, double> 
         */
        std::tuple<torch::Tensor, double> get_first(void){
            //----------------------------
            return get_first_internal_control();
            //----------------------------
        }// end torch::Tensor get_first(void)
        //--------------------------------------------------------------
        /**
         * @brief Get the first object
         * 
         * @param epsilon 
         * @return torch::Tensor 
         */
        torch::Tensor get_first(OUT double& epsilon){
            //----------------------------
            return get_first_internal_control(epsilon);
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
            return get_first_internal_control(epsilon, batch);
            //----------------------------
        }// end torch::Tensor get_first(void)
        //--------------------------------------------------------------
        /**
         * @brief 
         * 
         */
        void reset(void){
            //--------------------------
            reset_iterator();
            //--------------------------
        }// end void reset(void)
        //--------------------------------------------------------------
    protected:
        //--------------------------------------------------------------
        std::tuple<torch::Tensor, COST_OUTPUT, double, bool> internal_step_control(const Args&... args){
            //----------------------------
            if(m_enable_batch){
                //----------------------------
                if(m_randomizer){
                    //----------------------------
                    return internal_random_step(m_batch, args...);
                    //----------------------------
                }// end if(m_randomizer)
                //----------------------------
                return internal_step(m_batch, args...);
                //----------------------------
            }// end if(m_enable_batch)
            //----------------------------
            if(m_randomizer){
                //----------------------------
                return internal_random_step(args...);
                //----------------------------
            }// end if(m_randomizer)
            //----------------------------
            return internal_step(args...);
            //----------------------------
        }// end std::tuple<torch::Tensor, COST_OUTPUT, double, bool> internal_step_control(const Args&... args)
        //--------------------------------------------------------------
        std::tuple<torch::Tensor, COST_OUTPUT> internal_step_control(OUT double& epsilon, OUT bool& done, const Args&... args){
            //----------------------------
            if(m_enable_batch){
                //----------------------------
                if(m_randomizer){
                    //----------------------------
                    return internal_random_step(epsilon, done, m_batch, args...);
                    //----------------------------
                }// end if(m_randomizer)
                //----------------------------
                return internal_step(epsilon, done, m_batch, args...);
                //----------------------------
            }// end if(m_enable_batch)
            //----------------------------
            if(m_randomizer){
                //----------------------------
                return internal_random_step(epsilon, done, args...);
                //----------------------------
            }// end if(m_randomizer)
            //----------------------------
            return internal_step(epsilon, done, args...);
            //----------------------------
        }// end std::tuple<torch::Tensor, COST_OUTPUT, double, bool> internal_step_control(const Args&... args)
        //--------------------------------------------------------------
        std::tuple<torch::Tensor, COST_OUTPUT> internal_step_control(OUT double& epsilon, OUT bool& done, const size_t& batch, const Args&... args){
            //----------------------------
            if(m_randomizer){
                //----------------------------
                return internal_random_step(epsilon, done, batch, args...);
                //----------------------------
            }// end if(m_randomizer)
            //----------------------------
            return internal_step(epsilon, done, batch, args...);
            //----------------------------
        }// end std::tuple<torch::Tensor, COST_OUTPUT, double, bool> internal_step_control(const Args&... args)
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
        std::tuple<torch::Tensor, COST_OUTPUT, double, bool> internal_random_step(const Args&... args){
            //--------------------------
            if (m_data_iter == m_data.end()){
                //--------------------------
                throw std::out_of_range("End Of The Data Iterator");
                //--------------------------
            }// end if (m_data_iter == m_data.end())
            //--------------------------
            std::random_device rd;
            std::mt19937 gen(rd());
            std::discrete_distribution<> random_distribution(m_distribution.begin(), m_distribution.end());
            //--------------------------
            auto _random_position = random_distribution(gen);
            //--------------------------
            if (m_data_iter == m_data.begin()){
                //--------------------------
                auto epsilon = calculate_epsilon();
                //--------------------------
                ++m_data_iter;
                //--------------------------
                m_distribution.at(_random_position) = 0;
                //--------------------------
                return {*std::next(m_data.begin(), _random_position), torch::tensor(NULL), epsilon, false};
                //--------------------------
            }// end if (m_data_iter == m_data.begin())
            //--------------------------
             if(m_data_iter == m_data.end()-1){
                //--------------------------
                return {*std::next(m_data.begin(), _random_position), m_CostFunction(args...), calculate_epsilon(), true};
                //--------------------------
            }// if(m_data_iter == m_data.end())
            //--------------------------
            if(m_data_iter != m_data.end()-1){
                //--------------------------
                ++m_data_iter;
                //--------------------------
                m_distribution.at(_random_position) = 0;
                //--------------------------
            }// if(m_data_iter == m_data.end())
            //--------------------------
            return {*std::next(m_data.begin(), _random_position), m_CostFunction(args...), calculate_epsilon(), false};
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
        std::tuple<torch::Tensor, COST_OUTPUT> internal_random_step(OUT double& epsilon, OUT bool& done, const Args&... args){
            //--------------------------
            if (m_data_iter == m_data.end()){
                //--------------------------
                throw std::out_of_range("End Of The Data Iterator");
                //--------------------------
            }// end if (m_data_iter == m_data.end())
            //--------------------------
            std::random_device rd;
            std::mt19937 gen(rd());
            std::discrete_distribution<> random_distribution(m_distribution.begin(), m_distribution.end());
            //--------------------------
            auto _random_position = random_distribution(gen);
            //--------------------------
            if (m_data_iter == m_data.begin()){
                //--------------------------
                auto input = *std::next(m_data.begin(), _random_position);
                //--------------------------
                epsilon = calculate_epsilon();
                done = false;
                //--------------------------
                m_distribution.at(_random_position) = 0;
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
                return {*std::next(m_data.begin(), _random_position), m_CostFunction(args...)};
                //--------------------------
            }// if(m_data_iter == m_data.end())
            //--------------------------
            if(m_data_iter != m_data.end()-1){
                //--------------------------
                ++m_data_iter;
                //--------------------------
                m_distribution.at(_random_position) = 0;
                //--------------------------
            }// if(m_data_iter == m_data.end())
            //--------------------------
            epsilon = calculate_epsilon();
            done = false;
            //--------------------------
            return {*std::next(m_data.begin(), _random_position), m_CostFunction(args...)};
            //--------------------------
        }// end std::tuple<torch::Tensor, COST_OUTPUT> internal_step(double& epsilon, bool& done, Args... args)
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
                auto epsilon = calculate_epsilon();
                //--------------------------
                _data = *m_data_iter;
                //--------------------------
                for(size_t i = 1; i < batch; ++i){
                    //--------------------------
                    ++m_data_iter;
                    //--------------------------
                    _data = torch::cat({_data, *m_data_iter});
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
                    ++m_data_iter;
                    //--------------------------
                    _data = torch::cat({_data, *m_data_iter});
                    //--------------------------
                }// end for (size_t i = 1; i < batch; i++)
                //--------------------------
                return {_data, m_CostFunction(args...), calculate_epsilon(), true};
                //--------------------------
            }// if(m_data_iter == m_data.end())
            //--------------------------------------------------------------
            if(m_data_iter != m_data.end()-1 and std::next(m_data_iter, batch) != m_data.end()-1){
                //--------------------------
                ++m_data_iter;
                //--------------------------
                _data = *m_data_iter;
                //--------------------------
                for(size_t i = 1; i < batch; ++i){
                    //--------------------------
                    ++m_data_iter;
                    //--------------------------
                    _data = torch::cat({_data, *m_data_iter});
                    //--------------------------
                }// end for(size_t i = 0; i < batch; ++i)
                //--------------------------
            }// end if(m_data_iter != m_data.end()-1 && std::next(m_data_iter, batch) != m_data.end()-1)
            //--------------------------
            return {_data, m_CostFunction(args...), calculate_epsilon(), false};
            //--------------------------
        }// end std::tuple<torch::Tensor, COST_OUTPUT, double, bool> internal_step(const size_t& batch, Args... args)
        //--------------------------------------------------------------
        std::tuple<torch::Tensor, COST_OUTPUT, double, bool> internal_random_step(const size_t& batch, const Args&... args){
            //--------------------------------------------------------------
            if (m_data_iter == m_data.end() or std::next(m_data_iter, batch) == m_data.end()){
                //--------------------------
                throw std::out_of_range("End Of The Data Iterator");
                //--------------------------
            }// end if (m_data_iter == m_data.end() or std::next(m_data_iter, batch) == m_data.end())
            //--------------------------
            torch::Tensor _data;
            //--------------------------
            std::random_device rd;
            std::mt19937 gen(rd());
            std::discrete_distribution<> random_distribution(m_distribution.begin(), m_distribution.end());
            //--------------------------------------------------------------
            if (m_data_iter == m_data.begin() and std::next(m_data_iter, batch) != m_data.end()-1){
                //--------------------------
                auto epsilon = calculate_epsilon();
                //--------------------------
                auto _random_position = random_distribution(gen);
                //--------------------------
                _data = *std::next(m_data.begin(), _random_position);
                //--------------------------
                m_distribution.at(_random_position) = 0;
                //--------------------------
                for(size_t i = 1; i < batch; ++i){
                    //--------------------------
                    ++m_data_iter;
                    //--------------------------
                    _random_position = random_distribution(gen);
                    //--------------------------
                    _data = torch::cat({_data, *std::next(m_data.begin(), _random_position)});
                    //--------------------------
                    m_distribution.at(_random_position) = 0;
                    //--------------------------
                }// end for(size_t i = 0; i < batch; ++i)
                //--------------------------
                return {_data, torch::tensor(NULL), epsilon, false};
                //--------------------------
            }// end if (m_data_iter == m_data.begin())
            //--------------------------------------------------------------
            if(std::next(m_data_iter, batch) == m_data.end()-1){
                //--------------------------
                auto _random_position = random_distribution(gen);
                //--------------------------
                _data = *std::next(m_data.begin(), _random_position);
                //--------------------------
                m_distribution.at(_random_position) = 0;
                //--------------------------
                for (size_t i = 1; i < batch; ++i){
                    //--------------------------
                    ++m_data_iter;
                    //--------------------------
                    _random_position = random_distribution(gen);
                    //--------------------------
                    _data = torch::cat({_data, *std::next(m_data.begin(), _random_position)});
                    //--------------------------
                    m_distribution.at(_random_position) = 0;
                    //--------------------------
                }// end for (size_t i = 1; i < batch; i++)
                //--------------------------
                return {_data, m_CostFunction(args...), calculate_epsilon(), true};
                //--------------------------
            }// if(m_data_iter == m_data.end())
            //--------------------------------------------------------------
            if(m_data_iter != m_data.end()-1 and std::next(m_data_iter, batch) != m_data.end()-1){
                //--------------------------
                ++m_data_iter;
                //--------------------------
                auto _random_position = random_distribution(gen);
                //--------------------------
                _data = *std::next(m_data.begin(), _random_position);
                //--------------------------
                m_distribution.at(_random_position) = 0;
                //--------------------------
                for(size_t i = 1; i < batch; ++i){
                    //--------------------------
                    ++m_data_iter;
                    //--------------------------
                    _random_position = random_distribution(gen);
                    //--------------------------
                    _data = torch::cat({_data, *std::next(m_data.begin(), _random_position)});
                    //--------------------------
                    m_distribution.at(_random_position) = 0;
                    //--------------------------
                }// end for(size_t i = 0; i < batch; ++i)
                //--------------------------
            }// end if(m_data_iter != m_data.end()-1 && std::next(m_data_iter, batch) != m_data.end()-1)
            //--------------------------
            return {_data, m_CostFunction(args...), calculate_epsilon(), false};
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
                epsilon = calculate_epsilon();
                done = false;
                //--------------------------
                _data = *m_data_iter;
                //--------------------------
                for(size_t i = 1; i < batch; ++i){
                    //--------------------------
                    ++m_data_iter;
                    //--------------------------
                    _data = torch::cat({_data, *m_data_iter});
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
                epsilon = calculate_epsilon();
                done = true;
                //--------------------------
                for (size_t i = 1; i < batch; ++i){
                    //--------------------------
                    ++m_data_iter;
                    //--------------------------
                    _data = torch::cat({_data, *m_data_iter});
                    //--------------------------
                }// end for (size_t i = 1; i < batch; i++)
                //--------------------------
                return {_data, m_CostFunction(args...)};
                //--------------------------
            }// if(m_data_iter == m_data.end())
            //--------------------------------------------------------------
            if(m_data_iter != m_data.end()-1 and std::next(m_data_iter, batch) != m_data.end()-1){
                //--------------------------
                ++m_data_iter;
                //--------------------------
                _data = *m_data_iter;
                //--------------------------
                for(size_t i = 1; i < batch; ++i){
                    //--------------------------
                    ++m_data_iter;
                    //--------------------------
                    _data = torch::cat({_data, *m_data_iter});
                    //--------------------------
                }// end for(size_t i = 0; i < batch; ++i)
                //--------------------------
            }// end if(m_data_iter != m_data.end()-1 && std::next(m_data_iter, batch) != m_data.end()-1)
            //--------------------------
            epsilon = calculate_epsilon();
            done = false;
            //--------------------------
            return {_data, m_CostFunction(args...)};
            //--------------------------
        }// end std::tuple<torch::Tensor, COST_OUTPUT> internal_step(double& epsilon, bool& done, const size_t& batch, Args... args)
        //--------------------------------------------------------------
        std::tuple<torch::Tensor, COST_OUTPUT> internal_random_step(OUT double& epsilon, OUT bool& done, const size_t& batch, const Args&... args){
            //--------------------------------------------------------------
            if (m_data_iter == m_data.end() or std::next(m_data_iter, batch) == m_data.end()){
                //--------------------------
                throw std::out_of_range("End Of The Data Iterator");
                //--------------------------
            }// end if (m_data_iter == m_data.end() or std::next(m_data_iter, batch) == m_data.end())
            //--------------------------
            torch::Tensor _data;
            //--------------------------
            std::random_device rd;
            std::mt19937 gen(rd());
            std::discrete_distribution<> random_distribution(m_distribution.begin(), m_distribution.end());
            //--------------------------------------------------------------
            if (m_data_iter == m_data.begin() and std::next(m_data_iter, batch) != m_data.end()-1){
                //--------------------------
                epsilon = calculate_epsilon();
                done = false;
                //--------------------------
                auto _random_position = random_distribution(gen);
                //--------------------------
                _data = *std::next(m_data.begin(), _random_position);
                //--------------------------
                m_distribution.at(_random_position) = 0;
                //--------------------------
                for(size_t i = 1; i < batch; ++i){
                    //--------------------------
                    ++m_data_iter;
                    //--------------------------
                    _random_position = random_distribution(gen);
                    //--------------------------
                    _data = torch::cat({_data, *std::next(m_data.begin(), _random_position)});
                    //--------------------------
                    m_distribution.at(_random_position) = 0;
                    //--------------------------
                }// end for(size_t i = 0; i < batch; ++i)
                //--------------------------
                return {_data, torch::tensor(0)};
                //--------------------------
            }// end if (m_data_iter == m_data.begin())
            //--------------------------------------------------------------
            if(std::next(m_data_iter, batch) == m_data.end()-1){
                //--------------------------
                auto _random_position = random_distribution(gen);
                //--------------------------
                _data = *std::next(m_data.begin(), _random_position);
                //--------------------------
                m_distribution.at(_random_position) = 0;
                //--------------------------
                epsilon = calculate_epsilon();
                done = true;
                //--------------------------
                for (size_t i = 1; i < batch; ++i){
                    //--------------------------
                    ++m_data_iter;
                    //--------------------------
                    _random_position = random_distribution(gen);
                    //--------------------------
                    _data = torch::cat({_data, *std::next(m_data.begin(), _random_position)});
                    //--------------------------
                    m_distribution.at(_random_position) = 0;
                //--------------------------
                }// end for (size_t i = 1; i < batch; i++)
                //--------------------------
                return {_data, m_CostFunction(args...)};
                //--------------------------
            }// if(m_data_iter == m_data.end())
            //--------------------------------------------------------------
            if(m_data_iter != m_data.end()-1 and std::next(m_data_iter, batch) != m_data.end()-1){
                //--------------------------
                ++m_data_iter;
                //--------------------------
                auto _random_position = random_distribution(gen);
                //--------------------------
                _data = *std::next(m_data.begin(), _random_position);
                //--------------------------
                m_distribution.at(_random_position) = 0;
                //--------------------------
                for(size_t i = 1; i < batch; ++i){
                    //--------------------------
                    ++m_data_iter;
                    //--------------------------
                    _random_position = random_distribution(gen);
                    //--------------------------
                    _data = torch::cat({_data, *std::next(m_data.begin(), _random_position)});
                    //--------------------------
                    m_distribution.at(_random_position) = 0;
                    //--------------------------
                }// end for(size_t i = 0; i < batch; ++i)
                //--------------------------
            }// end if(m_data_iter != m_data.end()-1 && std::next(m_data_iter, batch) != m_data.end()-1)
            //--------------------------
            epsilon = calculate_epsilon();
            done = false;
            //--------------------------
            return {_data, m_CostFunction(args...)};
            //--------------------------
        }// end std::tuple<torch::Tensor, COST_OUTPUT> internal_step(double& epsilon, bool& done, const size_t& batch, Args... args)
        //--------------------------------------------------------------
        std::tuple<torch::Tensor, double> get_first_internal_control(void){
            //----------------------------
            if(m_enable_batch){
                //----------------------------
                if(m_randomizer){
                    //----------------------------
                    return get_first_random_internal(m_batch);
                    //----------------------------
                }// end if(m_randomizer)
                //----------------------------
                return get_first_internal(m_batch);
                //----------------------------
            }// end if(m_enable_batch)
            //----------------------------
            if(m_randomizer){
                //----------------------------
                return get_first_random_internal();
                //----------------------------
            }// end if(m_randomizer)
            //----------------------------
            return get_first_internal();
            //----------------------------
        }// std::tuple<torch::Tensor, double> get_first_control(void)
        //--------------------------------------------------------------
        torch::Tensor get_first_internal_control(OUT double& epsilon){
            //----------------------------
            if(m_enable_batch){
                //----------------------------
                if(m_randomizer){
                    //----------------------------
                    return get_first_random_internal(epsilon, m_batch);
                    //----------------------------
                }// end if(m_randomizer)
                //----------------------------
                return get_first_internal(epsilon, m_batch);
                //----------------------------
            }// end if(m_enable_batch)
            //----------------------------
            if(m_randomizer){
                //----------------------------
                return get_first_random_internal(epsilon);
                //----------------------------
            }// end if(m_randomizer)
            //----------------------------
            return get_first_internal(epsilon);
            //----------------------------
        }// std::tuple<torch::Tensor, double> get_first_control(void)
        //--------------------------------------------------------------
        torch::Tensor get_first_internal_control(OUT double& epsilon, const size_t& batch){
            //----------------------------
            if(m_randomizer){
                //----------------------------
                return get_first_random_internal(epsilon, batch);
                //----------------------------
            }// end if(m_randomizer)
            //----------------------------
            return get_first_internal(epsilon, batch);
            //----------------------------
        }// std::tuple<torch::Tensor, double> get_first_control(void)
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
        std::tuple<torch::Tensor, double> get_first_random_internal(void){
            //--------------------------
            if (m_data_iter == m_data.end()){
                //--------------------------
                throw std::out_of_range("End Of The Data Iterator");
                //--------------------------
            }// end if (m_data_iter == m_data.end() or std::next(m_data_iter, batch) == m_data.end())
            //--------------------------
            if (m_data_iter == m_data.begin()){
                //--------------------------
                std::random_device rd;
                std::mt19937 gen(rd());
                std::discrete_distribution<> random_distribution(m_distribution.begin(), m_distribution.end());
                //--------------------------
                auto _random_position = random_distribution(gen);
                //--------------------------
                auto epsilon = calculate_epsilon();
                //--------------------------
                ++m_data_iter;
                //--------------------------
                m_distribution.at(_random_position) = 0;
                //--------------------------
                return {*std::next(m_data.begin(), _random_position), epsilon};
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
        torch::Tensor get_first_random_internal(OUT double& epsilon){
            //--------------------------
            if (m_data_iter == m_data.end()){
                //--------------------------
                throw std::out_of_range("End Of The Data Iterator");
                //--------------------------
            }// end if (m_data_iter == m_data.end() or std::next(m_data_iter, batch) == m_data.end())
            //--------------------------
            if (m_data_iter == m_data.begin()){
                //--------------------------
                std::random_device rd;
                std::mt19937 gen(rd());
                std::discrete_distribution<> random_distribution(m_distribution.begin(), m_distribution.end());
                //--------------------------
                auto _random_position = random_distribution(gen);
                //--------------------------
                epsilon = calculate_epsilon();
                //--------------------------
                ++m_data_iter;
                //--------------------------
                m_distribution.at(_random_position) = 0;
                //--------------------------
                return *std::next(m_data.begin(), _random_position);
                //--------------------------
            }// end if (m_data_iter == m_data.begin())
            //--------------------------
            epsilon = 0.;
            //--------------------------
            return torch::tensor(0);
            //--------------------------
        }// end torch::Tensor get_first_internal(void)
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
                auto epsilon = calculate_epsilon();
                //--------------------------
                _data = *m_data_iter;
                //--------------------------
                for(size_t i = 1; i < batch; ++i){
                    //--------------------------
                    ++m_data_iter;
                    //--------------------------
                    _data = torch::cat({ _data, *m_data_iter});
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
        std::tuple<torch::Tensor, double> get_first_random_internal(const size_t& batch){
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
                std::random_device rd;
                std::mt19937 gen(rd());
                std::discrete_distribution<> random_distribution(m_distribution.begin(), m_distribution.end());
                //--------------------------
                auto _random_position = random_distribution(gen);
                //--------------------------
                auto epsilon = calculate_epsilon();
                //--------------------------
                _data = *std::next(m_data.begin(), _random_position);
                //--------------------------
                for(size_t i = 1; i < batch; ++i){
                    //--------------------------
                    ++m_data_iter;
                    //--------------------------
                    _random_position = random_distribution(gen);
                    //--------------------------
                    _data = torch::cat({ _data, *std::next(m_data.begin(), _random_position)});
                    //--------------------------
                    m_distribution.at(_random_position) = 0;
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
            torch::Tensor _data;
            //--------------------------
            if (m_data_iter == m_data.begin() and std::next(m_data_iter, batch) != m_data.end()-1){
                //--------------------------
                epsilon = calculate_epsilon();
                //--------------------------
                _data = *m_data_iter;
                //--------------------------
                for(size_t i = 1; i < batch; ++i){
                    //--------------------------
                    ++m_data_iter;
                    //--------------------------
                    _data = torch::cat({ _data, *m_data_iter});
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
        torch::Tensor get_first_random_internal(OUT double& epsilon, const size_t& batch){
            //--------------------------
            torch::Tensor _data;
            //--------------------------
            std::random_device rd;
            std::mt19937 gen(rd());
            std::discrete_distribution<> random_distribution(m_distribution.begin(), m_distribution.end());
            //--------------------------------------------------------------
            if (m_data_iter == m_data.begin() and std::next(m_data_iter, batch) != m_data.end()-1){
                //--------------------------
                epsilon = calculate_epsilon();
                //--------------------------
                auto _random_position = random_distribution(gen);
                //--------------------------
                _data = *std::next(m_data_iter, _random_position);
                //--------------------------
                m_distribution.at(_random_position) = 0;
                //--------------------------
                for(size_t i = 1; i < batch; ++i){
                    //--------------------------
                    ++m_data_iter;
                    //--------------------------
                    _random_position = random_distribution(gen);
                    //--------------------------
                    _data = torch::cat({ _data, *std::next(m_data.begin(), _random_position)});
                    //--------------------------
                    m_distribution.at(_random_position) = 0;
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
        void reset_iterator(void){
            //--------------------------
            m_data_iter = m_data.begin();
            //--------------------------
            if(m_randomizer){
                //--------------------------
                std::fill(std::execution::par, m_distribution.begin(), m_distribution.end(), 1u);
                //--------------------------
            }// end if(m_randomizer)
            //----------------------------
        }// end void rest_iterator(void)
        //--------------------------------------------------------------
        constexpr double calculate_epsilon(void){
            //--------------------------
            return m_egreedy_final + (m_egreedy - m_egreedy_final) * std::exp(-1. * std::distance(m_data.begin(), m_data_iter) / m_egreedy_decay);
            //--------------------------
        }// end double calculate_epsilon()
        //--------------------------------------------------------------
    private:
        //--------------------------------------------------------------
        std::vector<T> m_data;
        typename std::vector<T>::iterator m_data_iter;
        //--------------------------
        std::function<COST_OUTPUT(const Args&...)> m_CostFunction;
        //--------------------------
        double m_egreedy, m_egreedy_final, m_egreedy_decay;
        //--------------------------
        bool m_enable_batch;
        //--------------------------
        size_t m_batch, m_randomizer;
        //--------------------------
        std::vector<uint8_t> m_distribution;
    //--------------------------------------------------------------
};// end class RLEnvironment
//--------------------------------------------------------------