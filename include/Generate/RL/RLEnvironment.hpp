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
        //--------------------------------------------------------------
        RLEnvironment(  std::vector<T>&& data, 
                        const size_t& batch = 1) :  m_data(std::move(data)),
                                                    m_data_iter (m_data.begin()), 
                                                    m_enable_batch((batch > 1) ? true : false),
                                                    m_batch(batch){
            //----------------------------
            if(m_enable_batch and batch >= m_data.size()/2){
                //--------------------------
                throw std::out_of_range("Batch Size: [" + std::to_string(batch) + "] Must Be Less Then The data Size: [" + std::to_string(m_data.size()/2) + "]");
                //--------------------------
            }// end if(batch > m_data.size()/2)
            //----------------------------
        }// end RLEnvironment(Dataset&& data_loader)
        //--------------------------------------------------------------
        RLEnvironment(  std::vector<T>&& data, 
                        std::function<COST_OUTPUT(const Args&...)> costFunction,
                        const double& egreedy = 0.9,
                        const double& egreedy_final = 0.02,
                        const double& egreedy_decay = 500.,
                        const size_t& batch = 1) :  m_data(std::move(data)),
                                                    m_data_iter (m_data.begin()), 
                                                    m_CostFunction(std::move(costFunction)),
                                                    m_egreedy(egreedy),
                                                    m_egreedy_final(egreedy_final),
                                                    m_egreedy_decay(egreedy_decay),
                                                    m_enable_batch((batch > 1) ? true : false),
                                                    m_batch(batch){
            //----------------------------
            if(m_enable_batch and batch >= m_data.size()/2){
                //--------------------------
                // std::string _error = "Batch Size: [" + std::to_string(batch) + "] Must Be Less Then The data Size: [" + std::to_string(m_data.size()/2) + "]";
                //--------------------------
                // throw std::out_of_range(_error.c_str());
                //--------------------------
                throw std::out_of_range("Batch Size: [" + std::to_string(batch) + "] Must Be Less Then The data Size: [" + std::to_string(m_data.size()/2) + "]");
                //--------------------------
            }// end if(batch > m_data.size()/2)
            //----------------------------
        }// end RLEnvironment(Dataset&& data_loader)
        //--------------------------------------------------------------
        std::tuple<torch::Tensor, bool> step(void){
            //----------------------------
            if(m_enable_batch){
                //----------------------------
                return internal_step(m_batch);
                //----------------------------
            }// end if(m_enable_batch)
            //----------------------------
            return internal_step();
            //----------------------------
        }// std::tuple<torch::Tensor, bool> step(void)
        //--------------------------------------------------------------
        std::tuple<torch::Tensor, COST_OUTPUT, double, bool> step(const Args&... args){
            //----------------------------
            if(m_enable_batch){
                //----------------------------
                return internal_step(m_batch, args...);
                //----------------------------
            }// end if(m_enable_batch)
            //----------------------------
            return internal_step(args...);
            //----------------------------
        }// std::tuple<torch::Tensor, COST_OUTPUT, double, bool> step(Args... args)
        //--------------------------------------------------------------
        torch::Tensor step(bool& done){
            //----------------------------
            if(m_enable_batch){
                //----------------------------
                return internal_step(done, m_batch);
                //----------------------------
            }// end if(m_enable_batch)
            //----------------------------
            return internal_step(done);
            //----------------------------
        }// torch::Tensor step(bool& done)
        //--------------------------------------------------------------
        std::tuple<torch::Tensor, COST_OUTPUT> step(double& epsilon, bool& done, const Args&... args){
            //----------------------------
            if(m_enable_batch){
                //----------------------------
                return internal_step(epsilon, done, m_batch, args...);
                //----------------------------
            }// end if(m_enable_batch)
            //----------------------------
            return internal_step(epsilon, done, args...);
            //----------------------------
        }// std::tuple<torch::Tensor, COST_OUTPUT, double, bool> step(Args... args)
        //--------------------------------------------------------------
        std::tuple<torch::Tensor, COST_OUTPUT> step(double& epsilon, bool& done, const size_t& batch, const Args&... args){
            //----------------------------
            return internal_step(epsilon, done, batch, args...);
            //----------------------------
        }// std::tuple<torch::Tensor, COST_OUTPUT, double, bool> step(Args... args)
        //--------------------------------------------------------------
        torch::Tensor step(bool& done, const size_t& batch){
            //----------------------------
            return internal_step(done, batch);
            //----------------------------
        }// std::tuple<torch::Tensor, COST_OUTPUT, double, bool> step(Args... args)
        //--------------------------------------------------------------
        std::tuple<torch::Tensor, double> get_first(void){
            //----------------------------
            if(m_enable_batch){
                //----------------------------
                return get_first_internal(m_batch);
                //----------------------------
            }// end if(m_enable_batch)
            //----------------------------
            return get_first_internal();
            //----------------------------
        }// end torch::Tensor get_first(void)
        //--------------------------------------------------------------
        torch::Tensor get_first(double& epsilon){
            //----------------------------
            if(m_enable_batch){
                //----------------------------
                return get_first_internal(epsilon, m_batch);
                //----------------------------
            }// end if(m_enable_batch)
            //----------------------------
            return get_first_internal(epsilon);
            //----------------------------
        }// end torch::Tensor get_first(void)
        //--------------------------------------------------------------
        torch::Tensor get_first(double& epsilon, const size_t& batch){
            //----------------------------
            return get_first_internal(epsilon, batch);
            //----------------------------
        }// end torch::Tensor get_first(void)
        //--------------------------------------------------------------
        void reset(void){
            //--------------------------
            reset_iterator();
            //--------------------------
        }// end void reset(void)
        //--------------------------------------------------------------
    protected:
        //--------------------------------------------------------------
        std::tuple<torch::Tensor, bool> internal_step(void){
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
                //--------------------------
                ++m_data_iter;
                //--------------------------
                return {input, false};
                //--------------------------
            }// end if (m_data_iter == m_data.begin())
            //--------------------------
             if(m_data_iter == m_data.end()-1){
                //--------------------------
                return {*m_data_iter, true};
                //--------------------------
            }// if(m_data_iter == m_data.end())
            //--------------------------
            if(m_data_iter != m_data.end()-1){
                //--------------------------
                ++m_data_iter;
                //--------------------------
            }// if(m_data_iter == m_data.end())
            //--------------------------
            return {*m_data_iter, false};
            //--------------------------
        }// end std::tuple<torch::Tensor, bool> internal_step(void)
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
        //--------------------------------------------------------------
        std::tuple<torch::Tensor, COST_OUTPUT> internal_step(double& epsilon, bool& done, const Args&... args){
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
            auto _reward = m_CostFunction(args...);
            //--------------------------
             if(m_data_iter == m_data.end()-1){
                //--------------------------
                epsilon = calculate_epsilon();
                done = true;
                //--------------------------
                return {*m_data_iter, _reward};
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
            return {*m_data_iter, _reward};
            //--------------------------
        }// end std::tuple<torch::Tensor, COST_OUTPUT> internal_step(double& epsilon, bool& done, Args... args)
        //--------------------------------------------------------------
        torch::Tensor internal_step(bool& done){
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
                done = false;
                //--------------------------
                ++m_data_iter;
                //--------------------------
                return input;
                //--------------------------
            }// end if (m_data_iter == m_data.begin())
            //--------------------------
             if(m_data_iter == m_data.end()-1){
                //--------------------------
                done = true;
                //--------------------------
                return *m_data_iter;
                //--------------------------
            }// if(m_data_iter == m_data.end())
            //--------------------------
            if(m_data_iter != m_data.end()-1){
                //--------------------------
                ++m_data_iter;
                //--------------------------
            }// if(m_data_iter == m_data.end())
            //--------------------------
            done = false;
            //--------------------------
            return *m_data_iter;
            //--------------------------
        }// end torch::Tensor internal_step(bool& done)
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
            auto _reward = m_CostFunction(args...);
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
                return {_data, _reward, calculate_epsilon(), true};
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
            return {_data, _reward, calculate_epsilon(), false};
            //--------------------------
        }// end std::tuple<torch::Tensor, COST_OUTPUT, double, bool> internal_step(const size_t& batch, Args... args)
        //--------------------------------------------------------------
        std::tuple<torch::Tensor, bool> internal_step(const size_t& batch){
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
                return {_data, false};
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
                return {_data, true};
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
            return {_data, false};
            //--------------------------
        }// end std::tuple<torch::Tensor, bool> internal_step(const size_t& batch)
        //--------------------------------------------------------------
        std::tuple<torch::Tensor, COST_OUTPUT> internal_step(double& epsilon, bool& done, const size_t& batch, const Args&... args){
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
            auto _reward = m_CostFunction(args...);
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
                return {_data, _reward};
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
            return {_data, _reward};
            //--------------------------
        }// end std::tuple<torch::Tensor, COST_OUTPUT> internal_step(double& epsilon, bool& done, const size_t& batch, Args... args)
        //--------------------------------------------------------------
        torch::Tensor internal_step(bool& done, const size_t& batch){
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
                return _data;
                //--------------------------
            }// end if (m_data_iter == m_data.begin())
            //--------------------------------------------------------------
            if(std::next(m_data_iter, batch) == m_data.end()-1){
                //--------------------------
                _data = *m_data_iter;
                //--------------------------
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
                return _data;
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
            done = false;
            //--------------------------
            return _data;
            //--------------------------
        }// end torch::Tensor internal_step(bool& done, const size_t& batch)
        //--------------------------------------------------------------
        std::tuple<torch::Tensor, double> get_first_internal(void){
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
        //--------------------------
        torch::Tensor get_first_internal(double& epsilon){
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
        torch::Tensor get_first_internal(double& epsilon, const size_t& batch){
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
        void reset_iterator(void){
            //--------------------------
            m_data_iter = m_data.begin();
            //--------------------------
        }// end void rest_iterator(void)
        //--------------------------------------------------------------
        constexpr double calculate_epsilon(void){
            //--------------------------
            return m_egreedy_final + (m_egreedy - m_egreedy_final) * std::exp(-1. * std::distance(m_data.begin(), m_data_iter) / m_egreedy_decay );
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
        size_t m_batch;
    //--------------------------------------------------------------
};// end class RLEnvironment
//--------------------------------------------------------------