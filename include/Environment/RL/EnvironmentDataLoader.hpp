#pragma once
//--------------------------------------------------------------
// LibTorch library
//--------------------------------------------------------------
#include <torch/torch.h>
//--------------------------------------------------------------
namespace RL {
    //--------------------------------------------------------------
    namespace Environment {
        //--------------------------------------------------------------
        template<typename T>
        //--------------------------------------------------------------
        class EnvironmentDataLoader{
            //--------------------------------------------------------------
            public:
                //--------------------------------------------------------------
                EnvironmentDataLoader(void) = delete;
                //--------------------------------------------------------------
                EnvironmentDataLoader(std::vector<T>&& data, const size_t& batch = 1) : m_data(std::move(data)),
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
                }// end EnvironmentDataLoader(std::vector<T>&& data, const size_t& batch = 1)
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
                torch::Tensor step(bool& done, const size_t& batch){
                    //----------------------------
                    return internal_step(done, batch);
                    //----------------------------
                }// std::tuple<torch::Tensor, COST_OUTPUT, double, bool> step(Args... args)
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
                void reset_iterator(void){
                    //--------------------------
                    m_data_iter = m_data.begin();
                    //--------------------------
                }// end void rest_iterator(void)
                //--------------------------------------------------------------
            private:
                //--------------------------------------------------------------
                std::vector<T> m_data;
                typename std::vector<T>::iterator m_data_iter;
                //--------------------------
                bool m_enable_batch;
                //--------------------------
                size_t m_batch;
            //--------------------------------------------------------------
        };// end class EnvironmentDataLoader
        //--------------------------------------------------------------
    }// end namespace Environment
    //--------------------------------------------------------------
}// end namespace RL
//--------------------------------------------------------------