#pragma once
//--------------------------------------------------------------
// LibTorch library
//--------------------------------------------------------------
#include <torch/torch.h>
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <span>
//--------------------------------------------------------------
namespace RL {
    //--------------------------------------------------------------
    namespace Environment {
        //--------------------------------------------------------------
        template<typename T>
        //--------------------------------------------------------------
        class RLEnvironmentTest{
            //--------------------------------------------------------------
            public:
                //--------------------------------------------------------------
                RLEnvironmentTest(void) = delete;
                //--------------------------------------------------------------
                RLEnvironmentTest(const std::span<T>& data) :   m_data(data),
                                                                m_data_iter (m_data.begin()){
                    //----------------------------
                }// end RLEnvironmentTest(std::span<T>&& data, const size_t& batch = 1)
                //--------------------------------------------------------------
                RLEnvironmentTest(const RLEnvironmentTest&)            = default;
                RLEnvironmentTest& operator=(const RLEnvironmentTest&) = default;
                //----------------------------
                RLEnvironmentTest(RLEnvironmentTest&&)                 = default;
                RLEnvironmentTest& operator=(RLEnvironmentTest&&)      = default;
                //--------------------------------------------------------------
                virtual std::tuple<torch::Tensor, bool> step(void){
                    //----------------------------
                    return internal_step();
                    //----------------------------
                }// std::tuple<torch::Tensor, bool> step(void)
                //--------------------------------------------------------------
                virtual torch::Tensor step(bool& done){
                    //----------------------------
                    return internal_step(done);
                    //----------------------------
                }// torch::Tensor step(bool& done)
                //--------------------------------------------------------------
                void reset(void){
                    //--------------------------
                    reset_iterator();
                    //--------------------------
                }// end void reset(void)
                //--------------------------------------------------------------
                /**
                 * @brief 
                 * 
                 */
                constexpr size_t size(void) const{
                    //--------------------------
                    return vector_size();
                    //--------------------------
                }// end constexpr size_t size(void) const
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
                    ++m_data_iter;
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
                void reset_iterator(void){
                    //--------------------------
                    m_data_iter = m_data.begin();
                    //--------------------------
                }// end void rest_iterator(void)
                //--------------------------------------------------------------
                std::span<T>& get_data(void){
                    //--------------------------
                    return m_data;
                    //--------------------------
                }// end std::span<T>& get_data(void)
                //--------------------------------------------------------------
                typename std::span<T>::iterator& get_iterator(void){
                    //--------------------------
                    return m_data_iter;
                    //--------------------------
                }// end typename std::span<T>::iterator& get_iterator(void)
                //--------------------------------------------------------------
                constexpr size_t vector_size(void) const{
                    //--------------------------
                    return m_data.size();
                    //--------------------------
                }// end typename std::span<T>::iterator& get_iterator(void)
                //--------------------------------------------------------------
            private:
                //--------------------------------------------------------------
                std::span<T> m_data;
                typename std::span<T>::iterator m_data_iter;
            //--------------------------------------------------------------
        };// end class RLEnvironmentTest
        //--------------------------------------------------------------
    }// end namespace Environment
    //--------------------------------------------------------------
}// end namespace RL
//--------------------------------------------------------------