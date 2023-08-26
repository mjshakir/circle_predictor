#pragma once
//--------------------------------------------------------------
// User Defined library
//--------------------------------------------------------------
#include "Environment/RL/RLEnvironmentTest.hpp"
//--------------------------------------------------------------
namespace RL {
    //--------------------------------------------------------------
    namespace Environment {
        //--------------------------------------------------------------
        template<typename T>
        //--------------------------------------------------------------
        class RLEnvironmentTestLoader : public RLEnvironmentTest<T> {
            //--------------------------------------------------------------
            public:
                //--------------------------------------------------------------
                RLEnvironmentTestLoader(void) = delete;
                //--------------------------------------------------------------
                RLEnvironmentTestLoader(const std::span<T>& data, const size_t& batch = 2UL) :  RLEnvironmentTest<T>(data),
                                                                                                m_batch(batch){
                    //----------------------------
                    if(batch >= this->get_data().size()/2){
                        //--------------------------
                        throw std::out_of_range("Batch Size: [" + std::to_string(batch) + "] Must Be Less Then The data Size: [" + std::to_string(this->get_data().size()/2) + "]");
                        //--------------------------
                    }// end if(batch > this->get_data().size()/2)
                    //----------------------------
                }// end RLEnvironmentTestLoader(std::span<T>&& data, const size_t& batch = 1)
                //--------------------------------------------------------------
                RLEnvironmentTestLoader(const RLEnvironmentTestLoader&)            = default;
                RLEnvironmentTestLoader& operator=(const RLEnvironmentTestLoader&) = default;
                //----------------------------
                RLEnvironmentTestLoader(RLEnvironmentTestLoader&&)                 = default;
                RLEnvironmentTestLoader& operator=(RLEnvironmentTestLoader&&)      = default;
                //--------------------------------------------------------------
                virtual std::tuple<torch::Tensor, bool> step(void) override {
                    //----------------------------
                    return internal_step(m_batch);
                    //----------------------------
                }// std::tuple<torch::Tensor, bool> step(void)
                //--------------------------------------------------------------
                virtual torch::Tensor step(bool& done) override {
                    //----------------------------
                    return internal_step(done, m_batch);
                    //----------------------------
                }// torch::Tensor step(bool& done)
                //--------------------------------------------------------------
                torch::Tensor step(bool& done, const size_t& batch){
                    //----------------------------
                    return internal_step(done, batch);
                    //----------------------------
                }// std::tuple<torch::Tensor, COST_OUTPUT, double, bool> step(Args... args)
                //--------------------------------------------------------------
            protected:
                //--------------------------------------------------------------
                std::tuple<torch::Tensor, bool> internal_step(const size_t& batch){
                    //--------------------------------------------------------------
                    if (this->get_iterator() == this->get_data().end() or std::next(this->get_iterator(), batch) == this->get_data().end()){
                        //--------------------------
                        throw std::out_of_range("End Of The Data Iterator");
                        //--------------------------
                    }// end if (this->get_iterator() == this->get_data().end() or std::next(this->get_iterator(), batch) == this->get_data().end())
                    //--------------------------------------------------------------
                    std::vector<torch::Tensor> _data;
                    _data.reserve(batch);
                    //--------------------------
                    auto _data_end = std::next(this->get_iterator(), batch);
                    //--------------------------------------------------------------
                    if (this->get_iterator() == this->get_data().begin() and _data_end != this->get_data().end()-1){
                        //--------------------------
                        for(; this->get_iterator() != _data_end; ++this->get_iterator()){
                            //--------------------------
                            _data.push_back(*this->get_iterator());
                            //--------------------------
                        }// for(; _data_iter != _data_end; ++_data_iter)
                        //--------------------------
                        --this->get_iterator();
                        //--------------------------
                        return {torch::cat(_data, 0), false};
                        //--------------------------
                    }// end if (this->get_iterator() == this->get_data().begin())
                    //--------------------------------------------------------------
                    if(_data_end == this->get_data().end()-1){
                        //--------------------------
                        for(; this->get_iterator() != _data_end; ++this->get_iterator()) {
                            //--------------------------
                            _data.push_back(*this->get_iterator());
                            //--------------------------
                        }// end for(; this->get_iterator() != _data_end; ++this->get_iterator())
                        //--------------------------
                        return {torch::cat(_data, 0), true};
                        //--------------------------
                    }// if(this->get_iterator() == this->get_data().end())
                    //--------------------------------------------------------------
                    for(; this->get_iterator() != _data_end; ++this->get_iterator()) {
                        //--------------------------
                        _data.push_back(*this->get_iterator());
                        //--------------------------
                    }// end for(; this->get_iterator() != _data_end; ++this->get_iterator())
                    //--------------------------
                    return {torch::cat(_data, 0), false};
                    //--------------------------
                }// end std::tuple<torch::Tensor, bool> internal_step(const size_t& batch)
                //--------------------------------------------------------------
                torch::Tensor internal_step(bool& done, const size_t& batch){
                    //--------------------------------------------------------------
                    if (this->get_iterator() == this->get_data().end() or std::next(this->get_iterator(), batch) == this->get_data().end()){
                        //--------------------------
                        throw std::out_of_range("End Of The Data Iterator");
                        //--------------------------
                    }// end if (this->get_iterator() == this->get_data().end() or std::next(this->get_iterator(), batch) == this->get_data().end())
                    //--------------------------
                    std::vector<torch::Tensor> _data;
                    _data.reserve(batch);
                    //--------------------------
                    auto _data_end = std::next(this->get_iterator(), batch);
                    //--------------------------------------------------------------
                    if (this->get_iterator() == this->get_data().begin() and _data_end != this->get_data().end()-1){
                        //--------------------------
                        done = false;
                        //--------------------------
                        for(; this->get_iterator() != _data_end; ++this->get_iterator()){
                            //--------------------------
                            _data.push_back(*this->get_iterator());
                            //--------------------------
                        }// for(; _data_iter != _data_end; ++_data_iter)
                        //--------------------------
                        --this->get_iterator();
                        //--------------------------
                        return torch::cat(_data, 0);
                        //--------------------------
                    }// end if (this->get_iterator() == this->get_data().begin())
                    //--------------------------------------------------------------
                    if(_data_end == this->get_data().end()-1){
                        //--------------------------
                        done = true;
                        //--------------------------
                        for(; this->get_iterator() != _data_end; ++this->get_iterator()){
                            //--------------------------
                            _data.push_back(*this->get_iterator());
                            //--------------------------
                        }// for(; _data_iter != _data_end; ++_data_iter)
                        //--------------------------
                        return torch::cat(_data, 0);
                        //--------------------------
                    }// if(this->get_iterator() == this->get_data().end())
                    //--------------------------------------------------------------
                    for(; this->get_iterator() != _data_end; ++this->get_iterator()){
                        //--------------------------
                        _data.push_back(*this->get_iterator());
                        //--------------------------
                    }// for(; _data_iter != _data_end; ++_data_iter)
                    //--------------------------
                    done = false;
                    //--------------------------
                    return torch::cat(_data, 0);
                    //--------------------------
                }// end torch::Tensor internal_step(bool& done, const size_t& batch)
                //--------------------------------------------------------------
            private:
                //--------------------------------------------------------------
                const size_t m_batch;
            //--------------------------------------------------------------
        };// end class RLEnvironmentTestLoader
        //--------------------------------------------------------------
    }// end namespace Environment
    //--------------------------------------------------------------
}// end namespace RL
//--------------------------------------------------------------