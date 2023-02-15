//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Network/RL/ExperienceReplay.hpp"
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <random>
//--------------------------------------------------------------
ExperienceReplay::ExperienceReplay(const size_t& capacity) : m_capacity(capacity), m_position(0){
    //--------------------------
    m_memory.reserve(m_capacity);
    //--------------------------
}// end ExperienceReplay::ExperienceReplay(const size_t& capacity)
//--------------------------------------------------------------
void ExperienceReplay::push(const torch::Tensor& input, const torch::Tensor& next_input, const torch::Tensor& reward, const bool& done){
    //--------------------------
    return push_data(input, next_input, reward, done);
    //--------------------------
}// end void ExperienceReplay::push(const torch::Tensor& input, const torch::Tensor& next_input, const torch::Tensor& reward, const bool& done)
//--------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, bool> ExperienceReplay::sample(void){
    //--------------------------
    return sample_data();
    //--------------------------
}// end std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, bool> ExperienceReplay::sample_data(void)
//--------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ExperienceReplay::sample(bool& done){
    //--------------------------
    return sample_data(done);
    //--------------------------
}// end std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ExperienceReplay::sample(bool& done)
//--------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> ExperienceReplay::sample(const size_t& batch, bool& done){
    //--------------------------
    return sample_data(batch, done);
    //--------------------------
}// end std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> ExperienceReplay::sample(const size_t& batch, bool& done)
//--------------------------------------------------------------
size_t ExperienceReplay::size(void) const {
    //--------------------------
    return map_size();
    //--------------------------
}// end size_t ExperienceReplay::size(void) const
//--------------------------------------------------------------
void ExperienceReplay::push_data(const torch::Tensor& input, const torch::Tensor& next_input, const torch::Tensor& reward, const bool& done){
    //--------------------------
    m_memory.try_emplace(m_position, input, next_input, reward, done);
    //--------------------------
    if(m_position != LONG_MAX){
        //--------------------------
        m_position = ++m_position % m_capacity;
        //--------------------------
    }// end if(m_position != LONG_MAX)
    //--------------------------
}// end void ExperienceReplay::push_data(const torch::Tensor& input, const torch::Tensor& next_input, const torch::Tensor& reward, const bool& done)
//--------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, bool> ExperienceReplay::sample_data(void){
    //--------------------------
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> uniform_position(0, m_memory.size()-1);
    //--------------------------
    return std::next( m_memory.begin() , uniform_position(rng))->second;
    //--------------------------
}// end std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, bool> ExperienceReplay::sample_data(void)
//--------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ExperienceReplay::sample_data(bool& done){
    //--------------------------
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> uniform_position(0, m_memory.size()-1);
    //--------------------------
    auto it = std::next(m_memory.begin() , uniform_position(rng));
    //--------------------------
    auto [input, next_input, reward, _done] = it->second;
    //--------------------------
    done = _done;
    //--------------------------
    return {input, next_input, reward};
    //--------------------------
}// end std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ExperienceReplay::sample_data(bool& done)
//--------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> ExperienceReplay::sample_data(const size_t& batch, bool& done){
    //--------------------------
    if(batch > m_memory.size()){
        //--------------------------
        return {torch::tensor(0), torch::tensor(0), torch::tensor(0), torch::tensor(0)};
        //--------------------------
    }// end if(batch > m_memory.size())
    //--------------------------
    torch::Tensor t_input, t_next_input, t_reward, t_done = torch::zeros({static_cast<int64_t>(batch), 1}, torch::TensorOptions().dtype(torch::kBool));
    //--------------------------
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> uniform_position(0, m_memory.size()-1);
    //--------------------------
    for (size_t i{0}; i < batch; ++i){
        //--------------------------
        auto it = std::next(m_memory.begin() , uniform_position(rng));
        //--------------------------
        auto [input, next_input, reward, _done] = it->second;
        //--------------------------
        if(i == 0){
            //--------------------------
            t_input = input;
            //--------------------------
            t_next_input = next_input;
            //--------------------------
            t_reward = reward;
            //--------------------------
            t_done[i] = torch::tensor(_done);
            //--------------------------
            continue;
            //--------------------------
        }// end if(i == 0)
        //--------------------------
        if((static_cast<size_t>(t_input.size(0)) < batch)
            and (static_cast<size_t>(t_next_input.size(0)) < batch)
            and (static_cast<size_t>(t_reward.size(0)) < batch)
            and (static_cast<size_t>(t_done.size(0)) < batch)){
            //--------------------------
            break;
            //--------------------------
        }// end if((t_input.size(0) < batch) and (t_next_input.size(0) < batch) and (t_reward.size(0) < batch) and (t_done.size(0) < batch))
        //--------------------------
        if(static_cast<size_t>(t_input.size(0)) < batch){
            //--------------------------
            t_input = torch::cat({t_input, input});
            //--------------------------
        }// end if(t_input.size(0) < batch)
        //--------------------------
        if(static_cast<size_t>(t_next_input.size(0)) < batch){
            //--------------------------
            t_next_input = torch::cat({t_next_input, next_input});
            //--------------------------
        }// end if(t_next_input.size(0) < batch)
        //--------------------------
        if(static_cast<size_t>(t_reward.size(0)) < batch){
            //--------------------------
            t_reward = torch::cat({t_reward, reward});
            //--------------------------
        }// end if(t_reward.size(0) < batch)
        //--------------------------
        if(static_cast<size_t>(t_done.size(0)) < batch){
            //--------------------------
            t_done[i] =_done;
            //--------------------------
        }// end if(t_done.size(0) < batch)
        //--------------------------
    }// end for (size_t i = 0; i < batch; ++i)
    //--------------------------
    // (t_done.any().item().toBool() == true) ? done = true : done = false;
    //--------------------------
    done = t_done.any(true).item().toBool();
    //--------------------------
    if(done){
        std::cout << "done " << std::boolalpha <<  done << std::endl;
    }
    //--------------------------
    // std::cout << "t_done " << t_done << std::endl;
    //--------------------------
    if(static_cast<size_t>(t_input.size(0)) > batch){
        //--------------------------
        t_input = t_input.slice(0,0,batch);
        //--------------------------
    }//end if(t_input.size(0) > batch)
    // //--------------------------
    if(static_cast<size_t>(t_next_input.size(0)) > batch){
        //--------------------------
        t_next_input = t_next_input.slice(0,0,batch);
        //--------------------------
    }// end if(t_next_input.size(0) < batch)
    //--------------------------
    if(static_cast<size_t>(t_reward.size(0)) > batch){
        //--------------------------
        t_reward = t_reward.slice(0,0,batch);
        //--------------------------
    }// end if(t_reward.size(0) < batch)
    //--------------------------
    if(static_cast<size_t>(t_done.size(0)) > batch){
        //--------------------------
        t_done = t_done.slice(0,0,batch);
        //--------------------------
    }// end if(t_done.size(0) < batch)
    //--------------------------
    // std::cout   << "t_input: [" << t_input.sizes() << "]"
    //             << " t_next_input: [" << t_next_input.sizes() << "]"
    //             << " t_reward: [" << t_reward.sizes() << "]"
    //             << " t_done: [" << t_done.sizes() << "]" << std::endl;
    // //--------------------------
    return {t_input, t_next_input, t_reward, t_done};
    //--------------------------
}// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> ExperienceReplay::sample_data(const size_t& batch, bool& done)
//--------------------------------------------------------------
size_t ExperienceReplay::map_size(void) const{
    //--------------------------
    return m_memory.size();
    //--------------------------
}// end size_t ExperienceReplay::map_size(void) const
//--------------------------------------------------------------