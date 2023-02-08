#pragma once

//--------------------------------------------------------------
// User defind library
//--------------------------------------------------------------
#include "Network/RL/ReinforcementNetworkHandling.hpp"
//--------------------------------------------------------------
template<typename Network, typename... Args>
class ReinforcementNetworkHandlingDQN : public ReinforcementNetworkHandling<Network, Args...>{
    //--------------------------------------------------------------
    public:
        //--------------------------------------------------------------
        ReinforcementNetworkHandlingDQN(    Network&& model,
                                            Network&& target_model, 
                                            const torch::Device& device, 
                                            std::function<torch::Tensor(Args&...)> actions) :   ReinforcementNetworkHandling<Network, Args...>(std::move(model), device, std::move(actions)),
                                                                                                m_model(this->get_model()),
                                                                                                m_target_model(std::move(target_model)),
                                                                                                m_update_target_counter(0){
            //--------------------------
            // m_model = get_model();
            //--------------------------
        }// end ReinforcementNetworkHandling(Network& model, torch::Device& device)
        //--------------------------
        virtual void agent( const torch::Tensor& input, 
                    const torch::Tensor& next_input, 
                    torch::optim::Optimizer& optimizer, 
                    const torch::Tensor& rewards, 
                    const bool& done, 
                    const double& gamma = 0.5,
                    const size_t& update_frequency = 100) override {
            //--------------------------
            agent_optimizer(input, next_input, optimizer, rewards, done, gamma, update_frequency);
            //--------------------------
        }// end void agent
        //--------------------------
    //--------------------------------------------------------------
    protected:
        //--------------------------------------------------------------
        virtual void agent_optimizer(   const torch::Tensor& input, 
                                const torch::Tensor& next_input, 
                                torch::optim::Optimizer& optimizer, 
                                const torch::Tensor& rewards, 
                                const bool& done, 
                                const double& gamma,
                                const size_t& update_frequency) override {
            //--------------------------
            torch::Tensor _target_value, _input = input, _next_input = next_input;
            //--------------------------
            m_model.train(true);
            //--------------------------
            auto _state_value = m_target_model.forward(_next_input).detach();
            _target_value = rewards + (1 - done) * gamma * _state_value;
            //--------------------------
            optimizer.zero_grad();
            //--------------------------
            auto _predicted_value = m_model.forward(_input);
            //--------------------------
            torch::Tensor loss = torch::mse_loss(_predicted_value, _target_value);
            //--------------------------
            if(torch::isnan(loss).any().item<bool>()){
                //--------------------------
                throw std::overflow_error("\x1b[31m" "\033[1m" "NaN Is Detected In The Agent" "\033[m" "\x1b[0m");
                //--------------------------
            }//end if( torch::isnan(_predicted_value) || torch::isnan(_target_value))
            //--------------------------
            loss.backward({},c10::optional<bool>(true), false);
            optimizer.step();
            //-------------------------- 
            ++m_update_target_counter;
            //--------------------------
            if(m_update_target_counter % update_frequency == 0){
                //--------------------------
                // std::stringstream stream;
                //--------------------------
                // std::ostringstream stream;
                //--------------------------
                // torch::save(m_model, stream);
                // torch::load(m_target_model, stream);
                //--------------------------
                // torch::serialize::OutputArchive stream = "stream.pt";
                // torch::save(m_model, "stream.pt");
                // m_model.save(std::move(stream));
                // torch::load(m_target_model, "stream.pt");
                //--------------------------
                torch::NoGradGuard no_grad;
                //--------------------------
                m_target_model.parameters() = m_model.parameters();
                //--------------------------
            }// end if(m_update_target_counter % update_frequency == 0)
            //--------------------------
        }// end void agent(const torch::Tensor& input, const torch::Tensor& next_input, const torch::Tensor& action, const T& rewards, const bool& done)
        //--------------------------------------------------------------
    private:
        //--------------------------
        Network &m_model, m_target_model; 
        //--------------------------
        size_t m_update_target_counter;
        //--------------------------
    //--------------------------------------------------------------
};// end class ReinforcementNetworkHandling : protected NetworkHandling
//--------------------------------------------------------------