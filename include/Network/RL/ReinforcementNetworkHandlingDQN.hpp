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
        ReinforcementNetworkHandlingDQN(void) = delete;
        //--------------------------------------------------------------
        ReinforcementNetworkHandlingDQN(    Network&& model,
                                            Network&& target_model,
                                            const size_t& update_frequency,
                                            std::function<torch::Tensor(Args&...)> actions) :   ReinforcementNetworkHandling<Network, Args...>(std::move(model), std::move(actions)),
                                                                                                m_model(this->get_model()),
                                                                                                m_target_model(std::move(target_model)),
                                                                                                m_update_frequency(update_frequency),
                                                                                                m_update_target_counter(0),
                                                                                                m_clamp(false),
                                                                                                m_double_mode(false){
            //--------------------------
        }// end ReinforcementNetworkHandling(Network& model, torch::Device& device)
        //--------------------------------------------------------------
        ReinforcementNetworkHandlingDQN(    Network&& model,
                                            Network&& target_model,
                                            const size_t& update_frequency,
                                            const bool& clamp,
                                            std::function<torch::Tensor(Args&...)> actions) :   ReinforcementNetworkHandling<Network, Args...>(std::move(model), std::move(actions)),
                                                                                                m_model(this->get_model()),
                                                                                                m_target_model(std::move(target_model)),
                                                                                                m_update_frequency(update_frequency),
                                                                                                m_update_target_counter(0),
                                                                                                m_clamp(clamp),
                                                                                                m_double_mode(false){
            //--------------------------
        }// end ReinforcementNetworkHandling(Network& model, torch::Device& device)
        //--------------------------------------------------------------
        ReinforcementNetworkHandlingDQN(    Network&& model,
                                            Network&& target_model,
                                            const size_t& update_frequency,
                                            const bool& clamp,
                                            const bool& double_mode,
                                            std::function<torch::Tensor(Args&...)> actions) :   ReinforcementNetworkHandling<Network, Args...>(std::move(model), std::move(actions)),
                                                                                                m_model(this->get_model()),
                                                                                                m_target_model(std::move(target_model)),
                                                                                                m_update_frequency(update_frequency),
                                                                                                m_update_target_counter(0),
                                                                                                m_clamp(clamp),
                                                                                                m_double_mode(double_mode){
            //--------------------------
        }// end ReinforcementNetworkHandling(Network& model, torch::Device& device)
        //--------------------------------------------------------------
        virtual void agent( const torch::Tensor& input, 
                            const torch::Tensor& next_input, 
                            torch::optim::Optimizer& optimizer, 
                            const torch::Tensor& rewards, 
                            const bool& done, 
                            const double& gamma = 0.5) override {
            //--------------------------
            agent_optimizer(input, next_input, optimizer, rewards, done, gamma);
            //--------------------------
        }// end void agent
        //--------------------------------------------------------------
        inline void set_clamp(const bool& clamp = true){
            //--------------------------
            m_clamp = clamp;
            //--------------------------
        }// end void set_clamp(const bool& clamp)
        //--------------------------------------------------------------
        inline void set_mode(const bool& double_mode = true){
            //--------------------------
            m_double_mode = double_mode;
            //--------------------------
        }// end void set_mode(const bool& double_mode)
        //--------------------------------------------------------------
        inline void set_update_frequency(const size_t& update_frequency = 100ul){
            //--------------------------
            m_update_frequency = update_frequency;
            //--------------------------
        }// end void set_update_frequency(const size_t& update_frequency)
    //--------------------------------------------------------------
    protected:
        //--------------------------------------------------------------
        virtual void agent_optimizer(   const torch::Tensor& input, 
                                        const torch::Tensor& next_input, 
                                        torch::optim::Optimizer& optimizer, 
                                        const torch::Tensor& rewards, 
                                        const bool& done, 
                                        const double& gamma) override {
            //--------------------------
            m_model.train(true);
            //--------------------------
            auto _target_value = dqn_agent(next_input, rewards, done, gamma);
            //--------------------------
            optimizer.zero_grad();
            //--------------------------
            auto _predicted_value = m_model.forward(input);
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
            //-------------------------- 
            if(m_clamp){
                //--------------------------
                for(auto& param : m_model.parameters()){
                    //--------------------------
                    param.grad().data().clamp(-1,1);
                    //--------------------------
                }// end for(auto& param : m_model.parameters())
                //--------------------------
            }// end if(m_clamp)
            //--------------------------
            optimizer.step();
            //-------------------------- 
            if(++m_update_target_counter % m_update_frequency == 0){
                //--------------------------
                torch::NoGradGuard no_grad;
                //--------------------------
                m_target_model.parameters() = m_model.parameters();
                //--------------------------
            }// end if(m_update_target_counter % update_frequency == 0)
            //--------------------------
            if(m_update_target_counter == SIZE_MAX){
                //--------------------------
                m_update_target_counter = 0;
                //--------------------------
            }// end if(m_update_target_counter == SIZE_MAX)
            //--------------------------
        }// end void agent(const torch::Tensor& input, const torch::Tensor& next_input, const torch::Tensor& action, const T& rewards, const bool& done)
        //--------------------------------------------------------------
    private:
        //--------------------------
        Network m_model, m_target_model; 
        //--------------------------
        size_t m_update_frequency, m_update_target_counter;
        //--------------------------
        bool m_clamp, m_double_mode;
        //--------------------------
        torch::Tensor dqn_agent(const torch::Tensor& next_input, 
                                const torch::Tensor& rewards, 
                                const bool& done,
                                const double& gamma){
            //--------------------------
            if(m_double_mode){
                //--------------------------
                torch::Tensor _state_value;
                std::tie(std::ignore, _state_value) = torch::max(m_model.forward(next_input).detach(), 1);
                //--------------------------
                return rewards + (1 - done) * gamma * m_target_model.forward(next_input).detach().gather(1, _state_value.unsqueeze(1));
                //--------------------------
            }// end if(m_double_mode)
            //--------------------------
            // torch::Tensor _state_value;
            // std::tie(_state_value, std::ignore) = torch::max(m_target_model.forward(next_input).detach(), 1);
            // //--------------------------
            // return rewards + (1 - done) * gamma * m_target_model.forward(next_input).detach().gather(1, _state_value.unsqueeze(1)).squeeze(1);
            //--------------------------
            return rewards + (1 - done) * gamma * m_target_model.forward(next_input).detach();
            //--------------------------
        }// end torch::Tensor dqn_agent(const torch::Tensor& input, const torch::Tensor& next_input)
    //--------------------------------------------------------------
};// end class ReinforcementNetworkHandling : protected NetworkHandling
//--------------------------------------------------------------