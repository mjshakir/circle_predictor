#pragma once

//--------------------------------------------------------------
// User defind library
//--------------------------------------------------------------
#include "Network/RL/ReinforcementNetworkHandling.hpp"
//--------------------------------------------------------------
template<typename Network, typename SCHEDULER, typename... Args>
class ReinforcementNetworkHandlingDQN : public ReinforcementNetworkHandling<Network, SCHEDULER, Args...>{
    //--------------------------------------------------------------
    public:
        //--------------------------------------------------------------
        ReinforcementNetworkHandlingDQN(void) = delete;
        //--------------------------------------------------------------
        virtual ~ReinforcementNetworkHandlingDQN(void) = default;
        //--------------------------------------------------------------
        ReinforcementNetworkHandlingDQN(    Network&& model,
                                            Network&& target_model,
                                            SCHEDULER&& scheduler,
                                            const size_t& update_frequency,
                                            std::function<torch::Tensor(Args&...)>&& actions) : ReinforcementNetworkHandling<Network, SCHEDULER, Args...>(std::move(model), std::move(scheduler), std::move(actions)),
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
                                            SCHEDULER&& scheduler,
                                            const size_t& update_frequency,
                                            const bool& clamp,
                                            std::function<torch::Tensor(Args&...)>&& actions) : ReinforcementNetworkHandling<Network, SCHEDULER, Args...>(std::move(model), std::move(scheduler), std::move(actions)),
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
                                            SCHEDULER&& scheduler,
                                            const size_t& update_frequency,
                                            const bool& clamp,
                                            const bool& double_mode,
                                            std::function<torch::Tensor(Args&...)>&& actions) : ReinforcementNetworkHandling<Network, SCHEDULER, Args...>(std::move(model), std::move(scheduler), std::move(actions)),
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
                            const double& gamma = 0.5) {
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
                                        const double& gamma) {
            //--------------------------
            this->get_model().train(true);
            //--------------------------
            auto _target_value = dqn_agent(next_input, rewards, done, gamma);
            //--------------------------
            optimizer.zero_grad();
            //--------------------------
            auto _predicted_value = this->get_model().forward(input);
            //--------------------------
            // std::cout   << "_target_value: " << _target_value.sizes()
            //             << " _predicted_value: " << _predicted_value.sizes() << std::endl;
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
                for(auto& param : this->get_model().parameters()){
                    //--------------------------
                    param.grad().data().clamp(-1,1);
                    //--------------------------
                }// end for(auto& param : this->get_model().parameters())
                //--------------------------
            }// end if(m_clamp)
            //--------------------------
            this->get_scheduler().step();
            //-------------------------- 
            if(++m_update_target_counter % m_update_frequency == 0){
                //--------------------------
                torch::NoGradGuard no_grad;
                //--------------------------
                // m_target_model.parameters() = this->get_model().parameters();
                //--------------------------
                for (size_t i = 0; i < this->get_model().parameters().size(); ++i) {
                    //--------------------------
                    m_target_model.parameters().at(i).data().copy_(this->get_model().parameters().at(i).data());
                    //--------------------------
                }// end for (size_t i = 0; i < this->get_model().parameters().size(); ++i)
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
        Network m_target_model; 
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
            torch::Tensor _state_value;
            //--------------------------
            if(m_double_mode){
                //--------------------------
                auto _state_indexes = this->get_model().forward(next_input).detach();
                //--------------------------
                if (_state_indexes.dim() > 2){
                    //--------------------------
                    std::tie(std::ignore, _state_value) = torch::max(_state_indexes, 2);
                    //--------------------------
                    return rewards.unsqueeze(1).unsqueeze(2) + (1 - done) * gamma * 
                                m_target_model.forward(next_input).detach().gather(2, _state_value.unsqueeze(2)).expand_as(_state_indexes);
                    //--------------------------
                }// end if (_state_indexes.dim() > 2)
                //--------------------------
                std::tie(std::ignore, _state_value) = torch::max(_state_indexes, 1);
                //--------------------------
                return rewards + (1 - done) * gamma * m_target_model.forward(next_input).detach().gather(1, _state_value.unsqueeze(1));
                //--------------------------
            }// end if(m_double_mode)
            //--------------------------
            auto _state_indexes = m_target_model.forward(next_input).detach();
            //--------------------------
            if (_state_indexes.dim() > 2){
                //--------------------------
                std::tie(_state_value, std::ignore) = torch::max(_state_indexes, 1);
                //--------------------------
                return rewards.unsqueeze(1).unsqueeze(2) + (1 - done) * gamma * 
                            m_target_model.forward(next_input).detach().gather(2, _state_value.unsqueeze(2).to(torch::kDouble)).expand_as(_state_indexes);
                //--------------------------
            }// end if (_state_indexes.dim() > 2)
            //--------------------------
            std::tie(_state_value, std::ignore) = torch::max(_state_indexes, 1);
            //--------------------------
            return rewards + (1 - done) * gamma * m_target_model.forward(next_input).detach().gather(1, _state_value.unsqueeze(1).to(torch::kDouble));
            //--------------------------
            // return rewards + (1 - done) * gamma * m_target_model.forward(next_input).detach();
            //--------------------------
        }// end torch::Tensor dqn_agent(const torch::Tensor& input, const torch::Tensor& next_input)
    //--------------------------------------------------------------
};// end class ReinforcementNetworkHandling : protected NetworkHandling
//--------------------------------------------------------------