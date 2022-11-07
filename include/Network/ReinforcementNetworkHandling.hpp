#pragma once

//--------------------------------------------------------------
// LibTorch library
//--------------------------------------------------------------
#include <torch/torch.h>
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <random>
//--------------------------------------------------------------
template<typename Network, typename... Args>
class ReinforcementNetworkHandling{
    //--------------------------------------------------------------
    public:
        //--------------------------------------------------------------
        ReinforcementNetworkHandling() = delete;
        //--------------------------
        ReinforcementNetworkHandling(   Network& model, 
                                        const torch::Device& device,
                                        std::function<torch::Tensor(Args&...)> actions):    m_model(model), 
                                                                                            m_device(device),
                                                                                            m_random_action(std::move(actions)){
            //--------------------------
        }// end ReinforcementNetworkHandling(Network& model, torch::Device& device)
        //--------------------------
        ReinforcementNetworkHandling(   Network&& model, 
                                        const torch::Device& device, 
                                        std::function<torch::Tensor(Args&...)> actions) :   m_model(std::move(model)), 
                                                                                            m_device(device),
                                                                                            m_random_action(std::move(actions)){
            //--------------------------
        }// end ReinforcementNetworkHandling(Network& model, torch::Device& device)
        //--------------------------
        torch::Tensor action(const torch::Tensor& input, const double& epsilon, const Args... args){
            return select_action(input, epsilon, args...);
        }// end torch::Tensor select(const torch::Tensor& input, const double& epsilon, const Args... args)
        //--------------------------
        void agent( const torch::Tensor& input, 
                    const torch::Tensor& next_input, 
                    torch::optim::Optimizer& optimizer, 
                    const torch::Tensor& rewards, 
                    const bool& done, 
                    const double& gamma = 0.5){
            //--------------------------
            agent_optimizer(input, next_input, optimizer, rewards, done, gamma);
            //--------------------------
        }// end void agent
    //--------------------------------------------------------------
    protected:
        //--------------------------------------------------------------
        template<typename T>
        torch::Tensor select_action(const torch::Tensor& input, const double& epsilon, const Args... args){
            //--------------------------
            std::random_device rd;  // Will be used to obtain a seed for the random number engine
            std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd(
            std::uniform_real_distribution<double> uniform_angle(0., 1.);
            std::default_random_engine re;
            //--------------------------
            auto _random_egreedy = uniform_angle(re);
            //--------------------------
            torch::NoGradGuard no_grad;
            m_model.eval();
            //--------------------------
            if(_random_egreedy > epsilon){
                //--------------------------
                return m_model(input).forward();
                //--------------------------
            } // end if(_random_egreedy > epsilon)     
            //--------------------------
            return m_random_action(args...);
            //--------------------------
        }// end torch::Tensor select_action(const torch::Tensor& input, const T& epsilon)
        //--------------------------
        // template<typename AGENT_OUTPUT, typename... Args>
        // void agent(std::function<AGENT_OUTPUT(Args&...)> agent_function); 
        //--------------------------
        void agent_optimizer(   const torch::Tensor& input, 
                                const torch::Tensor& next_input, 
                                torch::optim::Optimizer& optimizer, 
                                const torch::Tensor& rewards, 
                                const bool& done, 
                                const double& gamma){
            //--------------------------
            torch::Tensor _target_value;
            //--------------------------
            if(done){
                //--------------------------
                _target_value = rewards;
                //--------------------------
            }// end if(done)
            else{
                //--------------------------
                auto _state_value = m_model(next_input).detach();
                _target_value = rewards + gamma * _state_value;
                //--------------------------
            }// end else
            //--------------------------
            optimizer.zero_grad();
            //--------------------------
            auto _predicted_value = m_model.forward(input);
            //--------------------------
            torch::Tensor loss = torch::mse_loss(_predicted_value, _target_value);
            //--------------------------
            loss.backward({},c10::optional<bool>(true), false);
            optimizer.step();
            //-------------------------- 
        }// end void agent(const torch::Tensor& input, const torch::Tensor& next_input, const torch::Tensor& action, const T& rewards, const bool& done)
        //--------------------------------------------------------------
    private:
        //--------------------------
        Network m_model; 
        torch::Device m_device;
        //--------------------------
        std::function<torch::Tensor(Args&...)> m_random_action;
    //--------------------------------------------------------------
};// end class ReinforcementNetworkHandling : protected NetworkHandling
//--------------------------------------------------------------