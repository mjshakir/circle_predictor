#pragma once

//--------------------------------------------------------------
// LibTorch library
//--------------------------------------------------------------
#include <torch/torch.h>
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <random>
#include <math.h>
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
                                        std::function<torch::Tensor(Args...)> actions):    m_model(model), 
                                                                                            m_device(device),
                                                                                            m_random_action(std::move(actions)){
            //--------------------------
        }// end ReinforcementNetworkHandling(Network& model, torch::Device& device)
        //--------------------------
        ReinforcementNetworkHandling(   Network&& model, 
                                        const torch::Device& device, 
                                        std::function<torch::Tensor(Args...)> actions) :   m_model(std::move(model)), 
                                                                                            m_device(device),
                                                                                            m_random_action(std::move(actions)){
            //--------------------------
        }// end ReinforcementNetworkHandling(Network& model, torch::Device& device)
        //--------------------------
        torch::Tensor action(torch::Tensor& input, const double& epsilon, Args... args){
            //--------------------------
            return select_action(input, epsilon, args...);
            //--------------------------
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
        //--------------------------
        void agent( const torch::Tensor& input, 
                    torch::optim::Optimizer& optimizer, 
                    const torch::Tensor& rewards, 
                    const bool& done, 
                    const double& gamma = 0.5){
            //--------------------------
            agent_optimizer(input, optimizer, rewards, done, gamma);
            //--------------------------
        }// end void agent
        //--------------------------
        torch::Tensor test(const torch::Tensor& input){
            //--------------------------
            return network_test(input);
            //--------------------------
        }// end torch::Tensor test(const torch::Tensor& input)
    //--------------------------------------------------------------
    protected:
        //--------------------------------------------------------------
        torch::Tensor select_action(torch::Tensor& input, const double& epsilon, Args... args){
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
                // std::cout << "select_action greedy" << std::endl;
                //--------------------------
                return m_model.forward(input);
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
            torch::Tensor _target_value, _input = input, _next_input = next_input;
            //--------------------------
            m_model.train(true);
            //--------------------------
            if(done){
                //--------------------------
                // _target_value = rewards;
                _target_value = m_model.forward(_next_input).detach();
                //--------------------------
            }// end if(done)
            else{
                //--------------------------
                // std::cout << "_state_value: "  << std::endl;
                //--------------------------
                auto _state_value = m_model.forward(_next_input).detach();
                // _target_value = rewards + gamma * _state_value;
                _target_value = torch::add(torch::mul(_state_value, gamma), rewards);
                //--------------------------
                // std::cout << "rewards: " << rewards << "_state_value: " << _state_value  << "\n _target_value: " << _target_value << std::endl;
                //--------------------------
                // std::cout << "_target_value: " << _target_value.sizes()  << std::endl;
                //--------------------------
            }// end else
            //--------------------------
            optimizer.zero_grad();
            //--------------------------
            // std::cout << " _predicted_value input: " << input.sizes()  << std::endl;
            //--------------------------
            auto _predicted_value = m_model.forward(_input);
            //--------------------------
            // std::cout << "_predicted_value: " << _predicted_value.sizes()  << std::endl;
            //--------------------------
            // std::cout << "_predicted_value: " << _predicted_value  << "\n _target_value: " << _target_value << std::endl;
            //--------------------------
            torch::Tensor loss = torch::mse_loss(_predicted_value, _target_value);
            //--------------------------
            if(torch::isnan(loss).any().item<bool>()){
                //--------------------------
                throw std::overflow_error("\x1b[31m" "\033[1m" "NaN Is Detected In The Agent" "\033[m" "\x1b[0m");
                //--------------------------
            }//end if( torch::isnan(_predicted_value) || torch::isnan(_target_value))
            //--------------------------
            // std::cout << "loss "  << std::endl;
            //--------------------------
            loss.backward({},c10::optional<bool>(true), false);
            optimizer.step();
            //-------------------------- 
        }// end void agent(const torch::Tensor& input, const torch::Tensor& next_input, const torch::Tensor& action, const T& rewards, const bool& done)
        //--------------------------
        void agent_optimizer(   const torch::Tensor& input, 
                                torch::optim::Optimizer& optimizer, 
                                const torch::Tensor& rewards, 
                                const bool& done, 
                                const double& gamma){
            //--------------------------
            torch::Tensor _input = input;
            //--------------------------
            std::random_device rd;  // Will be used to obtain a seed for the random number engine
            std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd(
            std::uniform_real_distribution<double> uniform_angle(0., 2*M_PI);
            std::default_random_engine re;
            //--------------------------
            std::vector<double> _points;
            _points.reserve(2);
            //--------------------------
            auto _angle = uniform_angle(re);
            _points.push_back(((_input[-1][0] * torch::tensor(std::sin(_angle)))+_input[-1][1]).item<double>());
            _points.push_back(((_input[-1][0] * torch::tensor(std::cos(_angle)))+_input[-1][2]).item<double>());
            //--------------------------
            auto _target_value = torch::tensor(_points).view({-1,2});
            //--------------------------
            if(!done){
                //--------------------------
                _target_value = rewards + gamma * _target_value;
                //--------------------------
            }// end if(done)
            //--------------------------
            m_model.train(true);
            //--------------------------
            optimizer.zero_grad();
            //--------------------------
            // std::cout << " _predicted_value input: " << input.sizes()  << std::endl;
            //--------------------------
            auto _predicted_value = m_model.forward(_input);
            //--------------------------
            // std::cout << "_predicted_value: " << _predicted_value.sizes()  << std::endl;
            //--------------------------
            torch::Tensor loss = torch::mse_loss(_predicted_value, _target_value);
            //--------------------------
            // std::cout << "loss "  << std::endl;
            //--------------------------
            loss.backward({},c10::optional<bool>(true), false);
            optimizer.step();
            //--------------------------
        }// end void agent(const torch::Tensor& input, const torch::Tensor& next_input, const torch::Tensor& action, const T& rewards, const bool& done)
        //--------------------------
        torch::Tensor network_test(const torch::Tensor& input){
            //--------------------------
            torch::Tensor _input = input;
            //--------------------------
            torch::NoGradGuard no_grad;
            m_model.eval();
            //--------------------------
            auto output = m_model.forward(_input);
            //--------------------------
            return output;
            //--------------------------
        }// end torch::Tensor network_test(const torch::Tensor& input)
        //--------------------------------------------------------------
    private:
        //--------------------------
        Network m_model; 
        torch::Device m_device;
        //--------------------------
        std::function<torch::Tensor(Args...)> m_random_action;
    //--------------------------------------------------------------
};// end class ReinforcementNetworkHandling : protected NetworkHandling
//--------------------------------------------------------------