#pragma once

//--------------------------------------------------------------
// LibTorch library
//--------------------------------------------------------------
#include <torch/torch.h>
//--------------------------------------------------------------
// Standard cpp library
//--------------------------------------------------------------
#include <random>
//--------------------------------------------------------------
template<typename Network, typename... Args>
class ReinforcementNetworkHandling{
    //--------------------------------------------------------------
    public:
        //--------------------------------------------------------------
        // ReinforcementNetworkHandling(void) = delete;
        //--------------------------
        ReinforcementNetworkHandling(   Network& model, 
                                        std::function<torch::Tensor(Args&...)> actions):    m_model(model), 
                                                                                            m_random_action(std::move(actions)){
            //--------------------------
        }// end ReinforcementNetworkHandling(Network& model, torch::Device& device)
        //--------------------------
        ReinforcementNetworkHandling(   Network&& model, 
                                        std::function<torch::Tensor(Args&...)> actions) :   m_model(std::move(model)), 
                                                                                            m_random_action(std::move(actions)){
            //--------------------------
        }// end ReinforcementNetworkHandling(Network& model, torch::Device& device)
        //--------------------------
        torch::Tensor action(const torch::Tensor& input, const double& epsilon, Args... args){
            //--------------------------
            return select_action(input, epsilon, args...);
            //--------------------------
        }// end torch::Tensor select(const torch::Tensor& input, const double& epsilon, const Args... args)
        //--------------------------
        // virtual void agent( const torch::Tensor& input, 
        //                     const torch::Tensor& next_input, 
        //                     torch::optim::Optimizer& optimizer, 
        //                     const torch::Tensor& rewards, 
        //                     const bool& done, 
        //                     const double& gamma = 0.5,
        //                     const size_t& update_frequency = 100) = 0;
        //--------------------------
        virtual void agent( const torch::Tensor& input, 
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
                    const torch::Tensor& next_input, 
                    torch::optim::Optimizer& optimizer, 
                    const torch::Tensor& rewards, 
                    const torch::Tensor& done, 
                    const double& gamma = 0.5){
            //--------------------------
            agent_optimizer(input, next_input, optimizer, rewards, done, gamma);
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
        torch::Tensor select_action(const torch::Tensor& input, const double& epsilon, Args... args){
            //--------------------------
            std::random_device rd;  // Will be used to obtain a seed for the random number engine
            std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd(
            std::uniform_real_distribution<double> uniform_egreedy(0., 1.);
            //--------------------------
            auto _random_egreedy = uniform_egreedy(gen);
            //--------------------------
            if(_random_egreedy > epsilon){
                //--------------------------
                torch::Tensor _input = input;
                //--------------------------
                torch::NoGradGuard no_grad;
                m_model.eval();
                //--------------------------
                return m_model.forward(_input) ;
                //--------------------------
            } // end if(_random_egreedy > epsilon)     
            //--------------------------
            return m_random_action(args...);
            //--------------------------
        }// end torch::Tensor select_action(const torch::Tensor& input, const T& epsilon)
        //--------------------------------------------------------------
        // virtual void agent_optimizer(   const torch::Tensor& input, 
        //                                 const torch::Tensor& next_input, 
        //                                 torch::optim::Optimizer& optimizer, 
        //                                 const torch::Tensor& rewards, 
        //                                 const bool& done, 
        //                                 const double& gamma,
        //                                 const size_t& update_frequency) = 0;
        //--------------------------------------------------------------
        virtual void agent_optimizer(   const torch::Tensor& input, 
                                const torch::Tensor& next_input, 
                                torch::optim::Optimizer& optimizer, 
                                const torch::Tensor& rewards, 
                                const bool& done, 
                                const double& gamma){
            //--------------------------
            m_model.train(true);
            //--------------------------
            auto _target_value = rewards + (1 - done) * gamma * m_model.forward(next_input).detach();
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
            optimizer.step();
            //-------------------------- 
        }// end void agent(const torch::Tensor& input, const torch::Tensor& next_input, const torch::Tensor& action, const T& rewards, const bool& done)
        //--------------------------------------------------------------
        void agent_optimizer(   const torch::Tensor& input, 
                                const torch::Tensor& next_input, 
                                torch::optim::Optimizer& optimizer, 
                                const torch::Tensor& rewards, 
                                const torch::Tensor& done, 
                                const double& gamma){
            //--------------------------
            m_model.train(true);
            //--------------------------
            auto _target_value = rewards + (1 - done.to(torch::kByte)) * gamma * m_model.forward(next_input).detach();
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
            optimizer.step();
            //-------------------------- 
        }// end void agent(const torch::Tensor& input, const torch::Tensor& next_input, const torch::Tensor& action, const T& rewards, const bool& done)
        //--------------------------------------------------------------
        torch::Tensor network_test(const torch::Tensor& input){
            //--------------------------
            torch::NoGradGuard no_grad;
            m_model.eval();
            //--------------------------
            return m_model.forward(input);;
            //--------------------------
        }// end torch::Tensor network_test(const torch::Tensor& input)
        //--------------------------------------------------------------
         Network& get_model(void){
            //--------------------------
            return m_model;
            //--------------------------
        }// end Network& get_model(void)
        //--------------------------------------------------------------
    private:
        //--------------------------
        Network m_model; 
        //--------------------------
        std::function<torch::Tensor(Args&...)> m_random_action;
    //--------------------------------------------------------------
};// end class ReinforcementNetworkHandling : protected NetworkHandling
//--------------------------------------------------------------