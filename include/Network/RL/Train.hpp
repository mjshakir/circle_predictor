#pragma once 
//--------------------------------------------------------------
// User defind library
//--------------------------------------------------------------
// Environment
//--------------------------
// #include "Environment/RL/Environment.hpp"
//--------------------------
// Network Handling
//--------------------------
// #include "Network/RL/ReinforcementNetworkHandlingDQN.hpp"
//--------------------------
// Memory Replay
//--------------------------
// #include "Network/RL/ExperienceReplay.hpp"
//--------------------------
// Check utilities
//--------------------------
#include "Utilities/StaticCheck.hpp"
//--------------------------
#include "Generate/RL/RLNormalize.hpp"
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <type_traits>
#include <random>
#include <optional>
#include <thread>
#include <algorithm>
#include <execution>
//--------------------------------------------------------------
// Progressbar library
//--------------------------------------------------------------
#include "progressbar/include/progressbar.hpp"
//--------------------------------------------------------------
// LibFort library (enable table printing)
//--------------------------------------------------------------
#include "fort.hpp"
//--------------------------------------------------------------

//--------------------------------------------------------------
namespace RL {
    //--------------------------------------------------------------
    template <typename ENVIRONMENT, typename HANDLER, typename MEMORY>
    class Train {
        public:
            //--------------------------------------------------------------
            Train(void) = delete;
            //--------------------------
            Train(  ENVIRONMENT&& environment, 
                    HANDLER&& handler, 
                    MEMORY&& memory, 
                    const torch::Device& device = torch::kCPU,
                    const double& memory_percentage = 0.3) :    m_environment(std::move(environment)),
                                                                m_handler(std::move(handler)),
                                                                m_memory(std::move(memory)),
                                                                m_device(device),
                                                                m_gen(std::random_device{}()), m_memory_activation(memory_percentage) {
                //--------------------------             
                static_assert(Utils::CheckEnvironment<std::decay_t<ENVIRONMENT>>::value, 
                            "\x1b[36m ENVIRONMENT template must be one of RLEnvironment, RLEnvironmentLoader, or RLEnvironmentRandomLoader class. \x1b[0m");
                //--------------------
                static_assert(Utils::CheckHandler<std::decay_t<HANDLER>>::value, 
                                "\x1b[36m HANDLER template must be one of ReinforcementNetworkHandling, or ReinforcementNetworkHandlingDQN class.\x1b[0m");
                //--------------------
                static_assert(Utils::CheckExperienceReplay<std::decay_t<MEMORY>>::value, "\x1b[36m MEMORY template must be ExperienceReplay class. \x1b[0m");
                //--------------------
            }//end  Train(ENVIRONMENT&& environment, HANDLER&& handler, MEMORY&& memory)
            //--------------------------
            template<typename... Args>
            void run(torch::optim::Optimizer& optimizer, std::function<torch::Tensor(torch::Tensor)> normalizing_function, const Args&... args){
                //--------------------------
                train_run(optimizer, normalizing_function, args...);
                //--------------------------
            }// end void run(void)
            //--------------------------
            template<typename... Args>
            void run(const size_t& epoch, torch::optim::Optimizer& optimizer, std::function<torch::Tensor(const torch::Tensor&)> normalizing_function, const Args&... args){
                //--------------------------
                train_run(epoch, optimizer, normalizing_function, args...);
                //--------------------------
            }// end void run(size_t epoch)
            //--------------------------
            template<typename... Args>
            void run(const size_t& epoch, const size_t jobs, torch::optim::Optimizer& optimizer, std::function<torch::Tensor(torch::Tensor)> normalizing_function, const Args&... args){
                //--------------------------
                train_run(epoch, jobs, optimizer, normalizing_function, args...);
                //--------------------------
            }// end void run(size_t epoch)
            //--------------------------
            template<typename ENVIRONMENT_TEST>
            void test(ENVIRONMENT_TEST&& test_environment, const torch::Tensor& tensor_min, const torch::Tensor& tensor_max, const bool& varbos = false){
                //--------------------------
                test_run(std::move(test_environment), tensor_min, tensor_max, varbos);
                //--------------------------
            }// end void test(ENVIRONMENT_TEST&& test_environment, torch::Tensor tensor_min, torch::Tensor tensor_max, const bool& varbos = false)
            //--------------------------------------------------------------
        protected:
            //--------------------------------------------------------------
            template<typename... Args>
            void train_run(torch::optim::Optimizer& optimizer, std::function<torch::Tensor(const torch::Tensor&)> normalizing_function, const Args&... args){
                //--------------------------
                std::lock_guard<std::mutex> _lock_guard(m_mutex);
                //--------------------------
                bool done = false;
                double epsilon = 0.;
                //--------------------------
                auto _input = m_environment.get_first(epsilon).to(m_device);
                //--------------------------
                // std::cout << "_input: " << _input.sizes() << std::endl;
                //--------------------------
                auto output = m_handler.action(_input, epsilon, args...).to(m_device);
                //--------------------------
                auto [next_input, reward] = m_environment.step(epsilon, done, _input, normalizing_function(output));
                //--------------------------
                m_handler.agent(_input, next_input.to(m_device), optimizer, reward, done);
                //--------------------------
                m_memory.push(_input, next_input, reward, done);
                //--------------------------
                torch::Tensor training_input = next_input;
                //--------------------------
                while(!done){
                    //--------------------------
                    output = m_handler.action(training_input, epsilon, args...);
                    //--------------------------
                    std::tie(next_input, reward) = m_environment.step(epsilon, done, training_input,  normalizing_function(output));
                    //--------------------------
                    m_memory.push(training_input, next_input.to(m_device), reward, done);
                    //--------------------------
                    try{
                        //--------------------------
                        if(m_memory_activation(m_gen)){
                            //--------------------------
                            auto [_memory_input, _memory_next_input, _memory_reward, _done] = m_memory.sample();
                            //--------------------------
                            m_handler.agent(_memory_input, _memory_next_input, optimizer, _memory_reward, _done);
                            //--------------------------
                        }//end if(m_memory_activation(m_gen))
                        else{
                            //--------------------------
                            m_handler.agent(training_input, next_input, optimizer, reward, done);
                            //--------------------------
                        }// end else
                        //--------------------------
                    }// end try
                    catch(std::overflow_error& e) {
                        //--------------------------
                        std::cerr << "\n" << e.what() << std::endl;
                        //--------------------------
                        std::exit(-1);
                        //--------------------------
                    }// end catch(std::out_of_range& e)
                    //--------------------------
                    training_input = next_input;
                    //--------------------------
                }// end while(!_done)
                //--------------------------
                m_environment.reset();
                //--------------------------
            }// end void train_run(void)
            //--------------------------------------------------------------
            template<typename... Args>
            void train_run( ENVIRONMENT environment,
                            torch::optim::Optimizer& optimizer,
                            std::function<torch::Tensor(const torch::Tensor&)> normalizing_function,
                            const Args&... args){
                //--------------------------
                std::lock_guard<std::mutex> _lock_guard(m_mutex);
                //--------------------------
                bool done = false;
                double epsilon = 0.;
                //--------------------------
                auto _input = environment.get_first(epsilon).to(m_device);
                //--------------------------
                auto output = m_handler.action(_input, epsilon, args...).to(m_device);
                //--------------------------
                auto [next_input, reward] = environment.step(epsilon, done, _input, normalizing_function(output));
                //--------------------------
                m_handler.agent(_input, next_input.to(m_device), optimizer, reward, done);
                //--------------------------
                m_memory.push(_input, next_input, reward, done);
                //--------------------------
                torch::Tensor training_input = next_input;
                //--------------------------
                while(!done){
                    //--------------------------
                    output = m_handler.action(training_input, epsilon, args...);
                    //--------------------------
                    std::tie(next_input, reward) = environment.step(epsilon, done, training_input,  normalizing_function(output));
                    //--------------------------
                    m_memory.push(training_input, next_input.to(m_device), reward, done);
                    //--------------------------
                    try{
                        //--------------------------
                        if(m_memory_activation(m_gen)){
                            //--------------------------
                            auto [_memory_input, _memory_next_input, _memory_reward, _done] = m_memory.sample();
                            //--------------------------
                            m_handler.agent(_memory_input, _memory_next_input, optimizer, _memory_reward, _done);
                            //--------------------------
                        }//end if(m_memory_activation(m_gen))
                        else{
                            //--------------------------
                            m_handler.agent(training_input, next_input, optimizer, reward, done);
                            //--------------------------
                        }// end else
                        //--------------------------
                    }// end try
                    catch(std::overflow_error& e) {
                        //--------------------------
                        std::cerr << "\n" << e.what() << std::endl;
                        //--------------------------
                        std::exit(-1);
                        //--------------------------
                    }// end catch(std::out_of_range& e)
                    //--------------------------
                    training_input = next_input;
                    //--------------------------
                }// end while(!_done)
                //--------------------------
                environment.reset();
                //--------------------------
            }// end void train_run(void)
            //--------------------------------------------------------------
            template<typename... Args>
            void train_run( std::shared_ptr<ENVIRONMENT> environment,
                            torch::optim::Optimizer& optimizer,
                            std::function<torch::Tensor(const torch::Tensor&)> normalizing_function,
                            const Args&... args){
                //--------------------------
                std::lock_guard<std::mutex> _lock_guard(m_mutex);
                //--------------------------
                bool done = false;
                double epsilon = 0.;
                //--------------------------
                auto _input = environment->get_first(epsilon).to(m_device);
                //--------------------------
                auto output = m_handler.action(_input, epsilon, args...).to(m_device);
                //--------------------------
                auto [next_input, reward] = environment->step(epsilon, done, _input, normalizing_function(output));
                //--------------------------
                m_handler.agent(_input, next_input.to(m_device), optimizer, reward, done);
                //--------------------------
                m_memory.push(_input, next_input, reward, done);
                //--------------------------
                torch::Tensor training_input = next_input;
                //--------------------------
                while(!done){
                    //--------------------------
                    output = m_handler.action(training_input, epsilon, args...);
                    //--------------------------
                    std::tie(next_input, reward) = environment->step(epsilon, done, training_input,  normalizing_function(output));
                    //--------------------------
                    m_memory.push(training_input, next_input.to(m_device), reward, done);
                    //--------------------------
                    try{
                        //--------------------------
                        if(m_memory_activation(m_gen)){
                            //--------------------------
                            auto [_memory_input, _memory_next_input, _memory_reward, _done] = m_memory.sample();
                            //--------------------------
                            m_handler.agent(_memory_input, _memory_next_input, optimizer, _memory_reward, _done);
                            //--------------------------
                        }//end if(m_memory_activation(m_gen))
                        else{
                            //--------------------------
                            m_handler.agent(training_input, next_input, optimizer, reward, done);
                            //--------------------------
                        }// end else
                        //--------------------------
                    }// end try
                    catch(std::overflow_error& e) {
                        //--------------------------
                        std::cerr << "\n" << e.what() << std::endl;
                        //--------------------------
                        std::exit(-1);
                        //--------------------------
                    }// end catch(std::out_of_range& e)
                    //--------------------------
                    training_input = next_input;
                    //--------------------------
                }// end while(!_done)
                //--------------------------
                environment->reset();
                //--------------------------
            }// end void train_run(void)
            //-------------------------------------------------------------
            template<typename... Args>
            void train_local_run(   torch::optim::Optimizer& optimizer,
                                    std::function<torch::Tensor(const torch::Tensor&)> normalizing_function,
                                    const Args&... args){
                //--------------------------
                auto environment = m_environment;
                auto memory = m_memory;
                //--------------------------
                std::lock_guard<std::mutex> _lock_guard(m_mutex);
                //--------------------------
                bool done = false;
                double epsilon = 0.;
                //--------------------------
                auto _input = environment.get_first(epsilon).to(m_device);
                //--------------------------
                auto output = m_handler.action(_input, epsilon, args...).to(m_device);
                //--------------------------
                auto [next_input, reward] = environment.step(epsilon, done, _input, normalizing_function(output));
                //--------------------------
                m_handler.agent(_input, next_input.to(m_device), optimizer, reward, done);
                //--------------------------
                memory.push(_input, next_input, reward, done);
                //--------------------------
                torch::Tensor training_input = next_input;
                //--------------------------
                while(!done){
                    //--------------------------
                    output = m_handler.action(training_input, epsilon, args...);
                    //--------------------------
                    std::tie(next_input, reward) = environment.step(epsilon, done, training_input,  normalizing_function(output));
                    //--------------------------
                    memory.push(training_input, next_input.to(m_device), reward, done);
                    //--------------------------
                    try{
                        //--------------------------
                        if(m_memory_activation(m_gen)){
                            //--------------------------
                            auto [_memory_input, _memory_next_input, _memory_reward, _done] = memory.sample();
                            //--------------------------
                            m_handler.agent(_memory_input, _memory_next_input, optimizer, _memory_reward, _done);
                            //--------------------------
                        }//end if(m_memory_activation(m_gen))
                        else{
                            //--------------------------
                            m_handler.agent(training_input, next_input, optimizer, reward, done);
                            //--------------------------
                        }// end else
                        //--------------------------
                    }// end try
                    catch(std::overflow_error& e) {
                        //--------------------------
                        std::cerr << "\n" << e.what() << std::endl;
                        //--------------------------
                        std::exit(-1);
                        //--------------------------
                    }// end catch(std::out_of_range& e)
                    //--------------------------
                    training_input = next_input;
                    //--------------------------
                }// end while(!_done)
                //--------------------------
                environment.reset();
                //--------------------------
            }// end void train_run(void)
            //---------------------------------------------------------------
            template<typename... Args>
            void train_run( ENVIRONMENT environment,
                            MEMORY memory,
                            torch::optim::Optimizer& optimizer,
                            std::function<torch::Tensor(const torch::Tensor&)> normalizing_function,
                            const Args&... args){
                //--------------------------
                bool done = false;
                double epsilon = 0.;
                //--------------------------
                auto _input = environment.get_first(epsilon).to(m_device);
                //--------------------------
                auto output = m_handler.action(_input, epsilon, args...).to(m_device);
                //--------------------------
                auto [next_input, reward] = environment.step(epsilon, done, _input, normalizing_function(output));
                //--------------------------
                m_handler.agent(_input, next_input.to(m_device), optimizer, reward, done);
                //--------------------------
                memory.push(_input, next_input, reward, done);
                //--------------------------
                torch::Tensor training_input = next_input;
                //--------------------------
                while(!done){
                    //--------------------------
                    output = m_handler.action(training_input, epsilon, args...);
                    //--------------------------
                    std::tie(next_input, reward) = environment.step(epsilon, done, training_input,  normalizing_function(output));
                    //--------------------------
                    memory.push(training_input, next_input.to(m_device), reward, done);
                    //--------------------------
                    try{
                        //--------------------------
                        if(m_memory_activation(m_gen)){
                            //--------------------------
                            auto [_memory_input, _memory_next_input, _memory_reward, _done] = memory.sample();
                            //--------------------------
                            m_handler.agent(_memory_input, _memory_next_input, optimizer, _memory_reward, _done);
                            //--------------------------
                        }//end if(m_memory_activation(m_gen))
                        else{
                            //--------------------------
                            m_handler.agent(training_input, next_input, optimizer, reward, done);
                            //--------------------------
                        }// end else
                        //--------------------------
                    }// end try
                    catch(std::overflow_error& e) {
                        //--------------------------
                        std::cerr << "\n" << e.what() << std::endl;
                        //--------------------------
                        std::exit(-1);
                        //--------------------------
                    }// end catch(std::out_of_range& e)
                    //--------------------------
                    training_input = next_input;
                    //--------------------------
                }// end while(!_done)
                //--------------------------
                environment.reset();
                //--------------------------
            }// end void train_run(void)
            //--------------------------------------------------------------
            template<typename... Args>
            void train_run(const size_t& epoch, torch::optim::Optimizer& optimizer, std::function<torch::Tensor(torch::Tensor)> normalizing_function, const Args&... args){
                //--------------------------
                progressbar bar(epoch);
                //--------------------------
                for (size_t i = 0; i < epoch; ++i) {
                    //--------------------------
                    ENVIRONMENT _environment = m_environment;
                    //--------------------------
                    train_run(_environment, optimizer, normalizing_function, args...);
                    //--------------------------
                    // train_run(optimizer, normalizing_function, args...);
                    //--------------------------
                    bar.update();
                    //--------------------------
                }// end for (size_t i = 0; i < epoch; ++i)
                //--------------------------
            }// end void train_run(const size_t& epoch)
            //--------------------------------------------------------------
            template<typename... Args>
            void train_run(const size_t& epoch, const size_t jobs, torch::optim::Optimizer& optimizer, std::function<torch::Tensor(torch::Tensor)> normalizing_function, const Args&... args){
                //--------------------------
                if(jobs > std::thread::hardware_concurrency()){
                    //--------------------------
                    throw std::out_of_range("jobs Size: [" + std::to_string(jobs) + 
                                            "] Must Be Less or Equal Hardware cores size: [" + std::to_string(std::thread::hardware_concurrency()) + "]");
                    //--------------------------
                }// end if(jobs > std::thread::hardware_concurrency())
                //--------------------------
                std::vector<std::jthread> threads;
                threads.reserve(jobs);
                //--------------------------
                auto _epoch = static_cast<size_t>(epoch/jobs);
                //--------------------------
                progressbar bar(_epoch);
                //--------------------------
                for (size_t i = 0; i < _epoch; ++i) {
                    //--------------------------
                    for(size_t j = 0; j < jobs; ++j){
                        //--------------------------
                        threads.emplace_back([this, &optimizer, &normalizing_function, &args...](){train_local_run(optimizer, normalizing_function, args...);});
                        //--------------------------
                    }// end for(size_t j = 0; j < jobs; ++j)
                    //--------------------------
                    threads.clear();
                    //--------------------------
                    bar.update();
                    //--------------------------
                }// end for (size_t i = 0; i < _epoch; ++i)
                //--------------------------
            }// end void train_run(const size_t& epoch)
            //--------------------------------------------------------------
            template<typename ENVIRONMENT_TEST>
            void test_run(ENVIRONMENT_TEST&& test_environment, const torch::Tensor& t_min, const torch::Tensor& t_max, const bool& varbos){
                //--------------------------------------------------------------
                // Print table settup
                //--------------------------
                std::optional<fort::char_table> table;
                //--------------------------
                if(varbos){
                    //--------------------------
                    // Initialize the optional table
                    //--------------------------
                    table.emplace();
                    //--------------------------
                    // Change border style
                    //--------------------------
                    table->set_border_style(FT_BASIC2_STYLE);
                    //--------------------------
                    // Set color
                    //--------------------------
                    table->row(0).set_cell_content_fg_color(fort::color::light_blue);
                    //--------------------------
                    // Set center alignment for the all columns
                    //--------------------------
                    table->column(0).set_cell_text_align(fort::text_align::center);
                    table->column(1).set_cell_text_align(fort::text_align::center);
                    table->column(2).set_cell_text_align(fort::text_align::center);
                    table->column(3).set_cell_text_align(fort::text_align::center);
                    table->column(4).set_cell_text_align(fort::text_align::center);
                    table->column(5).set_cell_text_align(fort::text_align::center);
                    //--------------------------
                    *table  << fort::header
                            << "X_1" << "X" << "Y_1" << "Y" << "Original Target" << "Output" << "Loss" << fort::endr;
                    //--------------------------------------------------------------
                }//end if(varbos)
                //--------------------------
                bool done{false};
                //--------------------------
                while (!done){
                    //--------------------------
                    auto _test = test_environment.step(done);
                    //--------------------------
                    auto _test_result = m_handler.test(_test);
                    //--------------------------
                    auto _circle = torch::pow((_test_result[0].slice(1,0,1) - _test.slice(1,0,1)),2) + (torch::pow((_test_result[0].slice(1,1,2)-_test.slice(1,1,2)),2));
                    //--------------------------
                    auto _loss = torch::mse_loss(_circle, _test.slice(1,2,3));
                    //--------------------------
                    if(varbos){
                        //--------------------------
                        *table  << RL::RLNormalize::unnormalization(_test_result[0].slice(1,0,1), t_min, t_max) 
                                << RL::RLNormalize::unnormalization(_test.slice(1,0,1), t_min, t_max)
                                << RL::RLNormalize::unnormalization(_test_result[0].slice(1,1,2), t_min, t_max) 
                                << RL::RLNormalize::unnormalization(_test.slice(1,1,2), t_min, t_max)
                                << RL::RLNormalize::unnormalization(_test.slice(1,2,3), t_min, t_max)
                                << RL::RLNormalize::unnormalization(_circle, t_min, t_max)
                                << _loss << fort::endr;
                        //--------------------------
                    }// end if(varbos)
                    //--------------------------
                    // auto _circle = torch::pow((_test_result.slice(1,0,1) - _test.slice(1,0,1)),2)+ (torch::pow((_test_result.slice(1,1,2)-_test.slice(1,1,2)),2));
                    // //--------------------------
                    // auto _loss = torch::mse_loss(_circle, _test.slice(1,2,3));
                    // //--------------------------
                    // table   << RL::RLNormalize::unnormalization(_test_result.slice(1,0,1), t_min, t_max) 
                    //         << RL::RLNormalize::unnormalization(_test.slice(1,0,1), t_min, t_max)
                    //         << RL::RLNormalize::unnormalization(_test_result.slice(1,1,2), t_min, t_max) 
                    //         << RL::RLNormalize::unnormalization(_test.slice(1,1,2), t_min, t_max)
                    //         << RL::RLNormalize::unnormalization(_test.slice(1,2,3), t_min, t_max)
                    //         << RL::RLNormalize::unnormalization(_circle, t_min, t_max)
                    //         << _loss << fort::endr;
                    // //--------------------------
                    // table   << _test_result.slice(1,0,1)
                    //         << _test.slice(1,0,1)
                    //         << _test_result.slice(1,1,2)
                    //         << _test.slice(1,1,2)
                    //         << _test.slice(1,2,3)
                    //         << _circle
                    //         << _loss*100 << fort::endr;
                    // //--------------------------
                }// end while (!done)
                //--------------------------
                if(varbos){
                    //--------------------------
                    std::cout << "\n" << table->to_string() << std::endl;
                    //--------------------------
                }//end if(varbos)
                //--------------------------
            }// end void test_run(ENVIRONMENT_TEST&& test_environment, const bool& varbos)
            //--------------------------------------------------------------
        private:
            //--------------------------------------------------------------
            ENVIRONMENT m_environment;
            //--------------------------
            HANDLER     m_handler;
            //--------------------------
            MEMORY      m_memory;
            //--------------------------
            torch::Device m_device;
            //--------------------------
            std::mt19937 m_gen;
            std::bernoulli_distribution m_memory_activation;
            //--------------------------
            std::mutex m_mutex;
            //--------------------------------------------------------------
    };// end class Train
    //--------------------------------------------------------------
}// end namespace RL
//--------------------------------------------------------------