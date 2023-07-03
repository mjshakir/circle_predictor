#pragma once 
//--------------------------------------------------------------
// User defind library
//--------------------------------------------------------------
// Environment
//--------------------------
#include "Environment/RL/Environment.hpp"
//--------------------------
// Network Handling
//--------------------------
#include "Network/RL/ReinforcementNetworkHandlingDQN.hpp"
//--------------------------
// Memory Replay
//--------------------------
#include "Network/RL/ExperienceReplay.hpp"
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
                                                                gen(std::random_device{}()), memory_activation(memory_percentage) {
                //--------------------------
                // static_assert(  std::is_same<ENVIRONMENT, RL::Environment::EnvironmentTestLoader<typename ENVIRONMENT::value_type>>::value or
                //                 std::is_same<ENVIRONMENT, RL::Environment::EnvironmentTestLoader<typename std::decay<ENVIRONMENT>::type>>::value or
                //                 std::is_same<ENVIRONMENT, RL::Environment::RLEnvironment<typename ENVIRONMENT::value_type, typename ENVIRONMENT::value_type, typename ENVIRONMENT::value_type>>::value or
                //                 std::is_same<ENVIRONMENT, RL::Environment::RLEnvironment<typename std::decay<ENVIRONMENT>::type, typename std::decay<ENVIRONMENT>::type, typename std::decay<ENVIRONMENT>::type>>::value or
                //                 std::is_same<ENVIRONMENT, RL::Environment::RLEnvironmentLoader<typename ENVIRONMENT::value_type, typename ENVIRONMENT::value_type, typename ENVIRONMENT::value_type>>::value or
                //                 std::is_same<ENVIRONMENT, RL::Environment::RLEnvironmentLoader<typename std::decay<ENVIRONMENT>::type, typename std::decay<ENVIRONMENT>::type, typename std::decay<ENVIRONMENT>::type>>::value or
                //                 std::is_same<ENVIRONMENT, RL::Environment::RLEnvironmentRandomLoader<typename ENVIRONMENT::value_type, typename ENVIRONMENT::value_type, typename ENVIRONMENT::value_type>>::value or
                //                 std::is_same<ENVIRONMENT, RL::Environment::RLEnvironmentRandomLoader<typename std::decay<ENVIRONMENT>::type, typename std::decay<ENVIRONMENT>::type, typename std::decay<ENVIRONMENT>::type>>::value, 
                //                 "ENVIRONMENT template must be one of EnvironmentTestLoader, RLEnvironment, RLEnvironmentLoader, or RLEnvironmentRandomLoader class.");
                // //--------------------------
                // static_assert(  std::is_same<HANDLER, ReinforcementNetworkHandling<typename HANDLER::value_type, typename HANDLER::value_type>>::value or
                //                 std::is_same<HANDLER, ReinforcementNetworkHandling<typename std::decay<HANDLER>::type, typename std::decay<HANDLER>::type>>::value or
                //                 std::is_same<HANDLER, ReinforcementNetworkHandlingDQN<typename HANDLER::value_type, typename HANDLER::value_type>>::value or
                //                 std::is_same<HANDLER, ReinforcementNetworkHandlingDQN<typename std::decay<HANDLER>::type, typename std::decay<HANDLER>::type>>::value,  
                //                 "HANDLER template must be one of ReinforcementNetworkHandling, or ReinforcementNetworkHandlingDQN class.");
                // //--------------------------
                // static_assert(  std::is_same<MEMORY, ExperienceReplay<typename MEMORY::value_type>>::value or
                //                 std::is_same<MEMORY, ExperienceReplay<typename std::decay<MEMORY>::type>>::value,
                //                 "MEMORY template must be ExperienceReplay class.");
                //--------------------------
                // static_assert(  std::is_base_of<Environment::EnvironmentDataLoaderBase, ENVIRONMENT>::value or
                //                 std::is_base_of<Environment::RLEnvironmentBase, ENVIRONMENT>::value or
                //                 std::is_base_of<Environment::RLEnvironmentLoaderBase, ENVIRONMENT>::value or
                //                 std::is_base_of<Environment::RLEnvironmentRandomLoaderBase, ENVIRONMENT>::value,
                //                 "ENVIRONMENT template must be one of EnvironmentTestLoader, RLEnvironment, RLEnvironmentLoader, or RLEnvironmentRandomLoader class.");
                // //--------------------------
                // static_assert(  std::is_base_of<ReinforcementNetworkHandling, HANDLER>::value or
                //                 std::is_base_of<ReinforcementNetworkHandlingDQN, HANDLER>::value, 
                //                 "HANDLER template must be one of ReinforcementNetworkHandling, or ReinforcementNetworkHandlingDQN class.");
                // //--------------------------
                // static_assert(  std::is_base_of<ExperienceReplay, MEMORY>::value,
                //                 "MEMORY template must be ExperienceReplay class.");
                //--------------------------
                // static_assert(  std::is_same<ENVIRONMENT, RL::Environment::EnvironmentTestLoader<typename ENVIRONMENT::value_type>>::value or
                //                 std::is_same<ENVIRONMENT, RL::Environment::RLEnvironment<typename ENVIRONMENT::value_type, typename ENVIRONMENT::value_type, typename ENVIRONMENT::value_type>>::value or
                //                 std::is_same<ENVIRONMENT, RL::Environment::RLEnvironmentLoader<typename ENVIRONMENT::value_type, typename ENVIRONMENT::value_type, typename ENVIRONMENT::value_type>>::value or
                //                 std::is_same<ENVIRONMENT, RL::Environment::RLEnvironmentRandomLoader<typename ENVIRONMENT::value_type, typename ENVIRONMENT::value_type, typename ENVIRONMENT::value_type>>::value,
                //                 "ENVIRONMENT template must be one of EnvironmentTestLoader, RLEnvironment, RLEnvironmentLoader, or RLEnvironmentRandomLoader class.");
                // //--------------------------
                // static_assert(  std::is_same<HANDLER, ReinforcementNetworkHandling<typename HANDLER::value_type, typename HANDLER::value_type>>::value or
                //                 std::is_same<HANDLER, ReinforcementNetworkHandlingDQN<typename HANDLER::value_type, typename HANDLER::value_type>>::value,  
                //                 "HANDLER template must be one of ReinforcementNetworkHandling, or ReinforcementNetworkHandlingDQN class.");
                // //--------------------------
                // static_assert(  std::is_same<MEMORY, ExperienceReplay<typename MEMORY::value_type>>::value,
                //                 "MEMORY template must be ExperienceReplay class.");
                //--------------------------
                // static_assert(std::is_same_v<std::decay_t<ENVIRONMENT>, RL::Environment::RLEnvironment<typename std::decay_t<ENVIRONMENT>::value_type, typename std::decay_t<ENVIRONMENT>::value_type, typename std::decay_t<ENVIRONMENT>::value_type>> ||
                //       std::is_same_v<std::decay_t<ENVIRONMENT>, RL::Environment::RLEnvironmentLoader<typename std::decay_t<ENVIRONMENT>::value_type, typename std::decay_t<ENVIRONMENT>::value_type, typename std::decay_t<ENVIRONMENT>::value_type, typename std::decay_t<ENVIRONMENT>::value_type>> ||
                //       std::is_same_v<std::decay_t<ENVIRONMENT>, RL::Environment::RLEnvironmentRandomLoader<typename std::decay_t<ENVIRONMENT>::value_type, typename std::decay_t<ENVIRONMENT>::value_type, typename std::decay_t<ENVIRONMENT>::value_type>>,
                //       "ENVIRONMENT template must be one of EnvironmentTestLoader, RLEnvironment, RLEnvironmentLoader, or RLEnvironmentRandomLoader class.");
                // //--------------------------
                // static_assert(std::is_same_v<std::decay_t<HANDLER>, ReinforcementNetworkHandling<typename std::decay_t<HANDLER>::value_type, typename std::decay_t<HANDLER>::value_type>> ||
                //             std::is_same_v<std::decay_t<HANDLER>, ReinforcementNetworkHandlingDQN<typename std::decay_t<HANDLER>::value_type, typename std::decay_t<HANDLER>::value_type>>,
                //             "HANDLER template must be one of ReinforcementNetworkHandling or ReinforcementNetworkHandlingDQN class.");
                // //--------------------------
                // static_assert(std::is_same_v<std::decay_t<MEMORY>, ExperienceReplay<typename std::decay_t<MEMORY>::value_type>>,
                //             "MEMORY template must be ExperienceReplay class.");
                //--------------------------
                // static_assert(std::is_same_v<std::decay_t<MEMORY>, ExperienceReplay<typename std::decay_t<MEMORY>::value_type>>, 
                //                     "MEMORY template must be ExperienceReplay class.");

                //--------------------------
                // static_assert(std::is_same_v<std::decay_t<MEMORY>, ExperienceReplay<typename std::decay_t<MEMORY>::value_type>>, 
                //                     "MEMORY template must be ExperienceReplay class.");

                //--------------------------
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
                        if(memory_activation(gen)){
                            //--------------------------
                            auto [_memory_input, _memory_next_input, _memory_reward, _done] = m_memory.sample();
                            //--------------------------
                            m_handler.agent(_memory_input, _memory_next_input, optimizer, _memory_reward, _done);
                            //--------------------------
                        }//end if(memory_activation(gen))
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
                // std::lock_guard<std::mutex> _lock_guard(m_mutex);
                //--------------------------
                bool done = false;
                double epsilon = 0.;
                //--------------------------
                auto _input = environment.get_first(epsilon).to(m_device);
                //--------------------------
                // std::cout << "_input: " << _input.sizes() << std::endl;
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
                        if(memory_activation(gen)){
                            //--------------------------
                            auto [_memory_input, _memory_next_input, _memory_reward, _done] = m_memory.sample();
                            //--------------------------
                            m_handler.agent(_memory_input, _memory_next_input, optimizer, _memory_reward, _done);
                            //--------------------------
                        }//end if(memory_activation(gen))
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
                        if(memory_activation(gen)){
                            //--------------------------
                            auto [_memory_input, _memory_next_input, _memory_reward, _done] = memory.sample();
                            //--------------------------
                            m_handler.agent(_memory_input, _memory_next_input, optimizer, _memory_reward, _done);
                            //--------------------------
                        }//end if(memory_activation(gen))
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
                    // ENVIRONMENT _environment = m_environment;
                    // MEMORY _memory = m_memory;
                    //--------------------------
                    // train_run(_environment, _memory, optimizer, normalizing_function, args...);
                    //--------------------------
                    // train_run(_environment, optimizer, normalizing_function, args...);
                    //--------------------------
                    train_run(optimizer, normalizing_function, args...);
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
                // if(jobs > std::thread::hardware_concurrency()){
                //     //--------------------------
                //     throw std::out_of_range("jobs Size: [" + std::to_string(jobs) + 
                //                             "] Must Be Less or Equal Hardware cores size: [" + std::to_string(std::thread::hardware_concurrency()) + "]");
                //     //--------------------------
                // }// end if(jobs > std::thread::hardware_concurrency())
                //--------------------------
                std::vector<std::thread> threads;
                threads.reserve(jobs);
                //--------------------------
                auto _epoch = static_cast<size_t>(epoch/jobs);
                //--------------------------
                progressbar bar(_epoch);
                //--------------------------
                for (size_t i = 0; i < _epoch; ++i) {
                    //--------------------------
                    // for(size_t j = 0; j < jobs; ++j){
                    //     //--------------------------
                    //     threads.emplace_back([this, &optimizer, &normalizing_function, &args...](){train_run(optimizer, normalizing_function, args...);});
                    //     //--------------------------
                    // }// end for(size_t j = 0; j < jobs; ++j)
                    //--------------------------
                    for(size_t j = 0; j < jobs; ++j){
                        //--------------------------
                        ENVIRONMENT _environment = m_environment;
                        //--------------------------
                        threads.emplace_back([this, &_environment, &optimizer, &normalizing_function, &args...](){
                            train_run(_environment, optimizer, normalizing_function, args...);});
                        //--------------------------
                    }// end for(size_t j = 0; j < jobs; ++j)
                    // //--------------------------
                    // std::generate_n(std::execution::par, std::back_inserter(threads), jobs,
                    //         [this, &optimizer, &normalizing_function, &args...](){ 
                    //             return std::thread([this, &optimizer, &normalizing_function, &args...](){train_run(optimizer, normalizing_function, args...);});});
                    //--------------------------
                    // std::generate_n(std::execution::par, std::back_inserter(threads), jobs,
                    //         [this, &optimizer, &normalizing_function, &args...](){ 
                    //             ENVIRONMENT _environment = m_environment;
                    //             MEMORY _memory = m_memory;
                    //             return std::thread([this,&_environment, &_memory, &optimizer, &normalizing_function, &args...](){
                    //                 train_run(_environment, _memory, optimizer,  normalizing_function, args...);});});
                    //--------------------------
                    std::for_each(std::execution::par, threads.begin(), threads.end(), [](auto& _thread){_thread.join();});
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
            std::mt19937 gen;
            std::bernoulli_distribution memory_activation;
            //--------------------------
            std::mutex m_mutex;
            //--------------------------------------------------------------
    };// end class Train
    //--------------------------------------------------------------
}// end namespace RL
//--------------------------------------------------------------