//--------------------------------------------------------------
// Standard cpp library
//--------------------------------------------------------------
#include <random>
#include <fstream>
//--------------------------------------------------------------
// Boost library
//--------------------------------------------------------------
#include <boost/program_options.hpp>
//--------------------------------------------------------------
// User Defined library
//--------------------------------------------------------------
#include "Network/Networks.hpp"
#include "Generate/RL/RLEnvironment.hpp"
#include "Generate/RL/RLGenerate.hpp"
#include "Network/RL/ReinforcementNetworkHandling.hpp"
#include "Network/RL/RLNormalize.hpp"
#include "Network/RL/ExperienceReplay.hpp"
//--------------------------
#include "Timing/TimeIT.hpp"
//--------------------------------------------------------------
// Standard cpp library
//--------------------------------------------------------------
#include <future>
#include <algorithm>
#include <execution>
//--------------------------------------------------------------
// Progressbar library
//--------------------------------------------------------------
#include "progressbar/include/progressbar.hpp"
//--------------------------------------------------------------
int main(void){
    //--------------------------------------------------------------
    // Command line arugments using boost options 
    //--------------------------
    // std::string filename;
    size_t generated_size = 60000, points_size = 3, test_size = 100, limiter = 10, output_size =2, batch_size = 100, epoch = 1000, capacity = 3*batch_size;
    // long double precision;
    //--------------------------
    //--------------------------------------------------------------
    // Initiate Torch seed, device type
    //--------------------------
    torch::manual_seed(17);
    //--------------------------
    torch::DeviceType device_type;
    //--------------------------
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
        torch::cuda::manual_seed(7);
    }// end if (torch::cuda::is_available()) 
    else {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }// end else
    //--------------------------
    torch::Device device(device_type);
    //--------------------------------------------------------------
    //--------------------------
    // TimeIT _timer; 
    RLGenerate _generate(generated_size, test_size, points_size, limiter);
    // std::cout << "RLGenerate time: " << _timer.get_time_seconds() << std::endl;
    //--------------------------
    RLNormalize _normalize(_generate.get_input());
    //--------------------------
    auto input = _normalize.normalization();
    //--------------------------
    // auto input_test = _normalize.normalization(_generate.data(test_size, 3));
    //--------------------------
    auto input_test_thread = std::async(std::launch::async, [&_generate](){
                                    // auto x = _generate.get_test_input();
                                    // std::cout << "_generate.get_test_input(): " << _generate.get_test_input() << std::endl;
                                    // auto y = RLNormalize::normalization_min_max(_generate.get_test_input());
                                    // auto [z, mi, ma] = y;
                                    // for (std::cout << "Normalized: "; const auto& a: z){
                                    //     std::cout << a << "\n";
                                    // }
                                    return RLNormalize::normalization_min_max(_generate.get_test_input());});
    //--------------------------------------------------------------
    // auto _circle_reward = [](const torch::Tensor& input, const torch::Tensor& output){
    //                             //--------------------------
    //                             auto _circle = torch::pow((output[-1][0]-input[-1][0]),2)+torch::pow((output[-1][1]-input[-1][1]),2);
    //                             //--------------------------
    //                             if( _circle.equal(torch::pow(input[-1][2],2))){
    //                                 //--------------------------
    //                                 return torch::tensor(0);
    //                                 //--------------------------
    //                             }// end if( _circle.equal(input[0]) or torch::less_equal(torch::abs(_circle-input[0]), 1E-1).any().item<bool>() )
    //                             //--------------------------
    //                             if(torch::abs(_circle-torch::pow(input[-1][2],2)).any().item<double>() <= 1E-4){
    //                                 //--------------------------
    //                                 return torch::tensor(0.1);
    //                                 //--------------------------
    //                             }// end if( _circle.equal(input[0]) or torch::less_equal(torch::abs(_circle-input[0]), 1E-1).any().item<bool>() )
    //                             //--------------------------
    //                             if(torch::abs(_circle-torch::pow(input[-1][2],2)).any().item<double>() <  1E-2){
    //                                 //--------------------------
    //                                 return torch::tensor(0.5);
    //                                 //--------------------------
    //                             }// end if( _circle.equal(input[0]) or torch::less_equal(torch::abs(_circle-input[0]), 1E-1).any().item<bool>() )
    //                             //--------------------------
    //                             if(torch::abs(_circle-torch::pow(input[-1][2],2)).any().item<double>() <  1E-1){
    //                                 //--------------------------
    //                                 return torch::tensor(0.2);
    //                                 //--------------------------
    //                             }// end if( _circle.equal(input[0]) or torch::less_equal(torch::abs(_circle-input[0]), 1E-1).any().item<bool>() )
    //                             //--------------------------
    //                             if(torch::abs(_circle-torch::pow(input[-1][2],2)).any().item<double>() >=  1E-1){
    //                                 //--------------------------
    //                                 return torch::tensor(0.1);
    //                                 //--------------------------
    //                             }// end if( _circle.equal(input[0]) or torch::less_equal(torch::abs(_circle-input[0]), 1E-1).any().item<bool>()  )
    //                             //--------------------------
    //                             return torch::tensor(2);
    //                             //--------------------------
    //                             };
    //--------------------------------------------------------------
    // auto _circle_reward = [](const torch::Tensor& input, const torch::Tensor& output, const size_t& batch){
    //     //--------------------------
    //     // return torch::abs((torch::pow((output[-1][0]- input[-1][0]),2)+ (torch::pow((output[-1][1]-input[-1][1]),2))) - input[-1][2]);
    //     //--------------------------
    //     // return torch::mse_loss((torch::pow((output[-1][0]- input[-1][0]),2)+ (torch::pow((output[-1][1]-input[-1][1]),2))), input[-1][2], torch::Reduction::Sum);
    //     //--------------------------
    //     // torch::Tensor _input = input, _output = output;
    //     //--------------------------
    //     // std::cout << "output.slice(1,0,1): " << output.slice(1,0,1) << " input.slice(1,0,1): " << input.slice(1,0,1) << std::endl;
    //     //--------------------------
    //     // return torch::mse_loss((torch::pow((output.slice(1,0,1).slice(0,0,10)- input.slice(1,0,1)),2)+ (torch::pow((output.slice(1,1,2).slice(0,0,10)-input.slice(1,1,2)),2))), input.slice(1,2,3));
    //     //--------------------------
    //     return (torch::pow((output.slice(1,0,1).slice(0,0,batch)- input.slice(1,0,1)),2)+ (torch::pow((output.slice(1,1,2).slice(0,0,batch)-input.slice(1,1,2)),2)))- input.slice(1,2,3);
    //     //--------------------------
    // };
    //--------------------------------------------------------------
    auto _circle_reward = [](const torch::Tensor& input, const torch::Tensor& output){
        //--------------------------
        return (torch::pow((output.slice(1,0,1) - input.slice(1,0,1)),2)+ (torch::pow((output.slice(1,1,2)-input.slice(1,1,2)),2)))- input.slice(1,2,3);
        //--------------------------
    };
    //--------------------------------------------------------------
    // RLEnvironment<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, size_t> _environment(std::move(input), _circle_reward, 0.9, 0.02, 500., batch_size);
    //--------------------------
    // RLEnvironment<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> _environment(std::move(input), _circle_reward, 0.9, 0.02, 500., 0);
    //--------------------------
    RLEnvironment<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> _environment(std::move(input), _circle_reward, 0.9, 0.02, 500., batch_size);
    //--------------------------
    // RLNetLSTM model({points_size, batch_size}, output_size, device);
    RLNet model(points_size, output_size);
    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(1E-3L).momentum(0.95).nesterov(true));
    //--------------------------
    ReinforcementNetworkHandling<decltype(model), size_t, size_t> handler(  std::move(model), 
                                                                            device_type, 
                                                                            [&_generate](size_t size = 10, size_t col = 2){ 
                                                                                return  _generate.get_output(size, col);});
    //--------------------------
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution memory_activation(0.50);
    //--------------------------
    ExperienceReplay memory(capacity);
    //--------------------------
    std::vector<torch::Tensor> _rewards;
    _rewards.reserve( input.size() * epoch);
    //--------------------------
    // auto done = std::make_unique<bool>(false);
    // auto epsilon = std::make_unique<double>(0.);
    //--------------------------
    // bool *done = nullptr;
    // double *epsilon = nullptr;
    //--------------------------
    progressbar bar(epoch);
    //--------------------------
    for(size_t i = 0; i < epoch; ++i){
        //--------------------------
        bool done = false;
        double epsilon = 0.;
        //--------------------------
        auto _input = _environment.get_first(epsilon);
        //--------------------------
        auto output = handler.action(_input, epsilon, batch_size, output_size);
        //--------------------------
        // auto [next_input, reward] = _environment.step(epsilon, done, _input, _normalize.normalization(output), batch_size);
        //--------------------------
        auto [next_input, reward] = _environment.step(epsilon, done, _input, _normalize.normalization(output));
        //--------------------------
        // std::cout << "reward: " << reward << std::endl;
        //--------------------------
        handler.agent(_input, optimizer, reward, done);
        //--------------------------
        memory.push(_input, next_input, reward, done);
        //--------------------------
        torch::Tensor training_input = next_input;
        //--------------------------
        _rewards.push_back(reward);
        //--------------------------
        // const auto log_thread = std::async(std::launch::async, [&_rewards, &reward, &_mutex](){
        //                                     std::lock_guard<std::mutex> guard(_mutex);
        //                                     _rewards.push_back(reward.item<float>());});
        //--------------------------
        while(!done){
            //--------------------------
            auto output = handler.action(training_input, epsilon, batch_size, output_size);
            //--------------------------
            // auto [next_input, reward] = _environment.step(epsilon, done, training_input,  _normalize.normalization(output), batch_size);
            //--------------------------
            auto [next_input, reward] = _environment.step(epsilon, done, training_input,  _normalize.normalization(output));
            //--------------------------
            // std::cout << "reward: " << reward << std::endl;
            //--------------------------
            memory.push(training_input, next_input, reward, done);
            //--------------------------
            // if (memory.size() < capacity){
            //     //--------------------------
            //     training_input = next_input;
            //     //--------------------------
            //     continue;
            //     //--------------------------
            // }// end if (memory.size() < capacity)
            //--------------------------
            try{
                //--------------------------
                if(memory_activation(gen)){
                    //--------------------------
                    auto [_memory_input, _memory_next_input, _memory_reward, _done] = memory.sample();
                    //--------------------------
                    handler.agent(_memory_input, _memory_next_input, optimizer, _memory_reward, _done);
                    //--------------------------
                }//end if(memory_activation(gen))
                else{
                    //--------------------------
                    handler.agent(training_input, next_input, optimizer, reward, done);
                    //--------------------------
                }// end else
            }// end try
            catch(std::overflow_error& e) {
                //--------------------------
                std::cerr << "\n" << e.what() << std::endl;
                //--------------------------
                std::exit(-1);
                //--------------------------
            }// end catch(std::out_of_range& e)
            // //--------------------------
            training_input = next_input;
            //--------------------------
            _rewards.push_back(reward);
            //--------------------------
            // const auto log_thread = std::async(std::launch::async, [&_rewards, &reward, &_mutex](){
            //                                 std::lock_guard<std::mutex> guard(_mutex);
            //                                 _rewards.push_back(reward.item<float>());});
            //--------------------------
        }// end while(!_done)
        //--------------------------
        _environment.reset();
        //------------
        bar.update();
        //--------------------------
    }//end for(size_t i = 0; i < epoch; ++i)
    //--------------------------
    /*
    size_t j = 0;
    for(size_t i = 0; i < epoch; ++i){
        //--------------------------
        bool done = false;
        double epsilon = 0.;
        //--------------------------
        auto _input = _environment.get_first(epsilon, batch_size);
        //--------------------------
        auto output = handler.action(_input, epsilon, 1, output_size);
        //--------------------------
        auto [next_input, reward] = _environment.step(epsilon, done, _input, _normalize.normalization(output));
        //--------------------------
        handler.agent(_input, optimizer, reward, done);
        //--------------------------
        memory.push(_input, next_input, reward, done);
        //--------------------------
        torch::Tensor training_input = next_input;
        //--------------------------
        _rewards.push_back(reward);
        //--------------------------
        while(!done){
            //--------------------------
            auto output = handler.action(training_input, epsilon, 1, output_size);
            //--------------------------
            auto [next_input, reward] = _environment.step(epsilon, done, training_input,  _normalize.normalization(output));
            //--------------------------
            // std::cout   << "training_input: " << training_input.sizes() 
            //             << " next_input: " << next_input.sizes() 
            //             << " output: " << output.sizes()
            //             << " reward: " << reward.sizes() << std::endl;
            //--------------------------
            memory.push(training_input, next_input, reward, done);
            //--------------------------
            if (memory.size() < capacity){
                //--------------------------
                training_input = next_input;
                //--------------------------
                continue;
                //--------------------------
            }// end if (memory.size() < capacity)
            //--------------------------
            try{
                //--------------------------
                auto [training_input, next_input, reward, _memory_done] = memory.sample(batch_size, done);
                //--------------------------
                handler.agent(training_input, next_input, optimizer, reward, _memory_done);
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
            _rewards.push_back(reward);
            //--------------------------
            // std::cout << "[" << ++j << "]" << " done: [" << std::boolalpha << done << "]" << std::endl;
            //--------------------------
        }// end while(!_done)
        //--------------------------
        _environment.reset();
        //------------
        bar.update();
        //--------------------------
    }//end for(size_t i = 0; i < epoch; ++i)
    */
    //--------------------------
    std::cout << " \n rewards: " << _rewards.size() << std::endl;
    //--------------------------
    auto [input_test, t_min, t_max] = input_test_thread.get();
    //--------------------------
    std::vector<torch::Tensor> _output_test;
    _output_test.reserve(input_test.size());
    //--------------------------
    std::cout << "--------------TEST--------------" << std::endl;
    //--------------------------
    for(const auto& _test : input_test){
        //--------------------------
        auto _test_temp = handler.test(_test);
        //--------------------------
        _output_test.push_back(_test_temp);
        //--------------------------
        // auto _circle = torch::pow((_test_temp[-1][0]-_test[-1][0]),2)+torch::pow((_test_temp[-1][1]-_test[-1][1]),2);
        //--------------------------
        // auto _circle = (_test_temp[-1][0]-_test[-1][0]) + (_test_temp[-1][0]-_test[-1][0]);
        //--------------------------
        // auto _lost = torch::mse_loss(_circle, _test[-1][2], torch::Reduction::Sum).template item<float>();
        //--------------------------
        // std::cout   << "circle: " << _circle.item().toFloat()
        //             << " actual: " << _test[-1][2].item().toFloat()
        //             << " error: " << _lost*100 << std::endl;
        //--------------------------
        // std::cout << _test << _output_test << std::endl;
        //--------------------------
        auto _circle = torch::pow((_test_temp.slice(1,0,1) - _test.slice(1,0,1)),2)+ (torch::pow((_test_temp.slice(1,1,2)-_test.slice(1,1,2)),2));
        //--------------------------
        auto _lost = torch::mse_loss(_circle, _test.slice(1,2,3), torch::Reduction::Sum).template item<float>();
        //--------------------------
        std::cout  << " error: " << _lost*100 << std::endl;
        //--------------------------
    }// end for(const auto& _test : _tests)
    //--------------------------
    return 0;
    //--------------------------------------------------------------
}// end int main(void)
