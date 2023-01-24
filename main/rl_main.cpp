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
#include "Generate/RLEnvironment.hpp"
#include "Generate/RLGenerate.hpp"
#include "Network/ReinforcementNetworkHandling.hpp"
#include "Network/Network.hpp"
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
    //--------------------------
    RLGenerate _generate(1000, 3, 10);
    //--------------------------
    Normalize _normalize(_generate.get_input());
    //--------------------------
    // std::cout << "generation: " << std::endl;
    auto input = _normalize.vnormalization();
    auto input_test = _normalize.normalization(_generate.get_test_output(100, 3));
    //--------------------------
    // std::cout << "_generate input: " << input << std::endl;
    //--------------------------
    // std::cout << "_generate input: " << input.size() << " input tensor: " << input[0].sizes() << std::endl;
    //--------------------------
    // torch::Tensor min = torch::tensor(10), max = torch::tensor(0);
    //--------------------------
    // for(const auto& x : input){  
    //     torch::Tensor temp_min = torch::min(x), temp_max = torch::max(x);
    //     if(temp_min.less(min).any().item<bool>()){
    //         min = temp_min;
    //     }
    //     if(max.less(temp_max).any().item<bool>()){
    //         max = temp_max;
    //     }
    // };
    //--------------------------
    // std::for_each(std::execution::par_unseq, input.begin(), input.end(), [&](const auto& x){
    //                 torch::Tensor temp_min = torch::min(x), temp_max = torch::max(x);
    //                 if(temp_min.less(min).any().item<bool>()){
    //                     min = temp_min;
    //                 }
    //                 if(max.less(temp_max).any().item<bool>()){
    //                     max = temp_max;
    //                 }
    //                 });
    //--------------------------
    // std::cout << "it_max: " << max << " it_min: " << min << std::endl;
    //--------------------------
    // for(const auto& x : input){
    //     std::cout << x << std::endl;
    // }
    //--------------------------------------------------------------
    // // Generate your data set. At this point you can add transforms to you data set, e.g. stack your
    // // batches into a single tensor.
    // // auto data_set = DataLoader(std::move(std::get<0>(data)), std::move(std::get<1>(data))).map(torch::data::transforms::Normalize<>(0.5, 0.25)).map(torch::data::transforms::Stack<>());
    // //--------------------------
    // auto data_set = RLDataLoader(_generate.get_input()).map(torch::data::transforms::Stack<>());
    // //--------------------------
    // // Generate a data loader.
    // //--------------------------
    // auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>( std::move(data_set), 
    //                                                                                         torch::data::DataLoaderOptions(20));
    //--------------------------------------------------------------
    // RLEnvironment<decltype((_generate.get_input)()), double> _environment(_generate.get_input());
    //--------------------------------------------------------------
    auto _circle_reward = [](const torch::Tensor& input, const torch::Tensor& output){
                                //--------------------------
                                // std::cout << "output[0]: " << output[-1][0] << " input[1]: " << input[-1][1] << std::endl;
                                //-------------------------- 
                                auto _circle = torch::pow((output[-1][0]-input[-1][1]),2)+torch::pow((output[-1][1]-input[-1][2]),2);
                                //--------------------------
                                // std::cout << "_circle: " << _circle.item<double>() << " input[-1][0]: " << input[-1][0].item<double>() << std::endl;
                                //--------------------------
                                if( _circle.equal(torch::pow(input[-1][0],2))){
                                    //--------------------------
                                    auto printing_threads = std::async(std::launch::async,[&_circle, &input](){
                                        std::cout << "_circle.equal" << "_circle: " << _circle.item<double>() << " input[-1][0]: " << torch::pow(input[-1][0],2).item<double>() << std::endl;
                                    });
                                    //--------------------------
                                    return torch::tensor(0);
                                    //--------------------------
                                }// end if( _circle.equal(input[0]) or torch::less_equal(torch::abs(_circle-input[0]), 1E-1).any().item<bool>() )
                                //--------------------------
                                if(torch::abs(_circle-torch::pow(input[-1][0],2)).any().item<double>() <= 1E-4){
                                    //--------------------------
                                    auto printing_threads = std::async(std::launch::async,[&_circle, &input](){
                                        std::cout << "torch::less_equal [1E-4]: " << "_circle: " << _circle.item<double>() << " input[-1][0]: " << torch::pow(input[-1][0],2).item<double>() << std::endl;
                                    });
                                    //--------------------------
                                    return torch::tensor(0.1);
                                    //--------------------------
                                }// end if( _circle.equal(input[0]) or torch::less_equal(torch::abs(_circle-input[0]), 1E-1).any().item<bool>() )
                                //--------------------------
                                if(torch::abs(_circle-torch::pow(input[-1][0],2)).any().item<double>() <  1E-2){
                                    //--------------------------
                                    auto printing_threads = std::async(std::launch::async,[&_circle, &input](){
                                        std::cout << "torch::less_equal [1E-2]: " << "_circle: " << _circle.item<double>() << " input[-1][0]: " << torch::pow(input[-1][0],2).item<double>() << std::endl;
                                    });
                                    //--------------------------
                                    return torch::tensor(0.5);
                                    //--------------------------
                                }// end if( _circle.equal(input[0]) or torch::less_equal(torch::abs(_circle-input[0]), 1E-1).any().item<bool>() )
                                //--------------------------
                                if(torch::abs(_circle-torch::pow(input[-1][0],2)).any().item<double>() <  1E-1){
                                    //--------------------------
                                    auto printing_threads = std::async(std::launch::async,[&_circle, &input](){
                                        std::cout << "torch::less_equal [1E-1]: " << "_circle: " << _circle.item<double>() << " input[-1][0]: " << torch::pow(input[-1][0],2).item<double>() << std::endl;
                                    });
                                    //--------------------------
                                    return torch::tensor(0.2);
                                    //--------------------------
                                }// end if( _circle.equal(input[0]) or torch::less_equal(torch::abs(_circle-input[0]), 1E-1).any().item<bool>() )
                                //--------------------------
                                if(torch::abs(_circle-torch::pow(input[-1][0],2)).any().item<double>() >=  1E-1){
                                    //--------------------------
                                    // auto printing_threads = std::async(std::launch::async,[&_circle, &input](){
                                    //     std::cout << "torch::greater [1E-1]: " << "_circle: " << _circle.item<double>() << " input[-1][0]: " << torch::pow(input[-1][0],2).item<double>() << std::endl;
                                    // });
                                    //--------------------------
                                    return torch::tensor(0.1);
                                    //--------------------------
                                }// end if( _circle.equal(input[0]) or torch::less_equal(torch::abs(_circle-input[0]), 1E-1).any().item<bool>()  )
                                //--------------------------
                                auto printing_threads = std::async(std::launch::async,[&_circle, &input](){
                                    std::cout << "default: " << "_circle: " << _circle.item<double>() << " input[-1][0]: " << torch::pow(input[-1][0],2).item<double>() << std::endl;
                                });
                                //--------------------------
                                return torch::tensor(0);
                                //--------------------------
                                };
    //--------------------------------------------------------------
    // auto _circle_reward = [](const torch::Tensor& input, const torch::Tensor& output){
    //     //--------------------------
    //     auto _circle = torch::pow((output[-1][0]-input[-1][1]),2)+torch::pow((output[-1][1]-input[-1][2]),2);
    //     //--------------------------
    //     return torch::abs(_circle - torch::pow(input[-1][0],2));
    //     //--------------------------
    // };
    //--------------------------------------------------------------
    RLEnvironment<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> _environment(std::move(input), _circle_reward, 0.9, 0.02, 500., 10);
    //--------------------------
    RLNetLSTM model(3, 10);
    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(1E-3L).momentum(0.95).nesterov(true));
    //--------------------------
    ReinforcementNetworkHandling<RLNetLSTM, size_t, size_t> handler(    std::move(model), 
                                                                        torch::kCPU, 
                                                                        [&_generate](size_t size = 1, size_t col = 2){ return  _generate.get_output(size, col);});
    //--------------------------
    bool _done = false;
    //--------------------------
    std::vector<float> _rewards;
    _rewards.reserve( input.size() * 10);
    torch::Tensor training_input;
    double _epsilon = 0.;
    //--------------------------
    // torch::Tensor tensor = torch::rand({10,10});
    // std::cout<< "Tensor " << tensor[1]<< std::endl;
    //--------------------------
    // auto x = _generate.get_output(1, 2);
    // std::cout<< "Tensor " << x << std::endl;
    //--------------------------
    progressbar bar(10);
    //--------------------------
    for(size_t i = 0; i < 10; ++i){
        //--------------------------
        auto [_input, init_epsilon] = _environment.get_first();
        //--------------------------
        // std::cout << "_environment.get_first()" << std::endl;
        // std::cout << "_input: " << _input.sizes() << std::endl;
        //--------------------------
        auto output = handler.action(_input, init_epsilon, 1, 2);
        //--------------------------
        // std::cout << "_input: " << _input.sizes() << " handler.action size: " << output.sizes() << std::endl;
        // std::cout << "_input: " << _input << std::endl;
        //--------------------------
        auto [next_input, reward, epsilon, done] = _environment.step(_input, _normalize.vnormalization(output));
        //--------------------------
        // std::cout << "_input:" << _input.sizes() << " next_input" << next_input.sizes() << std::endl;
        //--------------------------
        // std::cout << "[" << i << "] input[" << next_input << "] reward[" << reward << "] epsilon[" << epsilon << "] done[" << done << "]" << std::endl;
        //--------------------------
        // std::cout << "environment.step: " << std::endl;
        //--------------------------
        // handler.agent(_input, next_input, optimizer, reward, done);
        //--------------------------
        handler.agent(_input, optimizer, reward, done);
        //--------------------------
        // std::cout << "_input:" << _input.sizes() << " next_input" << next_input.sizes() << std::endl;
        //--------------------------
        training_input = next_input;
        _epsilon = init_epsilon;
        //--------------------------
        // std::cout << "_input:" << _input.sizes() << " training_input" << training_input.sizes() << " next_input: " << next_input.sizes() << std::endl;
        //--------------------------
        // std::cout << "first loop" << std::endl;
        //--------------------------
        _done = done;
        //--------------------------
        _rewards.push_back(reward.item<float>());
        //--------------------------
        while(!_done){
            //--------------------------
            // std::cout << "loop" << std::endl;
            //--------------------------
            auto output = handler.action(training_input, _epsilon, 1, 2);
            //--------------------------
            // std::cout << "output: " << output.sizes() << std::endl;
            //--------------------------
            auto [next_input, reward, epsilon, done] = _environment.step(training_input,  _normalize.vnormalization(output));
            //--------------------------
            // std::cout << "training_input: " << training_input.sizes() << " next_input: " << next_input.sizes() << std::endl; 
            //--------------------------
            _epsilon = epsilon;
            _done = done;
            //--------------------------
            // std::cout   << "[0]" << "\n training_input:" << training_input.sizes() 
            //             << " next_input: " << next_input.sizes() 
            //             << " output: "  << output.sizes()  << std::endl;
            //--------------------------
            try{
                //--------------------------
                // handler.agent(training_input, next_input, optimizer, reward, done);
                //--------------------------
                handler.agent(training_input, next_input, optimizer, reward, done);
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
            // handler.agent(training_input, optimizer, reward, done);
            //--------------------------
            // std::cout << "\n [1]" << " training_input: " << training_input.sizes() << " next_input: " << next_input.sizes() << std::endl; 
            //--------------------------
            training_input = next_input;
            //--------------------------
            // std::cout << "[2]" << " training_input: " << training_input.sizes() << " next_input: " << next_input.sizes() << std::endl; 
            //--------------------------
            _rewards.push_back(reward.item<float>());
            //--------------------------
            // if(reward.item<double>() > 40.f){ 
            //     std::cout << "reward:[" << reward.item<double>() << "]" << std::endl;
            // }
            //--------------------------
            // std::cout << "_input:" << training_input.sizes() << " training_input" << next_input.sizes() << std::endl;
            //--------------------------
            // std::cout << "[" << i << "] input[" << input << "] reward[" << reward << "] epsilon[" << epsilon << "] done[" << done << "]" << std::endl;
            //--------------------------
            // std::cout << "reward[" << reward << "]" << std::endl;
            //--------------------------
            // std::cout << "done: " << _done << std::endl;
            //--------------------------
            // ++i;
        }// end while(!_done)
        //--------------------------
        bar.update();
        //------------
        _environment.reset();
        //--------------------------
    }//end for(size_t i = 0; i < 10000; ++i)
    //--------------------------
    std::cout << "_rewards: " << _rewards.size() << std::endl;
    //--------------------------
    // for(const auto &x : _rewards){
    //     if(x.item<float>() <= 0.5){ 
    //         std::cout << "_rewards:["<< x.item<float>() << "] " ; 
    //     }
    // }// end for(const auto &x : _rewards)
    //--------------------------
    // std::for_each(std::execution::par_unseq , _rewards.begin(), _rewards.end(), [&](const auto& x){ //if(x < 0.1f){ 
    //                                                                                                 //     std::cout << "\x1b[33m"<< "["<< x <<  "]\x1b[0m" << std::endl;
    //                                                                                                 // }//end if(x <= 0.5)
    //                                                                                                 std::cout << "[" << x << "]" << std::endl ;
    //                                                                                             });
    //--------------------------
    auto input_test_temp = input_test.data_ptr<float>();
    std::vector<float> _temp;
    _temp.reserve(input_test.size(0) * input_test.size(1));
    std::vector<torch::Tensor> _tests;
    _tests.reserve(input_test.size(0));
    std::vector<torch::Tensor> _output_test;
    _output_test.reserve(input_test.size(0));
    //--------------------------
    torch::Tensor _batching_output;
    //--------------------------
    for (int64_t i = 0; i < input_test.size(0); ++i){
        //--------------------------
        for (int64_t j = 0; j < input_test.size(1); ++j){ 
            //--------------------------
            _temp.push_back(*input_test_temp++);
            //--------------------------
        }// end for (int64_t j = 0; j < input_test.size(1); ++j)
        //--------------------------
        for (size_t i = 0; i < 10; ++i){
            //--------------------------
            auto _test_temp = torch::tensor(_temp).view({-1, static_cast<int64_t>(input_test.size(1))});
            //--------------------------
            if (i == 0){
                //--------------------------
                _batching_output = _test_temp;
                //--------------------------
                continue;
                //--------------------------
            }// end if (i == 0)
            //--------------------------
            _batching_output = torch::cat({_batching_output, _test_temp});
        }// end for (size_t i = 0; i < 10; ++i)
        //--------------------------
        _temp.clear();
        //--------------------------
        _tests.push_back(_batching_output);
        //--------------------------
    }// end for (int64_t i = 0; i < input_test.size(0); ++i)
    //--------------------------
    std::cout << "--------------INPUT--------------" << std::endl;
    //--------------------------
    for(const auto& _test : _tests){
        //--------------------------
        auto _test_temp = handler.test(_test);
        //--------------------------
        _output_test.push_back(_test_temp);
        //--------------------------
        auto _circle = torch::pow((_test_temp[-1][0]-_test[-1][1]),2)+torch::pow((_test_temp[-1][1]-_test[-1][2]),2);
        //--------------------------
        // auto results = torch::abs(_circle - torch::pow(_test[-1][0],2));
        //--------------------------
        auto _lost = torch::mse_loss(_circle, torch::pow(_test[-1][0],2), torch::Reduction::Sum).template item<float>();
        //--------------------------
        std::cout << "circle: " << _circle.item<float>() << " output: " << torch::pow(_test[-1][0],2).item<float>() << " error: " << _lost*100 << std::endl;
        //--------------------------
        // std::cout << _test << _output_test << std::endl;
        //--------------------------
    }// end for(const auto& _test : _tests)
    //--------------------------
    std::cout << "--------------OUTPUT--------------" << std::endl;
    // //--------------------------
    // for(const auto& x : _output_test){
    //     //--------------------------
    //     std::cout << x << std::endl;
    //     //--------------------------
    // }// end for(const auto& _test : _tests)
    // //--------------------------
    // std::vector<torch::Tensor> _output_test;
    // _rewards.reserve(input_test.size(0));
    // torch::Tensor training_input_test;
    // _epsilon = 0.;
    // //--------------------------
    // for (int64_t i = 0; i < input_test.size(0); ++i){
    //     //--------------------------
    //     if(i == 0){
    //         //--------------------------
    //         auto [_input, init_epsilon] = _environment.get_first();
    //         //--------------------------
    //         auto output = handler.action(_input, init_epsilon, 1, 2);
    //         //--------------------------
    //         auto [next_input, reward, epsilon, done] = _environment.step(_input, _normalize.vnormalization(output));
    //         //--------------------------
    //         _output_test.push_back(handler.test(_input));
    //         //--------------------------
    //         training_input = next_input;
    //         _epsilon = init_epsilon;
    //         //--------------------------
    //     }// end if(i == 0)
    //     //--------------------------
    //     auto output = handler.action(training_input, _epsilon, 1, 2);
    //     //--------------------------
    //     auto [next_input, reward, epsilon, done] = _environment.step(training_input,  _normalize.vnormalization(output));
    //     //--------------------------
    //     _epsilon = epsilon;
    //     //--------------------------
    //     _output_test.push_back(handler.test(training_input));
    //     //--------------------------
    // }// end for (size_t i = 0; i < input_test.size(0); ++i)
    //--------------------------
    // _environment.set_reward_function(func);
    //--------------------------------------------------------------
    return 0;
    //--------------------------
}// end int main(void)
