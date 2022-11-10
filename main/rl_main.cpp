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
int main(void){
    //--------------------------
    RLGenerate _generate(100, 10);
    //--------------------------
    Normalize _normalize(_generate.get_input());
    //--------------------------
    // std::cout << "generation: " << std::endl;
    auto input = _normalize.vnormalization();
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
    auto _circle_reward = [](const torch::Tensor& input, const torch::Tensor& output){
                                //--------------------------
                                // std::cout << "output[0]: " << output[-1][0] << " input[1]: " << input[-1][1] << std::endl;
                                //-------------------------- 
                                auto _circle = torch::pow((output[-1][0]-input[-1][1]),2)+torch::pow((output[-1][1]-input[-1][2]),2);
                                //--------------------------
                                // std::cout << "_circle: " << _circle.item<double>() << " input[-1][0]: " << input[-1][0].item<double>() << std::endl;
                                //--------------------------
                                if( _circle.equal(input[-1][0])){
                                    //--------------------------
                                    std::cout << "_circle.equal" << "_circle: " << _circle.item<double>() << " input[-1][0]: " << input[-1][0].item<double>() << std::endl;
                                    //--------------------------
                                    return torch::tensor(10);
                                    //--------------------------
                                }// end if( _circle.equal(input[0]) or torch::less_equal(torch::abs(_circle-input[0]), 1E-1).any().item<bool>() )
                                //--------------------------
                                if(torch::abs(_circle-input[-1][0]).any().item<double>() <= 1E-4){
                                    //--------------------------
                                    std::cout << "torch::less_equal [1E-4]: " << "_circle: " << _circle.item<double>() << " input[-1][0]: " << input[-1][0].item<double>() << std::endl;
                                    //--------------------------
                                    return torch::tensor(5);
                                    //--------------------------
                                }// end if( _circle.equal(input[0]) or torch::less_equal(torch::abs(_circle-input[0]), 1E-1).any().item<bool>() )
                                //--------------------------
                                if(torch::abs(_circle-input[-1][0]).any().item<double>() <  1E-2){
                                    //--------------------------
                                    std::cout << "torch::less_equal [1E-2]: " << "_circle: " << _circle.item<double>() << " input[-1][0]: " << input[-1][0].item<double>() << std::endl;
                                    //--------------------------
                                    return torch::tensor(-1);
                                    //--------------------------
                                }// end if( _circle.equal(input[0]) or torch::less_equal(torch::abs(_circle-input[0]), 1E-1).any().item<bool>() )
                                //--------------------------
                                if(torch::abs(_circle-input[-1][0]).any().item<double>() <  1E-1){
                                    //--------------------------
                                    std::cout << "torch::less_equal [1E-1]: " << "_circle: " << _circle.item<double>() << " input[-1][0]: " << input[-1][0].item<double>() << std::endl;
                                    //--------------------------
                                    return torch::tensor(-5);
                                    //--------------------------
                                }// end if( _circle.equal(input[0]) or torch::less_equal(torch::abs(_circle-input[0]), 1E-1).any().item<bool>() )
                                //--------------------------
                                if(torch::abs(_circle-input[-1][0]).any().item<double>() >=  1E-1){
                                    //--------------------------
                                    std::cout << "torch::greater [1E-1]: " << "_circle: " << _circle.item<double>() << " input[-1][0]: " << input[-1][0].item<double>() << std::endl;
                                    //--------------------------
                                    return torch::tensor(-10);
                                    //--------------------------
                                }// end if( _circle.equal(input[0]) or torch::less_equal(torch::abs(_circle-input[0]), 1E-1).any().item<bool>()  )
                                // //--------------------------
                                std::cout << "default: " << "_circle: " << _circle.item<double>() << " input[-1][0]: " << input[-1][0].item<double>() << std::endl;
                                //--------------------------
                                return torch::tensor(0);
                                //--------------------------
                                };
    RLEnvironment<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> _environment(std::move(input), _circle_reward);
    //--------------------------
    RLNet model(3, 2);
    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(1E-1L).momentum(0.95).nesterov(true));
    //--------------------------
    ReinforcementNetworkHandling<RLNet, size_t, size_t> handler(    std::move(model), 
                                                                    torch::kCPU, 
                                                                    [&_generate](size_t size = 1, size_t col = 2){ return  _generate.get_output(size, col);});
    //--------------------------
    bool _done = false;
    //--------------------------
    std::vector<torch::Tensor> _rewards;
    _rewards.reserve( input.size() *100000);
    torch::Tensor training_input;
    double _epsilon = 0.;
    //--------------------------
    // torch::Tensor tensor = torch::rand({10,10});
    // std::cout<< "Tensor " << tensor[1]<< std::endl;
    //--------------------------
    // auto x = _generate.get_output(1, 2);
    // std::cout<< "Tensor " << x << std::endl;
    //--------------------------
    for(size_t i = 0; i < 100000; ++i){
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
        handler.agent(_input, next_input, optimizer, reward, done);
        //--------------------------
        // std::cout << "_input:" << _input.sizes() << " next_input" << next_input.sizes() << std::endl;
        //--------------------------
        training_input = next_input;
        _epsilon = init_epsilon;
        //--------------------------
        // std::cout << "_input:" << _input.sizes() << " training_input" << training_input.sizes() << std::endl;
        //--------------------------
        // std::cout << "first loop" << std::endl;
        //--------------------------
        _done = done;
        //--------------------------
        _rewards.push_back(reward);
        //--------------------------
        while(!_done){
            //--------------------------
            // std::cout << "loop" << std::endl;
            //--------------------------
            auto output = handler.action(training_input, _epsilon, 1, 2);
            //--------------------------
            auto [next_input, reward, epsilon, done] = _environment.step(training_input,  _normalize.vnormalization(output));
            //--------------------------
            _epsilon = epsilon;
            _done = done;
            //--------------------------
            // std::cout << "training_input:" << training_input.sizes() << " next_input: " << next_input.sizes() << std::endl;
            //--------------------------
            handler.agent(training_input, next_input, optimizer, reward, done);
            //--------------------------
            training_input = next_input;
            //--------------------------
            _rewards.push_back(reward);
            //--------------------------
            if(reward.item<double>() > 40.f){ 
                std::cout << "reward:[" << reward.item<double>() << "]" << std::endl;
            }
            //--------------------------
            // std::cout << "_input:" << training_input.sizes() << " training_input" << next_input.sizes() << std::endl;
            //--------------------------
            // std::cout << "[" << i << "] input[" << input << "] reward[" << reward << "] epsilon[" << epsilon << "] done[" << done << "]" << std::endl;
            //--------------------------
            // std::cout << "reward[" << reward << "]" << std::endl;
            //--------------------------
            // ++i;
        }// end while(!_done)
        //--------------------------
        _environment.reset();
        //--------------------------
    }//end for(size_t i = 0; i < 10000; ++i)
    //--------------------------
    // std::cout << "_rewards: " << _rewards.size() << std::endl;
    // for(const auto &x : _rewards){
    //     if(x.item<float>() == 100.f){ 
    //         std::cout << "_rewards:["<< x.item<float>() << "] " ; 
    //     }
    // }// end for(const auto &x : _rewards)
    //--------------------------
    // _environment.set_reward_function(func);
    //--------------------------------------------------------------
    return 0;
    //--------------------------
}// end int main(void)
