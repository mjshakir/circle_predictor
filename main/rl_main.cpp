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
//--------------------------------------------------------------
int main(void){
    //--------------------------
    RLGenerate _generate(100, 10);
    //--------------------------
    // std::cout << "generation: " << std::endl;
    auto input = _generate.get_data();
    // for(const auto& x : input ){
    //     std::cout << x << std::endl;
    // }
     //--------------------------------------------------------------
    // // Generate your data set. At this point you can add transforms to you data set, e.g. stack your
    // // batches into a single tensor.
    // // auto data_set = DataLoader(std::move(std::get<0>(data)), std::move(std::get<1>(data))).map(torch::data::transforms::Normalize<>(0.5, 0.25)).map(torch::data::transforms::Stack<>());
    // //--------------------------
    // auto data_set = RLDataLoader(_generate.get_data()).map(torch::data::transforms::Stack<>());
    // //--------------------------
    // // Generate a data loader.
    // //--------------------------
    // auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>( std::move(data_set), 
    //                                                                                         torch::data::DataLoaderOptions(20));
    //--------------------------------------------------------------
    // RLEnvironment<decltype((_generate.get_data)()), double> _environment(_generate.get_data());
    RLEnvironment<torch::Tensor, double, double> _environment(std::move(input), [](const double& i){return i + 4.f;});
    //--------------------------
    bool _done = false;
    size_t i = 0;
    //--------------------------
    while(!_done){
        //--------------------------
        // std::cout << "loop: " << std::endl;
        const auto [input, reward, done] = _environment.step(rand());
        //--------------------------
        _done = done;
        //--------------------------
        std::cout << "[" << i << "] input[" << input << "] reward[" << reward << "] done[" << done << "]" << std::endl;
        //--------------------------
        ++i;
    }// end for (size_t i = 0; i < 100; ++i
    
    //--------------------------
    // _environment.set_reward_function(func);
    //--------------------------------------------------------------
    return 0;
    //--------------------------
}// end int main(void)
