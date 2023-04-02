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
#include "Generate/RL/TestEnvironment.hpp"
#include "Generate/RL/RLGenerate.hpp"
// #include "Network/RL/ReinforcementNetworkHandling.hpp"
//--------------------------
#include "Network/RL/ReinforcementNetworkHandlingDQN.hpp"
//--------------------------
#include "Network/RL/RLNormalize.hpp"
#include "Network/RL/ExperienceReplay.hpp"
//--------------------------
#include "Timing/TimeIT.hpp"
//--------------------------------------------------------------
// Standard cpp library
//--------------------------------------------------------------
#include <future>
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
int main(int argc, char const *argv[]){
    //--------------------------------------------------------------
    // Command line arugments using boost options 
    //--------------------------
    std::string filename;
    //--------------------------
    size_t  generated_size,
            batch_size,
            test_size,
            epoch,
            points_size,
            output_size,
            capacity,
            limiter,
            update_frequency;
    //--------------------------
    double memory_percentage;
    //--------------------------
    bool clamp, double_mode;
    //--------------------------
    boost::program_options::options_description description("Allowed options:");
    //--------------------------
    description.add_options()
    ("help,h", "Display this help message")
    ("filename,s", boost::program_options::value<std::string>(&filename)->default_value("test_results"), "Name of the file saved")
    ("generated_size,g", boost::program_options::value<size_t>(&generated_size)->default_value(100000), "How many different points to train. Accepts an integer x > 0")
    ("batch_size,b", boost::program_options::value<size_t>(&batch_size)->default_value(1000), "Batch the generated data to train. Limitations: - Must be less then the generated_size - Must be less the 1000 this a libtorch limitation")
    ("test_size,t", boost::program_options::value<size_t>(&test_size)->default_value(3000), "How many points generated. Accepts an integer x >= 100")
    ("epoch,e", boost::program_options::value<size_t>(&epoch)->default_value(5000), "How many iterations to train")
    ("points_size,p", boost::program_options::value<size_t>(&points_size)->default_value(3), "Determine when to stop the training. This uses a validation set")
    ("output_size,o", boost::program_options::value<size_t>(&output_size)->default_value(2), "Determine when to stop the training. This uses a validation set")
    ("capacity,c", boost::program_options::value<size_t>(&capacity)->default_value(3000), "Determine when to stop the training. This uses a validation set")
    ("limiter,l", boost::program_options::value<size_t>(&limiter)->default_value(10), "Determine when to stop the training. This uses a validation set")
    ("update_frequency,f", boost::program_options::value<size_t>(&update_frequency)->default_value(100), "Determine when to stop the training. This uses a validation set")
    ("memory_percentage,m", boost::program_options::value<double>(&memory_percentage)->default_value(0.3), "Determine when to stop the training. This uses a validation set")
    ("double_mode,d", boost::program_options::value<bool>(&double_mode)->default_value(true), "false: validation precision or true: Train with an epoch iteration")
    ("clamp,u", boost::program_options::value<bool>(&clamp)->default_value(true), "false: validation precision or true: Train with an epoch iteration");
    //--------------------------
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(description).run(), vm);
    boost::program_options::notify(vm);
    //--------------------------
    // Add protection to the values
    //-----------
    if (vm.count("help")){
        //--------------------------
        std::cout << description;
        //--------------------------
        std::exit(0);
        //--------------------------
    }// end if (vm.count("help"))
    //-----------
    if (vm.count("filename")){
        filename = vm["filename"].as<std::string>() + std::string(".csv");
    }// end if (vm.count("filename"))
    //-----------
    if (vm["generated_size"].as<size_t>() < 0){
        throw std::out_of_range("Must be at least postive");
    }// end if (vm.count("training_size") < 0)
    //-----------
    if (vm["batch_size"].as<size_t>() < 0){
        throw std::out_of_range("Must be at least postive");
    }// end if (vm.count("generated_size") < 100)
    //-----------
    if (vm["batch_size"].as<size_t>() > (vm["generated_size"].as<size_t>()/2)){
        throw std::out_of_range("Batch size must be less then half of the generated size");
    }// end if (vm.count("generated_size") < 100)
    //-----------
    if (vm["test_size"].as<size_t>() < 0){
        throw std::out_of_range("Must be at least postive or less then generated size or less then 1000 (x <= 1000)");
    }// end  if (vm.count("batch_size") < 0 and vm.count("batch_size") > static_cast<int>(generated_size) and vm.count("batch_size") > 1000)
    //-----------
    if (vm["epoch"].as<size_t>() < 0){
        throw std::out_of_range("Must be at least postive");
    }// end if (vm.count("epoch") < 0)
    //-----------
    if (vm["points_size"].as<size_t>() < 0){
        throw std::out_of_range("Must be at least postive");
    }// end if (vm.count("epoch") < 0)
    //-----------
    if (vm["output_size"].as<size_t>() < 0){
        throw std::out_of_range("Must be at least postive");
    }// end if (vm.count("epoch") < 0)
    //-----------
    if (vm["capacity"].as<size_t>() < vm["batch_size"].as<size_t>()){
        throw std::out_of_range("Must higher or equal to the batch size[" + vm["batch_size"].as<std::string>() + "]");
    }// end if (vm.count("epoch") < 0)
    //-----------
    if (vm["limiter"].as<size_t>() < 0){
        throw std::out_of_range("Must be at least postive");
    }// end if (vm.count("epoch") < 0)
    //-----------
    if (vm["update_frequency"].as<size_t>() < 0){
        throw std::out_of_range("Must be at least postive");
    }// end if (vm.count("epoch") < 0)
    //-----------
    if (vm["memory_percentage"].as<double>() < 0. and vm["memory_percentage"].as<double>() > 1.){
        throw std::out_of_range("Must between 0 and 1");
    }// end if (vm.count("precision") < 0)
    //--------------------------------------------------------------
    std::cout   << "-------------------------------[Input Info]-------------------------------\n"
                << "filename:       " << filename << "\n" 
                << "generated size: " << generated_size << "\n"  
                << "test size:      " << test_size << "\n"
                << "batch size:     " << batch_size << "\n"
                << "epoch:          " << epoch << "\n"
                << "clamp:          " << std::boolalpha << clamp << "\n"
                << "double_mode:    " << std::boolalpha << double_mode << std::endl;
    //--------------------------------------------------------------
    // Initiate Torch seed, device type
    //--------------------------
    torch::manual_seed(17);
    //--------------------------
    torch::DeviceType device_type;
    //--------------------------
    std::cout   << "-------------------------------[Training Type]-------------------------------" << std::endl;
    //-----------
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
    std::cout   << "-------------------------------[Starting Timing]-------------------------------" << std::endl;
    //-----------
    TimeIT _timer_tester;
    //-----------
    RLGenerate _generate(generated_size, test_size, points_size, limiter);
    //-----------
    std::cout << "RLGenerate time:                  " << _timer_tester.get_time_seconds() << std::endl;
    // auto _temp = _generate.get_input();
    // for(const auto& x : _temp){
    //     std::cout << "data: " << x << std::endl;
    // }
    // std::cout << "---------------end data:["<< _timer_tester.get_time_seconds() << "]------------ " << std::endl;
    // std::exit(1);
    //--------------------------
    RLNormalize _normalize(_generate.get_input());
    //-----------
    std::cout << "RLGenerate and RLNormalize time:  " << _timer_tester.get_time_seconds() << std::endl;
    // std::exit(1);
    //--------------------------
    auto input = _normalize.normalization();
    //-----------
    // for(const auto& x : input){
    //     std::cout << "normalize data: " << _normalize.unnormalization(x) << std::endl;
    // }
    std::cout << "Input RLNormalize time:           " << _timer_tester.get_time_seconds() << std::endl;
    // std::exit(1);
    //--------------------------
    auto input_test_thread = std::async(std::launch::async, [&_generate](){return RLNormalize::normalization_min_max(_generate.get_test_input());});
    //--------------------------------------------------------------
    auto _circle_reward = [](const torch::Tensor& input, const torch::Tensor& output){
        //--------------------------
        return (torch::pow((output.slice(1,0,1) - input.slice(1,0,1)),2)+ (torch::pow((output.slice(1,1,2)-input.slice(1,1,2)),2)))- input.slice(1,2,3);
        //--------------------------
    };
    //--------------------------------------------------------------
    RLEnvironment<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> _environment(std::move(input), _circle_reward, 0.9, 0.02, 500., batch_size);
    //--------------------------
    // RLNetLSTM model({points_size, batch_size}, output_size, device);
    // RLNetLSTM target_model({points_size, batch_size}, output_size, device);
    //--------------------------
    // RLNet model(points_size, output_size);
    // RLNet target_model(points_size, output_size);
    //--------------------------
    DuelNet model(points_size, output_size);
    DuelNet target_model(points_size, output_size);
    //--------------------------
    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(1E-3L).momentum(0.95).nesterov(true));
    //--------------------------
    // ReinforcementNetworkHandling<decltype(model), size_t, size_t> handler(  std::move(model), 
    //                                                                         [&_generate](const size_t& size = 1, const size_t& col = 2){ 
    //                                                                             return  _generate.get_output(size, col);});
    //--------------------------
    ReinforcementNetworkHandlingDQN<decltype(model), size_t, size_t> handler(   std::move(model), 
                                                                                std::move(target_model),
                                                                                update_frequency,
                                                                                clamp,
                                                                                double_mode, 
                                                                                [&_generate](const size_t& size = 1, const size_t& col = 2){ 
                                                                                    return  _generate.get_output(size, col);});
    //--------------------------
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution memory_activation(memory_percentage);
    //--------------------------
    ExperienceReplay<torch::Tensor, torch::Tensor, torch::Tensor, bool> memory(capacity);
    //--------------------------
    std::vector<torch::Tensor> _rewards;
    _rewards.reserve( input.size() * epoch);
    //--------------------------
    // std::cout << "final time: " << _timer_tester.get_time_seconds() << std::endl;
    // std::exit(1);
    //--------------------------
    progressbar bar(epoch);
    //--------------------------------------------------------------
    // for(size_t i = 0; i < epoch; ++i){
    //     //--------------------------
    //     bool done = false;
    //     double epsilon = 0.;
    //     //--------------------------
    //     auto _input = _environment.get_first(epsilon);
    //     //--------------------------
    //     auto output = handler.action(_input, epsilon, batch_size, output_size);
    //     //--------------------------
    //     // auto [next_input, reward] = _environment.step(epsilon, done, _input, _normalize.normalization(output), batch_size);
    //     //--------------------------
    //     auto [next_input, reward] = _environment.step(epsilon, done, _input, _normalize.normalization(output));
    //     //--------------------------
    //     // std::cout << "reward: " << reward << std::endl;
    //     //--------------------------
    //     handler.agent(_input, optimizer, reward, done);
    //     //--------------------------
    //     memory.push(_input, next_input, reward, done);
    //     //--------------------------
    //     torch::Tensor training_input = next_input;
    //     //--------------------------
    //     _rewards.push_back(reward);
    //     //--------------------------
    //     while(!done){
    //         //--------------------------
    //         auto output = handler.action(training_input, epsilon, batch_size, output_size);
    //         //--------------------------
    //         auto [next_input, reward] = _environment.step(epsilon, done, training_input,  _normalize.normalization(output));
    //         //--------------------------
    //         memory.push(training_input, next_input, reward, done);
    //         //--------------------------
    //         try{
    //             //--------------------------
    //             if(memory_activation(gen)){
    //                 //--------------------------
    //                 auto [_memory_input, _memory_next_input, _memory_reward, _done] = memory.sample();
    //                 //--------------------------
    //                 handler.agent(_memory_input, _memory_next_input, optimizer, _memory_reward, _done);
    //                 //--------------------------
    //             }//end if(memory_activation(gen))
    //             else{
    //                 //--------------------------
    //                 handler.agent(training_input, next_input, optimizer, reward, done);
    //                 //--------------------------
    //             }// end else
    //         }// end try
    //         catch(std::overflow_error& e) {
    //             //--------------------------
    //             std::cerr << "\n" << e.what() << std::endl;
    //             //--------------------------
    //             std::exit(-1);
    //             //--------------------------
    //         }// end catch(std::out_of_range& e)
    //         // //--------------------------
    //         training_input = next_input;
    //         //--------------------------
    //         _rewards.push_back(reward);
    //         //--------------------------
    //     }// end while(!_done)
    //     //--------------------------
    //     _environment.reset();
    //     //------------
    //     bar.update();
    //     //--------------------------
    // }//end for(size_t i = 0; i < epoch; ++i)
    //--------------------------------------------------------------
    std::cout   << "-------------------------------[Training]-------------------------------" << std::endl;
    //-----------
    TimeIT _timer;
    //--------------------------
    for(size_t i = 0; i < epoch; ++i){
        //--------------------------
        bool done = false;
        double epsilon = 0.;
        //--------------------------
        auto _input = _environment.get_first(epsilon).to(device);
        //--------------------------
        auto output = handler.action(_input, epsilon, batch_size, output_size).to(device);
        //--------------------------
        auto [next_input, reward] = _environment.step(epsilon, done, _input, _normalize.normalization(output));
        //--------------------------
        handler.agent(_input, next_input.to(device), optimizer, reward, done);
        //--------------------------
        memory.push(_input, next_input, reward, done);
        //--------------------------
        torch::Tensor training_input = next_input;
        //--------------------------
        _rewards.push_back(reward);
        //--------------------------
        while(!done){
            //--------------------------
            auto output = handler.action(training_input, epsilon, batch_size, output_size);
            //--------------------------
            auto [next_input, reward] = _environment.step(epsilon, done, training_input,  _normalize.normalization(output));
            //--------------------------
            memory.push(training_input, next_input.to(device), reward, done);
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
            //--------------------------
            training_input = next_input;
            //--------------------------
            _rewards.push_back(reward);
            //--------------------------
        }// end while(!_done)
        //--------------------------
        _environment.reset();
        //------------
        bar.update();
        //--------------------------
    }//end for(size_t i = 0; i < epoch; ++i)
    //--------------------------
    //--------------------------------------------------------------
    // std::vector<std::thread> _threads(epoch);
    // _threads.reserve(std::thread::hardware_concurrency() - 1);
    //--------------------------
    // TimeIT _timer;
    //--------------------------
    /*
    for (size_t i = 0; i < epoch; ++i){
        //--------------------------
        for(size_t j = 0; j < std::thread::hardware_concurrency() - 1; ++j){
            //--------------------------
            _threads.emplace_back(std::move(train));
            //--------------------------
        }// end for(size_t j = 0; j < std::thread::hardware_concurrency() - 1; ++j)
        //--------------------------
        std::for_each(std::execution::par_unseq, _threads.begin(), _threads.end(), [](auto& _thread){_thread.join();});
        //--------------------------
        _threads.clear();
        //--------------------------
        bar.update();
        //--------------------------
    }// end for (size_t i = 0; i < epoch; ++i)
    */
    //--------------------------
    // std::generate(std::execution::par_unseq, _threads.begin(), _threads.end(), [&](){return std::thread(train);});
    // std::for_each(std::execution::par_unseq, _threads.begin(), _threads.end(), [](auto& _thread){_thread.join();});
    //--------------------------------------------------------------
    std::cout << "\n" << "Thread timer: " << _timer.get_time_seconds() << std::endl;
    //--------------------------
    std::cout << "\n" << "rewards: " << _rewards.size() << std::endl;
    //--------------------------
    auto [input_test, t_min, t_max] = input_test_thread.get();
    //--------------------------
    // std::vector<torch::Tensor> _output_test;
    // _output_test.reserve(input_test.size());
    //--------------------------
    std::cout << "--------------TEST--------------" << std::endl;
    //--------------------------
    // for(const auto& _test : input_test){
    //     //--------------------------
    //     auto _test_temp = handler.test(_test);
    //     //--------------------------
    //     _output_test.push_back(_test_temp);
    //     //--------------------------
    //     // auto _circle = torch::pow((_test_temp[-1][0]-_test[-1][0]),2)+torch::pow((_test_temp[-1][1]-_test[-1][1]),2);
    //     //--------------------------
    //     // auto _circle = (_test_temp[-1][0]-_test[-1][0]) + (_test_temp[-1][0]-_test[-1][0]);
    //     //--------------------------
    //     // auto _lost = torch::mse_loss(_circle, _test[-1][2], torch::Reduction::Sum).template item<float>();
    //     //--------------------------
    //     // std::cout   << "circle: " << _circle.item().toFloat()
    //     //             << " actual: " << _test[-1][2].item().toFloat()
    //     //             << " error: " << _lost*100 << std::endl;
    //     //--------------------------
    //     // std::cout << _test << _output_test << std::endl;
    //     //--------------------------
    //     auto _circle = torch::pow((_test_temp.slice(1,0,1) - _test.slice(1,0,1)),2)+ (torch::pow((_test_temp.slice(1,1,2)-_test.slice(1,1,2)),2));
    //     //--------------------------
    //     auto _lost = torch::mse_loss(_circle, _test.slice(1,2,3), torch::Reduction::Sum).item<float>();
    //     //--------------------------
    //     std::cout  << " error: " << _lost*100 << std::endl;
    //     //--------------------------
    // }// end for(const auto& _test : _tests)
    //--------------------------------------------------------------
    //--------------------------------------------------------------
    // Print table settup
    //--------------------------
    fort::char_table table;
    //--------------------------
    // Change border style
    //--------------------------
    table.set_border_style(FT_BASIC2_STYLE);
    //--------------------------
    // Set color
    //--------------------------
    table.row(0).set_cell_content_fg_color(fort::color::light_blue);
    //--------------------------
    // Set center alignment for the all columns
    //--------------------------
    table.column(0).set_cell_text_align(fort::text_align::center);
    table.column(1).set_cell_text_align(fort::text_align::center);
    table.column(2).set_cell_text_align(fort::text_align::center);
    table.column(3).set_cell_text_align(fort::text_align::center);
    table.column(4).set_cell_text_align(fort::text_align::center);
    table.column(5).set_cell_text_align(fort::text_align::center);
    //--------------------------
    table   << fort::header
            << "X_1" << "X" << "Y_1" << "Y" << "Original Target" << "Output" << "Loss" << fort::endr;
    //--------------------------------------------------------------
    TestEnvironment<torch::Tensor> _environment_test(std::move(input_test), batch_size);
    //--------------------------
    bool done{false};
    //--------------------------
    TimeIT _test_timer;
    //--------------------------
    while (!done){
        //--------------------------
        auto _test = _environment_test.step(done);
        //--------------------------
        auto _test_result = handler.test(_test);
        //--------------------------
        auto _circle = torch::pow((_test_result.slice(1,0,1) - _test.slice(1,0,1)),2)+ (torch::pow((_test_result.slice(1,1,2)-_test.slice(1,1,2)),2));
        //--------------------------
        // auto _loss = torch::mse_loss(_circle, _test.slice(1,2,3), torch::Reduction::Sum).item().toFloat();
        //--------------------------
        // std::cout << "done: " << std::boolalpha << done << " error: " << _loss*100 << std::endl;
        //--------------------------
        auto _loss = torch::mse_loss(_circle, _test.slice(1,2,3));
        //--------------------------
        // table   << RLNormalize::unnormalization(_test_result.slice(1,0,1), t_min, t_max) 
        //         << RLNormalize::unnormalization(_test.slice(1,0,1), t_min, t_max)
        //         << RLNormalize::unnormalization(_test_result.slice(1,1,2), t_min, t_max) 
        //         << RLNormalize::unnormalization(_test.slice(1,1,2), t_min, t_max)
        //         << RLNormalize::unnormalization(_test.slice(1,2,3), t_min, t_max)
        //         << RLNormalize::unnormalization(_circle, t_min, t_max)
        //         << _loss << fort::endr;
        //--------------------------
        table   << _test_result.slice(1,0,1)
                << _test.slice(1,0,1)
                << _test_result.slice(1,1,2)
                << _test.slice(1,1,2)
                << _test.slice(1,2,3)
                << _circle
                << _loss*100 << fort::endr;
        //--------------------------
    }// end while (!done)
    //--------------------------
    // std::for_each(std::execution::par_unseq, _test_threads.begin(), _test_threads.end(), [](auto& _thread){_thread.join();});
    //--------------------------
    TimeIT _print_timer;
    //--------------------------
    std::cout << "\n" << table.to_string() << std::endl;
    //--------------------------
    std::cout << "test time: " << _test_timer.get_time_seconds() << " print time: " << _print_timer.get_time_seconds() << std::endl;
    //--------------------------------------------------------------
    return 0;
    //--------------------------------------------------------------
}// end int main(void)
