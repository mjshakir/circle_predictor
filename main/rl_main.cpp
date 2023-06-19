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
#include "Generate/RL/GeneratePoints.hpp"
// #include "Network/RL/ReinforcementNetworkHandling.hpp"
//--------------------------
#include "Environment/RL/Environment.hpp"
//--------------------------
#include "Network/RL/ReinforcementNetworkHandlingDQN.hpp"
//--------------------------
#include "Generate/RL/RLNormalize.hpp"
#include "Network/RL/ExperienceReplay.hpp"
//--------------------------
#include "Utilities/Utils.hpp"
//--------------------------
#include "Timing/Timing.hpp"
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
            generated_points_size,
            capacity,
            limiter,
            update_frequency;
    //--------------------------
    double memory_percentage;
    //--------------------------
    bool clamp, double_mode, verbos, randomizer;
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
    ("generated_points_size,s", boost::program_options::value<size_t>(&generated_points_size)->default_value(10), "Determine when to stop the training. This uses a validation set")
    ("capacity,c", boost::program_options::value<size_t>(&capacity)->default_value(3000), "Determine when to stop the training. This uses a validation set")
    ("limiter,l", boost::program_options::value<size_t>(&limiter)->default_value(10), "Determine when to stop the training. This uses a validation set")
    ("update_frequency,f", boost::program_options::value<size_t>(&update_frequency)->default_value(100), "Determine when to stop the training. This uses a validation set")
    ("memory_percentage,m", boost::program_options::value<double>(&memory_percentage)->default_value(0.3), "Determine when to stop the training. This uses a validation set")
    ("double_mode,d", boost::program_options::value<bool>(&double_mode)->default_value(true), "validation precision or true: Train with an epoch iteration")
    ("clamp,u", boost::program_options::value<bool>(&clamp)->default_value(false), "validation precision or true: Train with an epoch iteration")
    ("randomizer,r", boost::program_options::value<bool>(&randomizer)->default_value(false), "validation precision or true: Train with an epoch iteration")
    ("verbos,v", boost::program_options::value<bool>(&verbos)->default_value(false), "validation precision or true: Train with an epoch iteration");
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
        throw std::out_of_range("Batch Size: [" + std::to_string(vm["batch_size"].as<size_t>()) + 
                                "] Must Be Less Then Half of The data Size: [" + std::to_string(vm["generated_size"].as<size_t>()) + "]");
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
        throw std::out_of_range("Must higher or equal to the batch size[" + std::to_string(vm["batch_size"].as<size_t>()) + "]");
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
    // Print table settup
    //--------------------------
    fort::char_table info_table;
    //--------------------------
    if(verbos){
        //--------------------------
        // Change border style
        //--------------------------
        info_table.set_border_style( FT_BOLD2_STYLE);
        //--------------------------
        // Set color
        //--------------------------
        info_table.column(0).set_cell_content_fg_color(fort::color::light_red);
        info_table.column(0).set_cell_content_text_style(fort::text_style::bold);
        //--------------------------
        // Set center alignment for the all columns
        //--------------------------
        info_table.column(0).set_cell_text_align(fort::text_align::left);
        info_table.column(1).set_cell_text_align(fort::text_align::center);
        //--------------------------
        info_table  << "filename"       << filename                      << fort::endr;      
        info_table  << "generated size" << generated_size                << fort::endr;
        info_table  << "test size"      << test_size                     << fort::endr;
        info_table  << "batch size"     << batch_size                    << fort::endr;
        info_table  << "epoch"          << epoch                         << fort::endr;
        info_table  << "clamp"          << std::boolalpha << clamp       << fort::endr;
        info_table  << "double_mode"    << std::boolalpha << double_mode << fort::endr;
        info_table  << "randomizer"     << std::boolalpha << randomizer  << fort::endr;
        //--------------------------
    }// end if(verbos)
    //--------------------------------------------------------------
    // Initiate Torch seed, device type
    //--------------------------
    torch::manual_seed(17);
    //--------------------------
    torch::DeviceType device_type;
    //--------------------------
    if (torch::cuda::is_available()) {
        //--------------------------
        device_type = torch::kCUDA;
        torch::cuda::manual_seed(7);
        //--------------------------
        if(verbos){
            //--------------------------
            info_table  << "Training" << "GPU" << fort::endr;
            //--------------------------
        }// if(verbos)
    }// end if (torch::cuda::is_available()) 
    else {
        //--------------------------
        device_type = torch::kCPU;
        //--------------------------
        if(verbos){
            //--------------------------
            info_table  << "Training" << "CPU" << fort::endr;
            //--------------------------
        }// if(verbos)
    }// end else
    //--------------------------
    torch::Device device(device_type);
    //--------------------------------------------------------------
    Timing _full_timer(__FUNCTION__);
    //--------------------------
    TimeIT _timer_tester;
    //--------------------------
    RL::GeneratePoints _generate(generated_size, test_size, points_size, limiter);
    //--------------------------
    if(verbos){
        //--------------------------
        info_table  << "GeneratePoints time" << _timer_tester.get_time_seconds() << fort::endr;
        //--------------------------
    }// if(verbos)
    //--------------------------
    RL::RLNormalize _normalize(_generate.get_input());
    //--------------------------
    if(verbos){
        //--------------------------
        info_table  << "RL::RLNormalize time" << _timer_tester.get_time_seconds() << fort::endr;
        //--------------------------
    }// if(verbos)
    //--------------------------
    auto input = _normalize.normalization();
   //--------------------------
    if(verbos){
        //--------------------------
        info_table  << "Get normalized time" << _timer_tester.get_time_seconds() << fort::endr;
        //--------------------------
        std::cout << info_table.to_string() << std::endl;
        //--------------------------
    }// if(verbos)
    //--------------------------------------------------------------
    // std::exit(1);
    //--------------------------------------------------------------
    auto input_test_thread = std::async(std::launch::async, [&_generate](){return RL::RLNormalize::normalization_min_max(_generate.get_test_input());});
    //--------------------------------------------------------------
    // auto _circle_reward = [](const torch::Tensor& input, const torch::Tensor& output){
    //     //--------------------------
    //     return (torch::pow((output.slice(1,0,1) - input.slice(1,0,1)),2)+ (torch::pow((output.slice(1,1,2)-input.slice(1,1,2)),2)))- input.slice(1,2,3);
    //     //--------------------------
    // };
    //--------------------------------------------------------------
    // auto _circle_reward = [](const torch::Tensor& input, const torch::Tensor& output){
    //     //--------------------------
    //     return torch::abs(((torch::pow((output.slice(1,0,1) - input.slice(1,0,1)),2)+ (torch::pow((output.slice(1,1,2)-input.slice(1,1,2)),2)))- input.slice(1,2,3))*10);
    //     //--------------------------
    // };
    //--------------------------------------------------------------
    // auto _circle_reward = [](const torch::Tensor& input, const torch::Tensor& output){
    //     //--------------------------
    //     std::random_device rd;
    //     std::mt19937 gen(rd());
    //     std::bernoulli_distribution memory_activation(0.5);
    //     //--------------------------
    //     auto result = (torch::pow((output.slice(1,0,1) - input.slice(1,0,1)),2)+ (torch::pow((output.slice(1,1,2)-input.slice(1,1,2)),2)))- input.slice(1,2,3);
    //     //--------------------------
    //     if(memory_activation(gen)){
    //         //--------------------------
    //         return at::normal(torch::abs(result/2), torch::arange(result.size(1)));
    //         //--------------------------
    //     }// end if(memory_activation(gen))
    //     //--------------------------
    //     return at::normal((result/2), torch::arange(result.size(1)));
    //     //--------------------------
    // };
    //--------------------------------------------------------------
    // auto _circle_reward = [](const torch::Tensor& input, const torch::Tensor& output){
    //     //--------------------------
    //     std::random_device rd;
    //     std::mt19937 gen(rd());
    //     std::bernoulli_distribution memory_activation(0.5);
    //     //--------------------------
    //     if(memory_activation(gen)){
    //         //--------------------------
    //         return ((torch::pow((output.slice(1,0,1) - input.slice(1,0,1)),2)+ (torch::pow((output.slice(1,1,2)-input.slice(1,1,2)),2)))- input.slice(1,2,3))*10;
    //         //--------------------------
    //     }// end if(memory_activation(gen))
    //     //--------------------------
    //     return torch::abs(((torch::pow((output.slice(1,0,1) - input.slice(1,0,1)),2)+ (torch::pow((output.slice(1,1,2)-input.slice(1,1,2)),2)))- input.slice(1,2,3))*10);
    //     //--------------------------
    // };
    //--------------------------------------------------------------
    // auto _circle_reward = [](const torch::Tensor& input, const torch::Tensor& output){
    //     //--------------------------
    //     std::random_device rd;
    //     std::mt19937 gen(rd());
    //     std::bernoulli_distribution abs_activation(0.50);
    //     std::bernoulli_distribution normal_activation(0.25);
    //     //--------------------------
    //     if(abs_activation(gen)){
    //         //--------------------------
    //         auto _results = (torch::pow((output.slice(1,0,1) - input.slice(1,0,1)),2)+ (torch::pow((output.slice(1,1,2)-input.slice(1,1,2)),2)))- input.slice(1,2,3);
    //         //--------------------------
    //         if(normal_activation(gen)){
    //             //--------------------------
    //             return at::normal(_results/2, torch::arange(_results.size(1)));
    //             //--------------------------
    //         }// end if(normal_activation(gen))
    //         //--------------------------
    //         return _results;
    //         //--------------------------
    //     }// end if(abs_activation(gen))
    //     //--------------------------
    //     auto _results = torch::abs((torch::pow((output.slice(1,0,1) - input.slice(1,0,1)),2)+ (torch::pow((output.slice(1,1,2)-input.slice(1,1,2)),2)))- input.slice(1,2,3));
    //     //--------------------------
    //     if(normal_activation(gen)){
    //         //--------------------------
    //         return at::normal(_results/2, torch::arange(_results.size(1)));
    //         //--------------------------
    //     }// end if(normal_activation(gen))
    //     //--------------------------
    //     return _results;
    //     //--------------------------
    // };
    //--------------------------------------------------------------
    // auto _circle_reward = [](const torch::Tensor& input, const torch::Tensor& output){
    //     //--------------------------
    //     // std::random_device rd;
    //     // std::mt19937 gen(rd());
    //     // std::bernoulli_distribution memory_activation(0.5);
    //     //--------------------------
    //     auto _circle = torch::pow((output.slice(1,0,1) - input.slice(1,0,1)),2) + (torch::pow((output.slice(1,1,2)-input.slice(1,1,2)),2));
    //     //--------------------------
    //     auto _results = _circle - input.slice(1,2,3);
    //     //--------------------------
    //     auto _loss = torch::mse_loss(_circle, input.slice(1,2,3))*100;
    //     //--------------------------
    //     auto _results_meam = torch::mean(_results);
    //     //--------------------------
    //     // auto _X = torch::pow((output.slice(1,0,1) - input.slice(1,0,1)),2);
    //     // //--------------------------
    //     // auto _Y = torch::pow((output.slice(1,1,2) - input.slice(1,1,2)),2);
    //     // //--------------------------
    //     // auto _circle = _X + _Y;
    //     // //--------------------------
    //     // auto _results = _circle - input.slice(1,2,3);
    //     // //--------------------------
    //     // auto _loss = torch::mse_loss(_circle, input.slice(1,2,3))*100;
    //     // //--------------------------
    //     // auto _results_meam = torch::mean(_results);
    //     // //--------------------------
    //     // auto _final_results = torch::cat({_X, _Y}, 1);
    //     //--------------------------
    //     if(_results_meam.less(1E-2).item().toBool() and _results_meam.greater_equal(1E-3).item().toBool()){
    //         //--------------------------
    //         // if(memory_activation(gen)){
    //         //     //--------------------------
    //         //     return torch::abs(_results)*_loss*20;
    //         //     //--------------------------
    //         // }// end if(abs_activation(gen))
    //         // //--------------------------
    //         return _results*_loss*-20;
    //         //--------------------------
    //     }//end if(_results.less(1E-1).item().toBool())
    //     // //--------------------------
    //     if(_results_meam.less(1E-1).item().toBool() and _results_meam.greater_equal(1E-2).item().toBool()){
    //         //--------------------------
    //         // if(memory_activation(gen)){
    //         //     //--------------------------
    //         //     return torch::abs(_results)*_loss*50;
    //         //     //--------------------------
    //         // }// end if(abs_activation(gen))
    //         //--------------------------
    //         return _results*_loss*-10;
    //         //--------------------------
    //     }//end if(torch::mean(_results).less(1E-2).item().toBool())
    //     // //--------------------------
    //     if(_results_meam.less(1E0).item().toBool() and _results_meam.greater_equal(1E-1).item().toBool()){
    //         //--------------------------
    //         // if(memory_activation(gen)){
    //         //     //--------------------------
    //         //     return torch::abs(_results)*_loss*20;
    //         //     //--------------------------
    //         // }// end if(abs_activation(gen))
    //         //--------------------------
    //         return _results*_loss*-5;
    //         //--------------------------
    //     }//end if(torch::mean(_results).less(1E-2).item().toBool())
    //     // //--------------------------
    //     if(_results_meam.less(1E1).item().toBool() and _results_meam.greater_equal(1E-0).item().toBool()){
    //         //--------------------------
    //         // if(memory_activation(gen)){
    //         //     //--------------------------
    //         //     return torch::abs(_results)*_loss*10;
    //         //     //--------------------------
    //         // }// end if(abs_activation(gen))
    //         // //--------------------------
    //         return _results*_loss*-2;
    //         //--------------------------
    //     }//end if(torch::mean(_results).less(1E-2).item().toBool())
    //     //--------------------------
    //     // if(memory_activation(gen)){
    //     //     //--------------------------
    //     //     return torch::abs(_results)*_loss;
    //     //     //--------------------------
    //     // }// end if(abs_activation(gen))
    //     // //--------------------------
    //     return _results*_loss;
    //     //--------------------------
    // };
    //--------------------------------------------------------------
    // auto _circle_reward = [](const torch::Tensor& input, const torch::Tensor& output){
    //     //--------------------------
    //     std::random_device rd;
    //     std::mt19937 gen(rd());
    //     std::bernoulli_distribution memory_activation(0.5);
    //     //--------------------------
    //     auto _circle = torch::pow((output.slice(1,0,1) - input.slice(1,0,1)),2) + (torch::pow((output.slice(1,1,2)-input.slice(1,1,2)),2));
    //     //--------------------------
    //     auto _results = _circle - input.slice(1,2,3);
    //     //--------------------------
    //     auto _loss = torch::mse_loss(_circle, input.slice(1,2,3))*100;
    //     //--------------------------
    //     auto _results_meam = torch::mean(_results);
    //     //--------------------------
    //     // auto _X = torch::pow((output.slice(1,0,1) - input.slice(1,0,1)),2);
    //     // //--------------------------
    //     // auto _Y = torch::pow((output.slice(1,1,2) - input.slice(1,1,2)),2);
    //     // //--------------------------
    //     // auto _circle = _X + _Y;
    //     // //--------------------------
    //     // auto _results = _circle - input.slice(1,2,3);
    //     // //--------------------------
    //     // auto _loss = torch::mse_loss(_circle, input.slice(1,2,3))*100;
    //     // //--------------------------
    //     // auto _results_meam = torch::mean(_results);
    //     // //--------------------------
    //     // auto _final_results = torch::cat({_X, _Y}, 1);
    //     //--------------------------
    //     // if(_results_meam.less(1E-2).item().toBool() and _results_meam.greater_equal(1E-3).item().toBool()){
    //     //     //--------------------------
    //     //     if(memory_activation(gen)){
    //     //         //--------------------------
    //     //         return _loss*20;
    //     //         //--------------------------
    //     //     }// end if(abs_activation(gen))
    //     //     //--------------------------
    //     //     return _loss*-20;
    //     //     //--------------------------
    //     // }//end if(_results.less(1E-1).item().toBool())
    //     // // //--------------------------
    //     // if(_results_meam.less(1E-1).item().toBool() and _results_meam.greater_equal(1E-2).item().toBool()){
    //     //     //--------------------------
    //     //     if(memory_activation(gen)){
    //     //         //--------------------------
    //     //         return _loss*10;
    //     //         //--------------------------
    //     //     }// end if(abs_activation(gen))
    //     //     //--------------------------
    //     //     return _loss*-10;
    //     //     //--------------------------
    //     // }//end if(torch::mean(_results).less(1E-2).item().toBool())
    //     // // //--------------------------
    //     // if(_results_meam.less(1E0).item().toBool() and _results_meam.greater_equal(1E-1).item().toBool()){
    //     //     //--------------------------
    //     //     if(memory_activation(gen)){
    //     //         //--------------------------
    //     //         return _loss*5;
    //     //         //--------------------------
    //     //     }// end if(abs_activation(gen))
    //     //     //--------------------------
    //     //     return _loss*-5;
    //     //     //--------------------------
    //     // }//end if(torch::mean(_results).less(1E-2).item().toBool())
    //     // // //--------------------------
    //     // if(_results_meam.less(1E1).item().toBool() and _results_meam.greater_equal(1E-0).item().toBool()){
    //     //     //--------------------------
    //     //     if(memory_activation(gen)){
    //     //         //--------------------------
    //     //         return _loss*2;
    //     //         //--------------------------
    //     //     }// end if(abs_activation(gen))
    //     //     //--------------------------
    //     //     return _loss*-2;
    //     //     //--------------------------
    //     // }//end if(torch::mean(_results).less(1E-2).item().toBool())
    //     //--------------------------
    //     if(memory_activation(gen)){
    //         //--------------------------
    //         return _results/2;
    //         //--------------------------
    //     }// end if(abs_activation(gen))
    //     // //--------------------------
    //     return (_loss/(torch::log10(_loss)*10))/2; 
    //     //--------------------------
    // };
    //--------------------------------------------------------------
    // auto _circle_reward = [&batch_size](const torch::Tensor& input, const torch::Tensor& output){
    //     //--------------------------
    //     std::vector<torch::Tensor> reward;
    //     reward.reserve(output.size(0));
    //     //--------------------------
    //     // std::cout << "output.size: " << output.sizes() << " batch_size: " << batch_size << std::endl;
    //     //--------------------------
    //     size_t _loop_size = output.size(0)/batch_size;
    //     //--------------------------
    //     for (size_t i = 0; i < _loop_size; ++i){
    //         //--------------------------
    //         auto _reward = torch::zeros(batch_size);
    //         //--------------------------
    //         // Criterion 1: Points aligned
    //         Utils::Aligned(_reward, output.slice(0,(i*batch_size), ((i+1)*batch_size)).slice(1,1,2), 1E-1);
    //         //--------------------------
    //         // std::cout << "Aligned: " << " _reward: " << _reward.sizes() << std::endl;
    //         //--------------------------
    //         // Criterion 2: Points close to the circumference of a circle
    //         Utils::CloseToCircumference(_reward,
    //                                     output.slice(0,(i*batch_size), ((i+1)*batch_size)),
    //                                     input.slice(1,0,2).slice(0,i,(i+1)),
    //                                     input.slice(1,2,3).slice(0,i,(i+1)).squeeze(),
    //                                     1E-1);
    //         //--------------------------
    //         // std::cout << "CloseToCircumference: " << " _reward: " << _reward.sizes() << std::endl;
    //         //--------------------------
    //         // Criterion 3: Points equidistant
    //         Utils::Equidistant(_reward, output.slice(0,(i*batch_size), ((i+1)*batch_size)), 1E-1);
    //         //--------------------------
    //         // std::cout << "Equidistant: " << " _reward: " << _reward.sizes() << std::endl;
    //         //--------------------------
    //         // Criterion 4: Angle ratios consistent
    //         Utils::AngleRatiosConsistent(_reward, output.slice(0,(i*batch_size), ((i+1)*batch_size)), 1E-1);
    //         //--------------------------
    //         // std::cout << "AngleRatiosConsistent: " << " _reward: " << _reward.sizes() << std::endl;
    //         //--------------------------
    //         // Criterion 5: Symmetry of the points
    //         Utils::Symmetric(_reward, output.slice(0,(i*batch_size), ((i+1)*batch_size))); 
    //         //--------------------------
    //         // std::cout << "Symmetric: " << " _reward: " << _reward.sizes() << std::endl;
    //         //--------------------------
    //         // Criterion 6: Triangle area
    //         _reward += Utils::TriangleArea(output.slice(0,(i*batch_size), ((i+1)*batch_size)));
    //         //--------------------------
    //         // std::cout << "TriangleArea: " << " _reward: " << _reward.sizes() << std::endl;
    //         //--------------------------
    //         // Criterion 7: Circle smoothness
    //         _reward /= Utils::CircleSmoothness( output.slice(0,(i*batch_size), ((i+1)*batch_size)),
    //                                             input.slice(1,0,2).slice(0,i,(i+1)),
    //                                             input.slice(1,2,3).slice(0,i,(i+1)).squeeze());
    //         //--------------------------
    //         // std::cout << "CircleSmoothness: " << " _reward: " << _reward.sizes() << std::endl;
    //         //--------------------------
    //         reward.push_back(_reward);
    //         //--------------------------
    //     }//end for (size_t i = 0; i < batch_size -1; ++i)        
    //     //--------------------------
    //     // auto reward_answer = torch::cat(reward, 0);
    //     // std::cout << "reward: " << reward_answer.sizes() << std::endl;
    //     //--------------------------
    //     // return reward_answer;
    //     //--------------------------
    //     return torch::cat(reward, 0);
    //     //--------------------------
    // };
    //--------------------------------------------------------------
    // auto _circle_reward = [](const torch::Tensor& input, const torch::Tensor& output){
    //     //--------------------------
    //     std::vector<double> reward;
    //     reward.reserve(output.size(0));
    //     //--------------------------
    //     std::fill_n(std::execution::par, std::back_inserter(reward), output.size(0), 0);
    //     //--------------------------
    //     for (size_t i = 0; i < static_cast<size_t>(output.size(0)); ++i){
    //         //--------------------------
    //         // Criterion 1: Points aligned
    //         if(Utils::Aligned(output[i], 1E-1)){
    //             //--------------------------
    //             reward.at(i) += 1; 
    //             //--------------------------
    //         }// end if(Utils::Aligned(output[i], 1E-1))
    //         //--------------------------
    //         // Criterion 2: Points close to the circumference of a circle
    //         if(Utils::CloseToCircumference( output[i],
    //                                         input.slice(1,0,2).slice(0,i,(i+1)),
    //                                         input.slice(1,2,3).slice(0,i,(i+1)).squeeze(),
    //                                         1E-1)){
    //             //--------------------------
    //             reward.at(i) += 1; 
    //             //--------------------------
    //         }// end if(Utils::CloseToCircumference(output[i], input.slice(1,0,2).slice(0,i,(i+1)), input.slice(1,2,3).slice(0,i,(i+1)).squeeze(), 1E-1))
    //         //--------------------------
    //         // Criterion 3: Points equidistant
    //         if(Utils::Equidistant(output[i], 1E-1)){
    //             //--------------------------
    //             reward.at(i) += 1; 
    //             //--------------------------
    //         }// end if(Utils::Equidistant(output[i], 1E-1))
    //         //--------------------------
    //         // Criterion 4: Angle ratios consistent
    //         if(Utils::AngleRatiosConsistent(output[i], 1E-1)){
    //             //--------------------------
    //             reward.at(i) += 1; 
    //             //--------------------------
    //         }// end if(Utils::AngleRatiosConsistent(output[i], 1E-1))
    //         //--------------------------
    //         // Criterion 5: Symmetry of the points
    //         if(Utils::Symmetric(output[i])){
    //             //--------------------------
    //             reward.at(i) += 1; 
    //             //--------------------------
    //         } // end if(Utils::Symmetric(output[i]))
    //         //--------------------------
    //         // Criterion 6: Triangle area
    //         reward.at(i) += Utils::TriangleArea(output[i]).item<double>();
    //         //--------------------------
    //         // Criterion 7: Circle smoothness
    //         reward.at(i) /= Utils::CircleSmoothness(output[i],
    //                                                 input.slice(1,0,2).slice(0,i,(i+1)),
    //                                                 input.slice(1,2,3).slice(0,i,(i+1)).squeeze()).item<double>();
    //         //--------------------------
    //     }//end for (size_t i = 0; i < batch_size -1; ++i)        
    //     //--------------------------
    //     return torch::tensor(reward);
    //     //--------------------------
    // };
    //--------------------------------------------------------------
    // auto _circle_reward = [](const torch::Tensor& input, const torch::Tensor& output){
    //     //--------------------------
    //     std::vector<double> reward(output.size(0), 0.);
    //     //--------------------------
    //     // Parallelize the loop using STL parallel for_each
    //     std::for_each(std::execution::par, reward.begin(), reward.end(), [&](double& r) {
    //         //--------------------------
    //         size_t i = &r - &reward[0];  // Get the index of the current element
    //         //--------------------------
    //         // Criterion 1: Points aligned
    //         if (Utils::Aligned(output[i], 1E-1)){
    //             //--------------------------
    //             r += 1.;
    //             //--------------------------
    //         }// end if (Utils::Aligned(output[i], 1E-1))
    //         //--------------------------
    //         // Criterion 2: Points close to the circumference of a circle
    //         if (Utils::CloseToCircumference(output[i],
    //                                         input.slice(1, 0, 2).slice(0, i, (i + 1)),
    //                                         input.slice(1, 2, 3).slice(0, i, (i + 1)).squeeze(),
    //                                         1E-1)){
    //             //--------------------------
    //             r += 1.;
    //             //--------------------------
    //         }// end if (Utils::CloseToCircumference(output[i], input.slice(1, 0, 2).slice(0, i, (i + 1)),input.slice(1, 2, 3).slice(0, i, (i + 1)).squeeze(), 1E-1))
    //         //--------------------------
    //         // Criterion 3: Points equidistant
    //         if (Utils::Equidistant(output[i], 1E-1)){
    //             //--------------------------
    //             r += 1.;
    //             //--------------------------
    //         }// end if (Utils::Equidistant(output[i], 1E-1))
    //         //--------------------------
    //         // Criterion 4: Angle ratios consistent
    //         if (Utils::AngleRatiosConsistent(output[i], 1E-1)){
    //             //--------------------------
    //             r += 1.;
    //             //--------------------------
    //         }// end if (Utils::AngleRatiosConsistent(output[i], 1E-1))
    //         //--------------------------
    //         // Criterion 5: Symmetry of the points
    //         if (Utils::Symmetric(output[i])){
    //             //--------------------------
    //             r += 1.;
    //             //--------------------------
    //         }// end if (Utils::Symmetric(output[i]))
    //         //--------------------------
    //         // Criterion 6: Triangle area
    //         r += Utils::TriangleArea(output[i]).item<double>();
    //         //--------------------------
    //         // Criterion 7: Circle smoothness
    //         r /= Utils::CircleSmoothness(output[i],
    //                                     input.slice(1, 0, 2).slice(0, i, (i + 1)),
    //                                     input.slice(1, 2, 3).slice(0, i, (i + 1)).squeeze()).item<double>();
    //         //--------------------------
    //     });      
    //     //--------------------------
    //     return torch::tensor(reward);
    //     //--------------------------
    // };
    //--------------------------------------------------------------
    auto _circle_reward = [](const torch::Tensor& input, const torch::Tensor& output){
        //--------------------------
        std::vector<double> reward(output.size(0), 0.);
        //--------------------------
        // Parallelize the loop using STL parallel for_each
        std::for_each(std::execution::par, reward.begin(), reward.end(), [&](double& r) {
            //--------------------------
            size_t i = &r - &reward[0];  // Get the index of the current element
            //--------------------------
            // Criterion 1: Points aligned
            Utils::Aligned(r, output[i], 1E-1);
            //--------------------------
            // Criterion 2: Points close to the circumference of a circle
            Utils::CloseToCircumference(r,
                                        output[i],
                                        input.slice(1, 0, 2).slice(0, i, (i + 1)),
                                        input.slice(1, 2, 3).slice(0, i, (i + 1)).squeeze(),
                                        1E-1);
            //--------------------------
            // Criterion 3: Points equidistant
            Utils::Equidistant(r, output[i], 1E-1);
            //--------------------------
            // Criterion 4: Angle ratios consistent
            Utils::AngleRatiosConsistent(r, output[i], 1E-1);
            //--------------------------
            // Criterion 5: Symmetry of the points
            Utils::Symmetric(r, output[i]);
            //--------------------------
            // Criterion 6: Triangle area
            Utils::TriangleArea(r, output[i]);
            //--------------------------
            // Criterion 7: Circle smoothness
            Utils::CircleSmoothness(r, 
                                    output[i],
                                    input.slice(1, 0, 2).slice(0, i, (i + 1)),
                                    input.slice(1, 2, 3).slice(0, i, (i + 1)).squeeze());
            //--------------------------
        });      
        //--------------------------
        return torch::tensor(reward);
        //--------------------------
    };
    //--------------------------------------------------------------
    RL::Environment::RLEnvironmentLoader<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> _environment(std::move(input), _circle_reward, 0.9, 0.02, 500., batch_size);
    //--------------------------
    // RLNetLSTM model({points_size, batch_size}, output_size, device, false);
    // RLNetLSTM target_model({points_size, batch_size}, output_size, device, false);
    //--------------------------
    // RLNet model(points_size, output_size);
    // RLNet target_model(points_size, output_size);
    //--------------------------
    // RLNet model(points_size, generated_points_size, output_size);
    // RLNet target_model(points_size, generated_points_size, output_size);
    //--------------------------
    // DuelNet model(points_size, output_size);
    // DuelNet target_model(points_size, output_size);
    //--------------------------
    DuelNet model(points_size, generated_points_size, output_size);
    DuelNet target_model(points_size, generated_points_size, output_size);
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
                                                                                [&_generate, &generated_points_size](const size_t& size = 1, const size_t& col = 2){ 
                                                                                    return  _generate.get_output(size, generated_points_size, col);});
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
        auto output = handler.action(_input, epsilon, batch_size*10, output_size).to(device);
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
            output = handler.action(training_input, epsilon, batch_size*10, output_size);
            //--------------------------
            std::tie(next_input, reward) = _environment.step(epsilon, done, training_input,  _normalize.normalization(output));
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
    for(const auto& x : _rewards){
        std::cout << "_rewards: " << x;
    }// end for(const auto& x : _rewards)
    std::exit(1);
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
    RL::Environment::EnvironmentDataLoader<torch::Tensor> _environment_test(std::move(input_test), batch_size);
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
        auto _loss = torch::mse_loss(_circle, _test.slice(1,2,3));
        //--------------------------
        table   << RL::RLNormalize::unnormalization(_test_result.slice(1,0,1), t_min, t_max) 
                << RL::RLNormalize::unnormalization(_test.slice(1,0,1), t_min, t_max)
                << RL::RLNormalize::unnormalization(_test_result.slice(1,1,2), t_min, t_max) 
                << RL::RLNormalize::unnormalization(_test.slice(1,1,2), t_min, t_max)
                << RL::RLNormalize::unnormalization(_test.slice(1,2,3), t_min, t_max)
                << RL::RLNormalize::unnormalization(_circle, t_min, t_max)
                << _loss << fort::endr;
        //--------------------------
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