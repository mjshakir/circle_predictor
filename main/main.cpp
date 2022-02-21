#include <random>
#include <fstream>
#include "Network/Network.hpp"
#include "Generate/Generate.hpp"
#include "Timing/Timing.hpp"

//--------------------------------------------------------------
int main(int argc, char const *argv[]){
    //--------------------------
    if (argc >= 6){
        throw std::invalid_argument("More arugments then can be allocation");
    }// end if (argc >= 6)
    //--------------------------------------------------------------
    // Command line arugments for training size and data generation 
    //--------------------------
    size_t training_size{100}, generated_size{10000}, epoch{100};
    bool isEpoch{false};
    long double precision{2.5E-1L};
    //--------------------------
    if (argc > 1){
        //--------------------------
        if (std::atoi(argv[1]) > 0){
            //--------------------------
            training_size = std::stoul(argv[1]);
            //--------------------------
        }// end if (std::atoi(argv[1]) > 0)
        else{
            //--------------------------
            throw std::out_of_range("Must be at least postive");
            //--------------------------
        }// end else 
        //--------------------------
    }// end if (argc > 1)
    //-----------
    if (argc > 2){
        //--------------------------
        if (std::atoi(argv[2]) >= 200){
            //--------------------------
            generated_size = std::stoul(argv[2]);
            //--------------------------
        }// end if (std::atoi(argv[2]) >= 200)
        else{
            //--------------------------
            throw std::out_of_range("Must be at least 200 (x >= 200)");
            //--------------------------
        }// end else
        //-------------------------- 
    }// end if (argc > 2)
    //-----------
    if (argc > 3){
        //--------------------------
        std::string _input = argv[3];
        std::transform(std::execution::par, _input.begin(), _input.end(), _input.begin(), [](uint8_t c){ return std::tolower(c);});
        //--------------------------
        if (std::atoi(argv[3]) == 1 or std::strncmp(_input.c_str(), "true", 4) == 0){
            //--------------------------
            isEpoch = true;
            //--------------------------
        }// end if (std::atoi(argv[4]) == 1 or tolower(argv[4]) == "true")
        //-------------------------- 
    }// end if (std::atoi(argv[3]) == 1 or std::strncmp(argv[3], "true", 4) == 0)
    //-----------
    if (argc > 4 and !isEpoch){
        //--------------------------
        precision = std::stold(argv[4]);
        //--------------------------
    }// end if (argc > 4 and !isEpoch)
    //-----------
    if (argc > 4 and isEpoch){
        //--------------------------
        if (std::atoi(argv[4]) > 0){
            //--------------------------
            epoch = std::stoul(argv[4]);
            //--------------------------
        }// end if (std::atoi(argv[5]) > 0)
        else{
            //--------------------------
            throw std::out_of_range("Must be at least postive");
            //--------------------------
        }// end else
        //-------------------------- 
    }// end if (argc > 4 and isEpoch)
    //--------------------------
    if(isEpoch){
        std::cout   << "training_size: " << training_size 
                    << "\ngenerated_size: " << generated_size 
                    << "\nepoch: " << epoch << std::endl;
    }// end if(isEpoch)
    else{
        std::cout   << "training_size: " << training_size 
                    << "\ngenerated_size: " << generated_size 
                    << "\nprecision: " << precision << std::endl;
    }// end else
    //--------------------------------------------------------------
    // Creating a random number generator
    //--------------------------
    std::random_device dev;
    std::mt19937 rng(dev()), center_rng(dev());
    std::uniform_real_distribution<double> random_radius(1,10);
    std::uniform_int_distribution<> random_centers(-10,10);
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
    // Initialize the network
    //--------------------------
    Net model;
    model.to(device);
    // LSTMNet model(device);
    // model.to(device);
    //--------------------------
    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(1E-1L).momentum(0.95).nesterov(true));
    //--------------------------
    NetworkHandling<Net> handler(std::move(model), device);
    //--------------------------------------------------------------
    // Training loop
    //--------------------------
    for (size_t i = 0; i < training_size; i++){
        //--------------------------------------------------------------
        // Generate data
        //--------------------------
        Generate _generate(random_radius(rng), generated_size, {random_centers(center_rng), random_centers(center_rng)}); 
        //------------
        std::cout   << "Training data radius: " << _generate.get_radius() 
                    << " at center: (" << std::get<0>(_generate.get_center()) << "," 
                    << std::get<1>(_generate.get_center()) << ")" << std::endl;
        //--------------------------------------------------------------
        // Generate your data set. At this point you can add transforms to you data set, e.g. stack your
        // batches into a single tensor.
        // auto data_set = DataLoader(std::move(std::get<0>(data)), std::move(std::get<1>(data))).map(torch::data::transforms::Normalize<>(0.5, 0.25)).map(torch::data::transforms::Stack<>());
        //--------------------------
        auto data_set = DataLoader( Normalize::normalization(std::get<0>(_generate.get_data())), 
                                    Normalize::normalization(std::get<1>(_generate.get_data()))).map(torch::data::transforms::Stack<>());
        //--------------------------
        // Generate a data loader.
        //--------------------------
        auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>( std::move(data_set), 
                                                                                                torch::data::DataLoaderOptions(20));
        //--------------------------------------------------------------
        // Time the loop
        //--------------------------
        Timing _timer(__FUNCTION__);
        //--------------------------------------------------------------
        // Train the network
        //--------------------------
        if (isEpoch){
            auto loss = handler.train(std::move(data_loader), optimizer, epoch);
        }// end if (isEpoch)
        else{
            //--------------------------------------------------------------
            // Generate your validation data set. At this point you can add transforms to you data set, e.g. stack your
            // batches into a single tensor.
            //--------------------------
            auto validation_data_set = DataLoader(  Normalize::normalization(std::get<0>(_generate.get_validation())), 
                                                    Normalize::normalization(std::get<1>(_generate.get_validation())))
                                                        .map(torch::data::transforms::Stack<>());
            //--------------------------
            // Generate a data loader.
            //--------------------------
            auto validation_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(validation_data_set), 
                                                                                                            torch::data::DataLoaderOptions(20));
            //--------------------------
            auto loss = handler.train(std::move(data_loader), std::move(validation_data_loader), optimizer, precision);
        }// end else 
        //--------------------------
        printf("\n-----------------Done:[%zu]-----------------\n", i);
        //--------------------------
    }// end (size_t i = 0; i < training_size; i++)
    //--------------------------------------------------------------
    // Generate test data
    //--------------------------
    Generate test_generate(random_radius(rng), 60, {random_centers(center_rng), random_centers(center_rng)});
    // Generate test_generate(5, 60, {0, 0}); 
    //--------------------------
    std::cout   << "test data radius: " << test_generate.get_radius() 
                << " at center: (" << std::get<0>(test_generate.get_center()) << "," 
                << std::get<1>(test_generate.get_center()) << ")" << std::endl;
    //--------------------------------------------------------------
    // Normalize the data
    //--------------------------
    Normalize test_input_normal(std::get<0>(test_generate.get_data()));
    Normalize test_target_normal(std::get<1>(test_generate.get_data()));
    //--------------------------
    // Generate your data set. At this point you can add transforms to you data set, e.g. stack your
    // batches into a single tensor.
    //--------------------------
    auto test_data_set = DataLoader(test_input_normal.normalization(), 
                                    test_target_normal.normalization()).map(torch::data::transforms::Stack<>());
    //--------------------------
    // Generate a data loader.
    //--------------------------
    auto test_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>( std::move(test_data_set), 
                                                                                            torch::data::DataLoaderOptions(20));
    //--------------------------
    // Test the data
    //--------------------------
    auto test = handler.test(std::move(test_data_loader));
    //--------------------------
    for (const auto& _test : test){
        //--------------------------
        std::cout   << "\ntarget : \n" << test_target_normal.unnormalization(std::get<0>(_test)) 
                    << "\noutput: \n" << test_target_normal.unnormalization(std::get<1>(_test)) 
                    << "\ntarget origial: \n" << std::get<0>(_test) 
                    << "\noutput origial: \n" << std::get<1>(_test)
                    << "\nloss: " << std::get<2>(_test) << std::endl;
        //--------------------------
    }// end for (const auto& _test : test)
    //--------------------------------------------------------------
    // file pointer
    //--------------------------------------------------------------
    std::fstream fout;
    //--------------------------
    // opens an existing csv file or creates a new file.
    //--------------------------
    fout.open("test_data.csv", std::ios::out | std::ios::app);
    //--------------------------
    fout    << "Radius: " << test_generate.get_radius() 
            << " at center: (" << std::get<0>(test_generate.get_center()) << "." 
            << std::get<1>(test_generate.get_center()) << ")" << "\n"
            << "target" << ","
            << "output" << "," 
            << "target original" << ","
            << "output original" << ","
            << "loss" << "\n";
    //--------------------------
    fout.flush();
    //--------------------------
    for (const auto& _test : test){
        //--------------------------
        auto num_row = std::get<0>(_test).size(0);
        auto num_col = std::get<0>(_test).size(1);
        //--------------------------
        auto _output_temp = std::get<0>(_test).cpu();
        auto _output = _output_temp.accessor<float, 2>();
        auto _target_temp = std::get<1>(_test).cpu();
        auto _target = _target_temp.accessor<float, 2>();
        //--------------------------
        for (int64_t i = 0; i < num_row; i++){
            //--------------------------
            for (int64_t j = 0; j < num_col; j++){
                //--------------------------
                fout    << test_target_normal.unnormalization_nonTensor(_output[i][j]) << ","
                        << test_target_normal.unnormalization_nonTensor(_target[i][j]) << "," 
                        << _output[i][j] << ","
                        << _target[i][j] << "," 
                        << std::get<2>(_test) << "\n";
                //--------------------------
                fout.flush();
                //--------------------------
            }// end for (size_t i = 0; i < num_col; i++)
            //--------------------------
        }// end for (size_t i = 0; i < num_col; i++)
        //--------------------------
    }// end for (const auto& _test : test)
    //--------------------------
    fout.close();
    //--------------------------------------------------------------
    std::cout << "test data Saved" << std::endl;
    //--------------------------------------------------------------
    return 0;
    //--------------------------
}// end int main(int argc, char const *argv[])
