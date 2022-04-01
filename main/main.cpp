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
#include "Network/Network.hpp"
#include "Generate/Generate.hpp"
#include "Timing/Timing.hpp"
//--------------------------------------------------------------

int main(int argc, char const *argv[]){
    //--------------------------------------------------------------
    // Command line arugments for training size and data generation 
    //--------------------------
    std::string filename;
    size_t training_size, generated_size, batch_size, epoch;
    bool isEpoch;
    long double precision;
    //--------------------------
    boost::program_options::options_description description("Allowed options:");
    //--------------------------
    description.add_options()
    ("help,h", "Display this help message")
    ("version,v", "Display the version number")
    ("training_size,t", boost::program_options::value<size_t>(&training_size)->default_value(100), "Training Size")
    ("generated_size,g", boost::program_options::value<size_t>(&generated_size)->default_value(10000), "Generated Size")
    ("batch_size,b", boost::program_options::value<size_t>(&batch_size)->default_value(20), "Batch Size")
    ("epoch,e", boost::program_options::value<size_t>(&epoch)->default_value(100), "Epoch Size")
    ("use_epoch,u", boost::program_options::value<bool>(&isEpoch)->default_value(false), "Using Epoch")
    ("precision,p", boost::program_options::value<long double>(&precision)->default_value(2.5E-1L), "Precision")
    ("filename,s", boost::program_options::value<std::string>(&filename)->default_value("test_results.csv"), "Name of the file saved");
    //--------------------------
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(description).run(), vm);
    boost::program_options::notify(vm);
    //--------------------------
    if (vm.count("help")){
        std::cout << description;
    }
    //--------------------------
    if(isEpoch){
        std::cout   << "filename: " << filename << "\n"
                    << "training_size: " << training_size << "\n"
                    << "generated_size: " << generated_size << "\n"
                    << "batch_size: " << batch_size << "\n"
                    << "epoch: " << epoch << std::endl;
    }// end if(isEpoch)
    else{
        std::cout   << "filename: " << filename << "\n"
                    << "training_size: " << training_size << "\n"
                    << "generated_size: " << generated_size << "\n"
                    << "batch_size: " << batch_size << "\n"
                    << "precision: " << precision << std::endl;
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
    Net model(batch_size);
    model.to(device);
    // LSTMNet model(device);
    // model.to(device);
    //--------------------------
    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(1E-1L).momentum(0.95).nesterov(true));
    //--------------------------
    NetworkHandling<Net> handler(std::move(model), device);
    //--------------------------------------------------------------
    // Timing main
    //--------------------------
    Timing _timer(__FUNCTION__);
    //--------------------------------------------------------------
    // Training loop
    //--------------------------
    for (size_t i = 0; i < training_size; ++i){
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
                                                                                                torch::data::DataLoaderOptions(batch_size));
        //--------------------------------------------------------------
        // Time the loop
        //--------------------------
        Timing _loop_timer("Main for loop");
        //--------------------------------------------------------------
        // Train the network
        //--------------------------
        if (isEpoch){
            //--------------------------
            auto loss = handler.train(std::move(data_loader), optimizer, epoch);
            //--------------------------
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
                                                                                                            torch::data::DataLoaderOptions(batch_size));
            //--------------------------
            auto loss = handler.train(std::move(data_loader), std::move(validation_data_loader), optimizer, precision);
            //--------------------------
        }// end else 
        //--------------------------
        printf("\n-----------------Done:[%zu]-----------------\n", i);
        //--------------------------
    }// end (size_t i = 0; i < training_size; ++i)
    //--------------------------------------------------------------
    // Generate test data
    //--------------------------
    Generate test_generate(random_radius(rng), 3*batch_size, {random_centers(center_rng), random_centers(center_rng)});
    //--------------------------
    std::cout   << "\n" << "Test Data"
                << "\n" << "test data radius: " << test_generate.get_radius() 
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
                                                                                            torch::data::DataLoaderOptions(batch_size));
    //--------------------------
    // Test the data
    //--------------------------
    auto test = handler.test(std::move(test_data_loader));
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
    //--------------------------
    table   << fort::header
            << "Target" << "Output" << "Target Origial" << "Output origial" << "Loss" << fort::endr;
    //--------------------------------------------------------------
    // File pointer
    //--------------------------
    std::fstream fout;
    //--------------------------
    // opens an existing csv file or creates a new file.
    //--------------------------
    fout.open(filename, std::ios::out | std::ios::app);
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
    for (const auto& [_test_target, _test_output, _loss] : test){
        //--------------------------
        table   << test_target_normal.unnormalization(_test_target) 
                << test_target_normal.unnormalization(_test_output) 
                << _test_target 
                << _test_output
                << _loss << fort::endr;
        //--------------------------
        auto _output_temp = _test_target.cpu();
        auto _output = _output_temp.accessor<float, 2>();
        auto _target_temp = _test_output.cpu();
        auto _target = _target_temp.accessor<float, 2>();
        //--------------------------
        for (int64_t i = 0; i < _test_target.size(0); ++i){
            //--------------------------
            for (int64_t j = 0; j < _test_target.size(1); ++j){
                //--------------------------
                fout    << test_target_normal.unnormalization_nonTensor(_output[i][j]) << ","
                        << test_target_normal.unnormalization_nonTensor(_target[i][j]) << "," 
                        << _output[i][j] << ","
                        << _target[i][j] << "," 
                        << _loss << "\n";
                //--------------------------
                fout.flush();
                //--------------------------
            }// end for (size_t i = 0; i < _test_target.size(0); ++i)
            //--------------------------
        }// end for (size_t i = 0; i < _test_target.size(1); ++i)
        //--------------------------
    }// end for (const auto& [_test_target, _test_output, _loss] : test)
    //--------------------------
    fout.close();
    //--------------------------------------------------------------
    std::cout << "\n" << table.to_string() << std::endl;
    //--------------------------
    std::cout << "test data Saved" << std::endl;
    //--------------------------------------------------------------
    return 0;
    //--------------------------
}// end int main(int argc, char const *argv[])
