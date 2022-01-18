#include "Network/NetworkHandling.hpp"
#include "Generate/Generate.hpp"
#include "Timing/TimingFunction.hpp"
#include "Network/DataLoader.hpp"
#include <random>

int main(){
    //--------------------------
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<double> random_radius(1,10);
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
    //--------------------------
    Net model(device);
    model.to(device);
    //--------------------------
    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(1E-1).momentum(0.95).nesterov(true));
    //--------------------------
    NetworkHandling handler(model, device);
    //--------------------------
    for (size_t i = 0; i < 5; i++){
        //--------------------------
        Generate _generate(random_radius(rng), 600); 
        auto data = _generate.get_data();
        auto validation_data = _generate.get_validation();
        //------------
        std::cout   << "Training data radius: " << _generate.get_radius()  << std::endl;
        //--------------------------
        // Generate your data set. At this point you can add transforms to you data set, e.g. stack your
        // batches into a single tensor.
        // auto data_set = DataLoader(std::move(std::get<0>(data)), std::move(std::get<1>(data))).map(torch::data::transforms::Normalize<>(0.5, 0.25)).map(torch::data::transforms::Stack<>());
        auto data_set = DataLoader(std::move(std::get<0>(data).normal_(0.5,0.25)), std::move(std::get<1>(data).normal_(0.5,0.25))).map(torch::data::transforms::Stack<>());
        //--------------------------
        // Generate a data loader.
        auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>( std::move(data_set), 
                                                                                                torch::data::DataLoaderOptions(20));
        //--------------------------
        // Generate your data set. At this point you can add transforms to you data set, e.g. stack your
        // batches into a single tensor.
        auto validation_data_set = DataLoader(std::move(std::get<0>(validation_data).normal_(0.5,0.25)), std::move(std::get<1>(validation_data).normal_(0.5,0.25)))
                                            .map(torch::data::transforms::Stack<>());
        //--------------------------
        // Generate a data loader.
        auto validation_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(validation_data_set), 
                                                                                                        torch::data::DataLoaderOptions(20));

        //--------------------------
        Timing _timer(__FUNCTION__);
        auto loss = handler.train(std::move(data_loader), std::move(validation_data_loader), optimizer, 1E-5L);
        //--------------------------
        printf("\n-----------------Done:[%zu]-----------------\n", i);
        //--------------------------
    }// end (size_t i = 0; i < 10; i++)
    //--------------------------
    auto test_data = Generate(1, 60).get_data();
    //--------------------------
    // Generate your data set. At this point you can add transforms to you data set, e.g. stack your
    // batches into a single tensor.
    auto test_data_set = DataLoader(std::move(std::get<0>(test_data)), std::move(std::get<1>(test_data))).map(torch::data::transforms::Stack<>());
    //--------------------------
    // Generate a data loader.
    auto test_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>( std::move(test_data_set), 
                                                                                            torch::data::DataLoaderOptions(20));
    //--------------------------
    auto test = handler.test(std::move(test_data_loader));
    for (const auto& _test : test){
        std::cout << "target: " << std::get<0>(_test) << " output: " << std::get<1>(_test) << " loss " << std::get<2>(_test) << std::endl;
    }
    //--------------------------
    return 0;
    //--------------------------
}// end int main(int argc, char const *argv[])
