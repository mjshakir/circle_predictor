#include "Network/NetworkHandling.hpp"
#include "Generate/Generate.hpp"
#include "SharedLibrary/SharedLibrary.hpp"
#include "Timing/TimingFunction.hpp"
#include "Network/DataLoader.hpp"
#include <random>
#include <omp.h>


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
    // torch::optim::AdamW optimizer(model.parameters(), torch::optim::AdamWOptions(1E-1));
    // torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1E-1));
    //--------------------------
    NetworkHandling handler(model, device);
    //--------------------------
    #pragma omp parallel shared(handler, optimizer)
    {
        //--------------------------
        #pragma omp for nowait schedule(dynamic)
        //--------------------------
        for (int i = 0; i < omp_get_thread_num()/2; i++){
            //--------------------------
            Generate _generate(random_radius(rng), 60000); 
            GenerateDate data = _generate.get_data();
            GenerateDate test_data = _generate.get_test();
            //------------
            std::cout   << "Training data radius: " << _generate.get_radius()  << std::endl;
            //--------------------------
            // Generate your data set. At this point you can add transforms to you data set, e.g. stack your
            // batches into a single tensor.
            auto data_set = DataLoader(data.data.normal_(0.5 ,0.25), data.target.normal_(0.5 ,0.25)).map(torch::data::transforms::Stack<>());
            //--------------------------
            // Generate a data loader.
            auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>( std::move(data_set), 
                                                                                                    torch::data::DataLoaderOptions(20));
            //--------------------------
            // Generate your data set. At this point you can add transforms to you data set, e.g. stack your
            // batches into a single tensor.
            auto test_data_set = DataLoader(test_data.data.normal_(0.5 ,0.25), test_data.target.normal_(0.5 ,0.25)).map(torch::data::transforms::Stack<>());
            //--------------------------
            // Generate a data loader.
            auto test_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_data_set), 
                                                                                                            torch::data::DataLoaderOptions(20));
            //--------------------------
            #pragma omp critical
            {
                //--------------------------
                Timing _timer(__FUNCTION__);
                auto loss = handler.train(std::move(data_loader), std::move(test_data_loader), optimizer, 1000);
                //--------------------------
            }// end #pragma omp critical 
            //--------------------------
        }// end 
        //--------------------------
    }// end #pragma omp parallel shared(data)
    //--------------------------
    return 0;
    //--------------------------
}// end int main(int argc, char const *argv[])
