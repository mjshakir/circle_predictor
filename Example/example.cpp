#include "Network/NetworkHandling.hpp"
#include "Generate/Generate.hpp"
#include "SharedLibrary/SharedLibrary.hpp"
#include "Timing/TimingFunction.hpp"
#include "Network/DataLoader.hpp"

int main(){
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
    Net model;
    model.to(device);
    //--------------------------
    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(1E-3).momentum(0.75).nesterov(true));
    // torch::optim::AdamW optimizer(model.parameters(), torch::optim::AdamWOptions(1E-3));
    //--------------------------
    // GenerateDate data = Generate::GetInstance(1.f)->get_data();
    Generate _generator(1.f, 60);
    GenerateDate data = _generator.get_data();
    //------------
    GenerateDate test_data = Generate(1.5, 40).get_data();
    //--------------------------
    // Generate your data set. At this point you can add transforms to you data set, e.g. stack your
    // batches into a single tensor.
    auto data_set = DataLoader(data.data, data.target).map(torch::data::transforms::Stack<>());
    //--------------------------
    // Generate a data loader.
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(data_set), 20);
    //--------------------------
    // Generate your data set. At this point you can add transforms to you data set, e.g. stack your
    // batches into a single tensor.
    auto test_data_set = DataLoader(test_data.data, test_data.target).map(torch::data::transforms::Stack<>());
    //--------------------------
    // Generate a data loader.
    auto test_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_data_set), 20);
    //--------------------------
    // NetworkHandling handler(model, device);
    // Timing _timer(__FUNCTION__);
    // // auto loss = TimingFunction(handler.train(10, data_loader, optimizer), "train").get_result();
    // auto loss = handler.train(10, data_loader, optimizer);
    // // std::cout << "Loss size: " << loss.size() << std::endl;
    //--------------------------
    NetworkHandling handler(model, device);
    Timing _timer(__FUNCTION__);
    auto loss = handler.train(data_loader, test_data_loader, optimizer, 10);
    //--------------------------
    // for (const auto& i : loss){
    //     std::cout << "Current loss: " << i << std::endl;
    // }
    //--------------------------
    // std::cout << "Data: " << x.sizes() << std::endl;
    //--------------------------
    // std::cout << "Data: " << data.data << std::endl;
    //--------------------------
    // std::cout << "Target: " << data.target << std::endl;
    //--------------------------
    return 0;
    //--------------------------
}
