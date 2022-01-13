#pragma once

//--------------------------------------------------------------
#include <torch/torch.h>
#include <future>
#include <thread>
#include <iterator>
#include <algorithm>
#include <execution>
//--------------------------------------------------------------
#include "Network/Network.hpp"
//--------------------------------------------------------------
#include "SharedLibrary/SharedLibrary.hpp"
#include "Timing/Timing.hpp"
//--------------------------------------------------------------
class NetworkHandling{
    public:
        //--------------------------------------------------------------
        NetworkHandling(Net& model, torch::Device& device);
        //--------------------------
        template <typename Dataloader>
        std::vector<float> train(const size_t& epoch, Dataloader& data_loader, torch::optim::Optimizer& optimizer){
            //--------------------------
            return network_train(epoch, data_loader, optimizer);
            //--------------------------
        }// end std::vector<torch::Tensor> NetworkHandling::train(const size_t& epoch, Dataloader& data_loader, torch::optim::Optimizer& optimizer)
        //--------------------------
        template <typename Dataloader, typename Test_Dataloader>
        std::vector<float> train(Dataloader& data_loader, Test_Dataloader& data_loader_test, torch::optim::Optimizer& optimizer, float precision = 10){
            //--------------------------
            return network_train(data_loader, data_loader_test, optimizer, precision);
            //--------------------------
        }// end std::vector<torch::Tensor> NetworkHandling::train(const size_t& epoch, Dataloader& data_loader, torch::optim::Optimizer& optimizer)
        //--------------------------
        template <typename Dataset>
        std::vector<float> test(Dataset& data_loader){
            //--------------------------
            return network_test(data_loader);
            //--------------------------
        }// end std::vector<double> NetworkHandling::network_test(DataLoader& data_loader)
        //--------------------------------------------------------------
    private:
        //--------------------------
        Net m_model; 
        torch::Device m_device;
        //--------------------------
        template <typename Batch>
        float network_train_batch(Batch& batch, torch::optim::Optimizer& optimizer){
            //--------------------------
            m_model.train(true);
            //--------------------------
            // auto data = batch.data.to(m_device), targets = batch.target.to(torch::kLong).to(m_device);
            auto data = batch.data.to(m_device), targets = batch.target.to(m_device);
            //--------------------------
            optimizer.zero_grad();
            //--------------------------
            auto output = m_model.forward(data);
            //--------------------------
            // std::cout << "output: " << output.sizes() << std::endl;
            //--------------------------
            // std::cout << __FUNCTION__ << std::endl;
            //--------------------------
            // torch::Tensor loss = torch::nll_loss(output, targets);
            torch::Tensor loss = torch::mse_loss(output, targets);
            // AT_ASSERT(!std::isnan(loss.template item<float>()));
            //--------------------------
            // loss.backward(torch::nullopt, /*keep_graph=*/ true, /*create_graph=*/ false);
            loss.backward({},c10::optional<bool>(true), false);
            optimizer.step();
            //--------------------------
            return loss.template item<float>();
            //--------------------------
        }// end at::Tensor NetworkHandling::network_train(Batch& batch)
        //--------------------------
        template <typename Batch>
        float network_test_batch(Batch& batch){
            //--------------------------
            torch::NoGradGuard no_grad;
            m_model.train(false);
            m_model.eval();
            //--------------------------
            auto data = batch.data.to(m_device), targets = batch.target.to(m_device);
            auto output = m_model.forward(data);
            // auto printing_threads = std::async(std::launch::async, [&targets, &output](){
            //                                                                                 std::cout << "targets: [" << targets << "] " <<  "output: [" << output << "]" << std::endl;
            //                                                                                 return true;
            //                                                                             });
            //--------------------------
            // return torch::nll_loss(output, targets,/*weight=*/{}, torch::Reduction::Sum).template item<double>();
            return torch::mse_loss(output, targets, torch::Reduction::Sum).template item<double>();
            //--------------------------
        }// end double NetworkHandling::network_test(Batch& batch)
        //--------------------------
        template <typename Dataloader>
        std::vector<float> network_train(const size_t& epoch, Dataloader& data_loader, torch::optim::Optimizer& optimizer){
            //--------------------------
            std::vector<float> Loss;
            //--------------------------
            auto _scheduler = torch::optim::StepLR(optimizer, 30, 0.01);
            //--------------------------
            for (size_t i = 0; i < epoch; i++){
                //--------------------------
                Timing _timer_loop("epoch: " + std::to_string(i));
                //--------------------------
                for (const auto& batch : *data_loader){
                    //--------------------------
                    Loss.emplace_back(network_train_batch(batch, optimizer));
                    //--------------------------
                }// end for (const auto& batch : *data_loader)
                //--------------------------
                _scheduler.step();
                //--------------------------
                auto printing_threads = std::async(std::launch::async, [&i, &Loss](){   for (const auto& loss : Loss){
                                                                                            printf("Loss [\x1b[36m%ld\x1b[0m]: [\x1b[31m%0.2f\x1b[0m] ", i, loss);
                                                                                        }// end for (const auto& loss : Loss) 
                                                                                        printf("\n-----------------size of loss [%ld]----------------------\n", Loss.size());
                                                                                    });
                //--------------------------
            }// end for (size_t i = 0; i < epoch; i++)
            //--------------------------
            return Loss;
            //--------------------------
        }// end std::vector<at::Tensor> NetworkHandling::network_train(DataLoader& data_loader, size_t& epoch)
        //--------------------------
        template <typename Dataloader, typename Test_Dataloader>
        std::vector<float> network_train(Dataloader& data_loader, Test_Dataloader& data_loader_test, torch::optim::Optimizer& optimizer, const float& precision){
            //--------------------------
            std::mutex mutex;
            //--------------------------
            double _element_sum{100};
            std::vector<float> Loss;
            //--------------------------
            auto _scheduler = torch::optim::StepLR(optimizer, 30, 1E-2);
            //--------------------------
            do{
                //--------------------------
                Timing _timer_loop("While loop");
                //--------------------------
                // for (const auto& batch : *data_loader){
                //     //--------------------------
                //     Loss.push_back(network_train_batch(batch, optimizer));
                //     //--------------------------
                // }// end for (const auto& batch : *data_loader)
                //--------------------------
                std::for_each(std::execution::par, data_loader->begin(), data_loader->end(), [&](auto&& batch){ std::lock_guard<std::mutex> lock(mutex);
                                                                                                                Loss.push_back(network_train_batch(batch, optimizer));});
                // std::ranges::for_each(std::begin(*data_loader), std::end(*data_loader), [&](const auto& batch){Loss.push_back(network_train_batch(batch, optimizer));});
                //--------------------------
                _scheduler.step();
                //--------------------------        
                auto _test_loss = network_test(data_loader_test);
                //--------------------------
                if (!_test_loss.empty()){
                    //--------------------------
                    _element_sum = 0.f;
                    //--------------------------
                    for (const auto& _loss : _test_loss){
                        //--------------------------
                        _element_sum += _loss;
                        //--------------------------
                    }// end for (const auto& _loss : _test_loss)
                    //--------------------------
                }// end if (!_test_loss.empty())
                //--------------------------
                auto printing_threads = std::async(std::launch::async, loss_disply, _test_loss, _element_sum);
                //--------------------------
            } while(_element_sum >= precision);
            //--------------------------
            return Loss;
            //--------------------------
        }// end std::vector<at::Tensor> NetworkHandling::network_train(DataLoader& data_loader, size_t& epoch)
        //--------------------------
        template <typename Dataset>
        std::vector<float> network_test(Dataset& data_loader){
            //--------------------------
            std::vector<float> test_loss;
            //--------------------------
            for (const auto& batch : *data_loader){
                //--------------------------
                test_loss.emplace_back(network_test_batch(batch));
                //--------------------------
            }// end for (const auto& batch : data_loader)
            //--------------------------
            return test_loss;
            //--------------------------
        }// end std::vector<double> NetworkHandling::network_test(DataLoader& data_loader)
        //--------------------------------------------------------------
        static void loss_disply(const std::vector<float>& loss, const double& elements_sum);
        //--------------------------------------------------------------
};
//--------------------------------------------------------------