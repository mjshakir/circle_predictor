#pragma once

//--------------------------------------------------------------
#include <torch/torch.h>
#include <future>
#include <algorithm>
#include <execution>
//--------------------------------------------------------------
#include "Network/Network.hpp"
//--------------------------------------------------------------
#include "Timing/Timing.hpp"
//--------------------------------------------------------------
#include "progressbar/include/progressbar.hpp"
//--------------------------------------------------------------
class NetworkHandling{
    public:
        //--------------------------------------------------------------
        NetworkHandling() = delete;
        //--------------------------
        NetworkHandling(Net& model, torch::Device& device);
        //--------------------------
        template <typename Dataloader>
        std::vector<float> train(Dataloader&& data_loader, torch::optim::Optimizer& optimizer, const size_t& epoch){
            //--------------------------
            return network_train(epoch, data_loader, optimizer);
            //--------------------------
        }// end std::vector<float> train(Dataloader&& data_loader, torch::optim::Optimizer& optimizer, const size_t& epoch)
        //--------------------------
        template <typename Dataloader, typename Test_Dataloader>
        std::vector<float> train(Dataloader&& data_loader, Test_Dataloader&& data_loader_test, torch::optim::Optimizer& optimizer, float precision = 10){
            //--------------------------
            return network_train(data_loader, data_loader_test, optimizer, precision);
            //--------------------------
        }// end  std::vector<float> train(Dataloader&& data_loader, Test_Dataloader&& data_loader_test, torch::optim::Optimizer& optimizer, float precision = 10)
        //--------------------------
        template <typename Dataset>
        std::vector<float> validation(Dataset&& data_loader){
            //--------------------------
            return network_validation(data_loader);
            //--------------------------
        }// end std::vector<float> validation(Dataset&& data_loader)
        //--------------------------------------------------------------
        template <typename Dataset>
        std::vector<std::tuple<torch::Tensor, torch::Tensor, float>> test(Dataset&& data_loader){
            //--------------------------
            return network_test(data_loader);
            //--------------------------
        }// end std::vector<std::tuple<torch::Tensor, torch::Tensor, float>> test(Dataset&& data_loader)
        //--------------------------------------------------------------
    private:
        //--------------------------
        Net m_model; 
        torch::Device m_device;
        //--------------------------
        template <typename Batch>
        float network_train_batch(Batch&& batch, torch::optim::Optimizer& optimizer, bool *tensorIsNan){
            //--------------------------
            m_model.train(true);
            //--------------------------
            auto data = batch.data.to(m_device), targets = batch.target.to(m_device);
            //--------------------------
            optimizer.zero_grad();
            //--------------------------
            auto output = m_model.forward(data);
            output = torch::transpose(output.view({2,-1}), 0, 1);
            //--------------------------
            torch::Tensor loss = torch::mse_loss(output, targets);
            // AT_ASSERT(!std::isnan(loss.template item<float>()));
            //--------------------------
            *tensorIsNan = at::isnan(loss).any().item<bool>(); // will be of type bool
            //--------------------------
            // loss.backward(torch::nullopt, /*keep_graph=*/ true, /*create_graph=*/ false);
            loss.backward({},c10::optional<bool>(true), false);
            optimizer.step();
            //--------------------------
            return loss.template item<float>();
            //--------------------------
        }// end float network_train_batch(Batch&& batch, torch::optim::Optimizer& optimizer)
        //--------------------------
        template <typename Batch>
        float network_validation_batch(Batch&& batch){
            //--------------------------
            torch::NoGradGuard no_grad;
            // m_model.train(false);
            m_model.eval();
            //--------------------------
            auto data = batch.data.to(m_device), targets = batch.target.to(m_device);
            auto output = m_model.forward(data);
            output = torch::transpose(output.view({2,-1}), 0, 1);
            //--------------------------
            return torch::mse_loss(output, targets, torch::Reduction::Sum).template item<double>();
            //--------------------------
        }// end float network_validation_batch(Batch&& batch)
        //--------------------------
        template <typename Batch>
        std::tuple<torch::Tensor, torch::Tensor, float> network_test_batch(Batch&& batch){
            //--------------------------
            torch::NoGradGuard no_grad;
            // m_model.train(false);
            m_model.eval();
            //--------------------------
            auto data = batch.data.to(m_device), targets = batch.target.to(m_device);
            auto output = m_model.forward(data);
            output = torch::transpose(output.view({2,-1}), 0, 1);
            //--------------------------
            return {targets, output, torch::mse_loss(output, targets, torch::Reduction::Sum).template item<float>()};
            //--------------------------
        }// end std::tuple<torch::Tensor, torch::Tensor, float> network_test_batch(Batch&& batch)
        //--------------------------
        template <typename Dataloader>
        std::vector<float> network_train(Dataloader&& data_loader, torch::optim::Optimizer& optimizer, const size_t& epoch){
            //--------------------------
            progressbar bar(epoch);
            //--------------------------
            bool tensorIsNan = false;
            //--------------------------
            std::vector<float> Loss;
            //--------------------------
            torch::optim::StepLR _scheduler(optimizer, 30, 1E-2);
            //--------------------------
            for (size_t i = 0; i < epoch; i++){
                //--------------------------
                Timing _timer_loop("epoch: " + std::to_string(i));
                //--------------------------
                for (const auto& batch : *data_loader){
                    //--------------------------
                    bar.update();
                    //------------
                    Loss.push_back(network_train_batch(std::move(batch), optimizer, &tensorIsNan));
                    //--------------------------
                }// end for (const auto& batch : *data_loader)
                //--------------------------
                if(tensorIsNan){
                    std::cout << "\x1b[33m\ntensor is nan\x1b[0m" << std::endl;
                    break;
                }// end if(tensorIsNan)
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
        }// end std::vector<float> network_train(Dataloader&& data_loader, torch::optim::Optimizer& optimizer, const size_t& epoch)
        //--------------------------
        template <typename Dataloader, typename Test_Dataloader>
        std::vector<float> network_train(Dataloader&& data_loader, Test_Dataloader&& data_loader_test, torch::optim::Optimizer& optimizer, const float& precision){
            //--------------------------
            // std::mutex mutex;
            //--------------------------
            double _element_sum{100};
            std::vector<float> Loss;
            //--------------------------
            auto data_loader_size = std::distance(data_loader->begin(), data_loader->end());
            //--------------------------
            bool _learning = true, tensorIsNan = false;
            std::vector<double> _learning_elements;
            _learning_elements.reserve(5);
            //--------------------------
            torch::optim::StepLR _scheduler(optimizer, 30, 1E-2);
            //--------------------------
            do{
                //--------------------------
                progressbar bar(data_loader_size);
                //--------------------------
                std::cout << "Training: ";
                //--------------------------
                Timing _timer_loop("While loop");
                //--------------------------
                for (const auto& batch : *data_loader){
                    //--------------------------
                    bar.update();
                    //------------
                    Loss.push_back(network_train_batch(std::move(batch), optimizer, &tensorIsNan));
                    //--------------------------
                    if(tensorIsNan){
                    std::cout << "\x1b[33m\nTensor is [nan]\x1b[0m" << std::endl;
                    break;
                }// end if(tensorIsNan)
                //--------------------------
                }// end for (const auto& batch : *data_loader)
                //--------------------------
                _scheduler.step();
                //--------------------------
                // if(tensorIsNan){
                //     std::cout << "\x1b[33m\nTensor is [nan]\x1b[0m" << std::endl;
                //     break;
                // }// end if(tensorIsNan)
                //--------------------------
                auto _test_loss = network_validation(std::move(data_loader_test));
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
                _learning_elements.push_back(_element_sum);
                //--------------------------
                auto printing_threads = std::async(std::launch::async, loss_display, _test_loss, _element_sum);
                //--------------------------
                if (_learning_elements.size() > 2){
                    _learning = check_learning(_learning_elements, precision);
                    _learning_elements.clear();
                    printing_threads = std::async(std::launch::async, [&_learning](){
                                        printf("\n-----------------Learning:[%s]-----------------\n", (_learning) ? "True" : "False");});
                }// end if (_learning_elements.size > 4)
                //--------------------------
            } while(_learning and !tensorIsNan);
            //--------------------------
            return Loss;
            //--------------------------
        }// end std::vector<at::Tensor> NetworkHandling::network_train(DataLoader& data_loader, size_t& epoch)
        //--------------------------
        template <typename Dataset>
        std::vector<float> network_validation(Dataset&& data_loader){
            //--------------------------
            progressbar bar(std::distance(data_loader->begin(), data_loader->end()));
            //--------------------------
            std::vector<float> test_loss;
            //--------------------------
            std::cout << "\nValidation: ";
            //--------------------------
            for (const auto& batch : *data_loader){
                //--------------------------
                bar.update();
                //------------
                test_loss.emplace_back(network_validation_batch(std::move(batch)));
                //--------------------------
            }// end for (const auto& batch : data_loader)
            //--------------------------
            return test_loss;
            //--------------------------
        }// end std::vector<double> NetworkHandling::network_validation(DataLoader& data_loader)
        //--------------------------------------------------------------
        template <typename Dataset>
        std::vector<std::tuple<torch::Tensor, torch::Tensor, float>> network_test(Dataset&& data_loader){
            //--------------------------
            progressbar bar(std::distance(data_loader->begin(), data_loader->end()));
            //--------------------------
            std::vector<std::tuple<torch::Tensor, torch::Tensor, float>> results;
            //--------------------------
            std::cout << "\nValidation: ";
            //--------------------------
            for (const auto& batch : *data_loader){
                //--------------------------
                bar.update();
                //------------
                results.emplace_back(network_test_batch(std::move(batch)));
                //--------------------------
            }// end for (const auto& batch : data_loader)
            //--------------------------
            return results;
            //--------------------------
        }// end std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<float>> network_test(Dataset&& data_loader)
        //--------------------------------------------------------------
        template <typename T>
        bool check_learning(const std::vector<T>& elements, const long double tolerance = 1E-2L){
            //--------------------------
            long double average = std::reduce(std::execution::par_unseq, elements.begin(), elements.end(), 0.L) / elements.size();
            //--------------------------
            if (std::abs(average - elements.front()) <= tolerance){
                return false;
            }// end std::abs(average - elements.front()) <= tolerance)
            //--------------------------
            return true;
        }// end bool NetworkHandling::check_learning(const std::vector<double>& elements, const double tolerance)
        //--------------------------------------------------------------
        static void loss_display(const std::vector<float>& loss, const double& elements_sum);
        //--------------------------------------------------------------
};
//--------------------------------------------------------------