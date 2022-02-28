#pragma once

//--------------------------------------------------------------
#include <torch/torch.h>
#include <future>
#include <algorithm>
#include <execution>
//--------------------------------------------------------------
#include "Timing/TimeIT.hpp"
//--------------------------------------------------------------
#include "fort.hpp"
//--------------------------------------------------------------
#include "progressbar/include/progressbar.hpp"
//--------------------------------------------------------------

template<typename Network>
class NetworkHandling{
    public:
        //--------------------------------------------------------------
        NetworkHandling() = delete;
        //--------------------------
        /**
         *  @brief A constructor 
         *
         *  @tparam model: A torch network struct that inherits from torch::nn::Module.
         *  @tparam device  torch::Device cpu or gpu.
         */
        NetworkHandling(Network& model, const torch::Device& device): m_model(model), m_device(device){
            //--------------------------
        }// end NetworkHandling(Network& model, torch::Device& device)
        //--------------------------
        /**
         *  @brief A constructor 
         *
         *  @tparam model: A torch network struct that inherits from torch::nn::Module.
         *  @tparam device  torch::Device cpu or gpu.
         */
        NetworkHandling(Network&& model, const torch::Device& device): m_model(std::move(model)), m_device(device){
            //--------------------------
        }// end NetworkHandling(Network& model, torch::Device& device)
        //--------------------------
        /**
         *  @brief Train the model with fix epoch iteration
         *
         *  @tparam data_loader: A torch dataloader.
         *  @tparam optimizer:  torch::optim::Optimizer.
         *  @tparam epoch:  How many iteration to train.
         *  @return vector of float
         */
        template <typename Dataloader>
        std::vector<float> train(Dataloader&& data_loader, torch::optim::Optimizer& optimizer, const size_t& epoch){
            //--------------------------
            return network_train(std::move(data_loader), optimizer, epoch);
            //--------------------------
        }// end std::vector<float> train(Dataloader&& data_loader, torch::optim::Optimizer& optimizer, const size_t& epoch)
        //--------------------------
        /**
         *  @brief Train the model with a validation set. The training stop when precision is hit
         *
         *  @tparam data_loader: A torch dataloader.
         *  @tparam data_loader_test: A torch dataloader.
         *  @tparam optimizer:  torch::optim::Optimizer.
         *  @tparam precision:  a value that will stop the training once hit.
         * 
         *  @return vector of float
         */
        template <typename Dataloader, typename Test_Dataloader>
        std::vector<float> train(Dataloader&& data_loader, Test_Dataloader&& data_loader_test, torch::optim::Optimizer& optimizer, long double precision = 1E-2L){
            //--------------------------
            return network_train(std::move(data_loader), std::move(data_loader_test), optimizer, precision);
            //--------------------------
        }// end  std::vector<float> train(Dataloader&& data_loader, Test_Dataloader&& data_loader_test, torch::optim::Optimizer& optimizer, float precision = 10)
        //--------------------------
        /**
         *  @brief Run a validation set on the modal
         *
         *  @tparam data_loader: A torch dataloader.
         * 
         *  @return vector of float
         */
        template <typename Dataset>
        std::vector<float> validation(Dataset&& data_loader){
            //--------------------------
            return network_validation(std::move(data_loader));
            //--------------------------
        }// end std::vector<float> validation(Dataset&& data_loader)
        //--------------------------------------------------------------
        /**
         *  @brief Run a test set on the modal
         *
         *  @tparam data_loader: A torch dataloader.
         * 
         *  @return vector of tuples
         *  1) torch::Tensor: Original targets set
         *  2) torch::Tensor: Results set
         *  3) float: Set Loss
         */
        template <typename Dataset>
        std::vector<std::tuple<torch::Tensor, torch::Tensor, float>> test(Dataset&& data_loader){
            //--------------------------
            return network_test(std::move(data_loader));
            //--------------------------
        }// end std::vector<std::tuple<torch::Tensor, torch::Tensor, float>> test(Dataset&& data_loader)
        //--------------------------------------------------------------
    protected:
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
            // output = torch::transpose(output.view({2,-1}), 0, 1);
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
            m_model.eval();
            //--------------------------
            auto data = batch.data.to(m_device), targets = batch.target.to(m_device);
            auto output = m_model.forward(data);
            // output = torch::transpose(output.view({2,-1}), 0, 1);
            //--------------------------
            return torch::mse_loss(output, targets, torch::Reduction::Sum).template item<double>();
            //--------------------------
        }// end float network_validation_batch(Batch&& batch)
        //--------------------------
        template <typename Batch>
        std::tuple<torch::Tensor, torch::Tensor, float> network_test_batch(Batch&& batch){
            //--------------------------
            torch::NoGradGuard no_grad;
            m_model.eval();
            //--------------------------
            auto data = batch.data.to(m_device), targets = batch.target.to(m_device);
            auto output = m_model.forward(data);
            // output = torch::transpose(output.view({2,-1}), 0, 1);
            //--------------------------
            return {targets, output, torch::mse_loss(output, targets, torch::Reduction::Sum).template item<float>()};
            //--------------------------
        }// end std::tuple<torch::Tensor, torch::Tensor, float> network_test_batch(Batch&& batch)
        //--------------------------
        template <typename Dataloader>
        std::vector<float> network_train(Dataloader&& data_loader, torch::optim::Optimizer& optimizer, const size_t& epoch){
            //--------------------------
            auto data_loader_size = std::distance(data_loader->begin(), data_loader->end());
            //--------------------------
            bool tensorIsNan = false;
            //--------------------------
            std::vector<float> Loss;
            //--------------------------
            torch::optim::StepLR _scheduler(optimizer, 30, 1E-2);
            //--------------------------
            for (size_t i = 0; i < epoch; i++){
                //--------------------------
                TimeIT _timer;
                //--------------------------
                progressbar bar(data_loader_size);
                //--------------------------
                std::cout << "Training: ";
                //--------------------------
                for (const auto& batch : *data_loader){
                    //--------------------------
                    bar.update();
                    //------------
                    Loss.push_back(network_train_batch(std::move(batch), optimizer, &tensorIsNan));
                    //--------------------------
                    if(tensorIsNan){
                        std::cout << "\n\x1b[33m\033[1mTensor is [nan]\033[0m\x1b[0m" << std::endl;
                        break;
                    }// end if(tensorIsNan)
                    //--------------------------
                }// end for (const auto& batch : *data_loader)
                //--------------------------
                if(tensorIsNan){
                    break;
                }// end if(tensorIsNan)
                //--------------------------
                _scheduler.step();
                //--------------------------
                auto printing_threads = std::async(std::launch::async, [&](){loss_display(Loss, _timer.get_time());
                                                                             Loss.clear();                                       });
                //--------------------------
            }// end for (size_t i = 0; i < epoch; i++)
            //--------------------------
            return Loss;
            //--------------------------
        }// end std::vector<float> network_train(Dataloader&& data_loader, torch::optim::Optimizer& optimizer, const size_t& epoch)
        //--------------------------
        template <typename Dataloader, typename Test_Dataloader, typename R>
        std::vector<float> network_train(Dataloader&& data_loader, Test_Dataloader&& data_loader_test, torch::optim::Optimizer& optimizer, const R& precision){
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
                TimeIT _timer;
                //--------------------------
                for (const auto& batch : *data_loader){
                    //--------------------------
                    bar.update();
                    //------------
                    Loss.push_back(network_train_batch(std::move(batch), optimizer, &tensorIsNan));
                    //--------------------------
                    if(tensorIsNan){
                        std::cout << "\n\x1b[33m\033[1mTensor is [nan]\033[0m\x1b[0m" << std::endl;
                        break;
                    }// end if(tensorIsNan)
                //--------------------------
                }// end for (const auto& batch : *data_loader)
                //--------------------------
                _scheduler.step();
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
                auto printing_threads = std::async(std::launch::async, [&](){loss_display(_test_loss, _element_sum, _timer.get_time());});
                //--------------------------
                if (_learning_elements.size() > 2){
                    _learning = check_learning(_learning_elements, precision);
                    _learning_elements.clear();
                    printing_threads = std::async(std::launch::async, [&_learning](){
                                    if(_learning){
                                        printf("\n\x1b[33m-----------------Learning:[%s]-----------------\x1b[0m\n", (_learning) ? "True" : "False");
                                    }// end if(_learning)
                                    else{
                                        printf("\n\x1b[36m-----------------Learning:[%s]-----------------\x1b[0m\n", (_learning) ? "True" : "False");
                                    }// end else
                                });
                }// end if (_learning_elements.size > 2)
                //--------------------------
            } while(_learning and !tensorIsNan);
            //--------------------------
            return Loss;
            //--------------------------
        }// end std::vector<float> network_train(Dataloader&& data_loader, Test_Dataloader&& data_loader_test, torch::optim::Optimizer& optimizer, const R& precision)
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
                test_loss.push_back(network_validation_batch(std::move(batch)));
                //--------------------------
            }// end for (const auto& batch : *data_loader)
            //--------------------------
            return test_loss;
            //--------------------------
        }// end std::vector<float> network_validation(Dataset&& data_loader)
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
        }// end std::vector<std::tuple<torch::Tensor, torch::Tensor, float>> network_test(Dataset&& data_loader)
        //--------------------------------------------------------------
    private:
        //--------------------------
        Network m_model; 
        torch::Device m_device;
        //--------------------------
        template <typename T, typename R>
        bool check_learning(const std::vector<T>& elements, const R& tolerance){
            //--------------------------
            long double average = std::reduce(std::execution::par_unseq, elements.begin(), elements.end(), 0.L) / elements.size();
            //--------------------------
            if (std::abs(average - elements.front()) <= tolerance){
                return false;
            }// end std::abs(average - elements.front()) <= tolerance)
            //--------------------------
            return true;
            //--------------------------
        }// end bool check_learning(const std::vector<T>& elements, const R& tolerance)
        //--------------------------------------------------------------
        template <typename T, typename R>
        void loss_display(const std::vector<T>& loss, const R& ns_time){
            //--------------------------
            double elements_sum = std::reduce(std::execution::par_unseq, loss.begin(), loss.end(), 0.L);
            auto _max_element = std::max_element(std::execution::par_unseq, loss.begin(), loss.end());
            auto _min_element = std::min_element(std::execution::par_unseq, loss.begin(), loss.end());
            //--------------------------
            fort::char_table table;
            //--------------------------
            // Change border style
            //--------------------------
            table.set_border_style(FT_NICE_STYLE);
            //--------------------------
            table   << fort::header
                    << "Sum Loss" << "Min Position" << "Min loss" << "Max Position" << "Max loss" << "Execution time [ns]" << fort::endr
                    << elements_sum
                    << std::distance(loss.begin(), _min_element)
                    << *_min_element
                    << std::distance(loss.begin(), _max_element)
                    << *_max_element 
                    << ns_time << fort::endr;
            //--------------------------
            // Set center alignment for the 1st and 3rd columns
            //--------------------------
            table.column(1).set_cell_text_align(fort::text_align::center);
            table.column(3).set_cell_text_align(fort::text_align::center);
            table.column(5).set_cell_text_align(fort::text_align::center);
            table.column(5).set_cell_content_fg_color(fort::color::red);
            //--------------------------
            std::cout << "\n" << table.to_string() << std::endl;
            //--------------------------
        }// end void loss_display(const std::vector<T>& loss, const R& ns_time)
        //--------------------------------------------------------------
        template <typename T, typename D, typename R>
        void loss_display(const std::vector<T>& loss, const D& elements_sum, const R& ns_time){
            //--------------------------
            auto _max_element = std::max_element(std::execution::par_unseq, loss.begin(), loss.end());
            auto _min_element = std::min_element(std::execution::par_unseq, loss.begin(), loss.end());
            //--------------------------
            fort::char_table table;
            //--------------------------
            // Change border style
            //--------------------------
            table.set_border_style(FT_NICE_STYLE);
            //--------------------------
            table   << fort::header
                    << "Loss Sum" << "Min Position" << "Min loss" << "Max Position" << "Max loss" << "Execution time [ns]" << fort::endr
                    << elements_sum
                    << std::distance(loss.begin(), _min_element)
                    << *_min_element
                    << std::distance(loss.begin(), _max_element)
                    << *_max_element 
                    << ns_time << fort::endr;
            //--------------------------
            // Set center alignment for the 1st and 3rd columns
            //--------------------------
            table.column(1).set_cell_text_align(fort::text_align::center);
            table.column(3).set_cell_text_align(fort::text_align::center);
            table.column(5).set_cell_text_align(fort::text_align::center);
            table.column(5).set_cell_content_fg_color(fort::color::red);
            //--------------------------
            std::cout << "\n" << table.to_string() << std::endl;
            //--------------------------
        }// end void loss_display(const std::vector<T>& loss, const D& elements_sum, const R& ns_time)
        //--------------------------------------------------------------
};
//--------------------------------------------------------------