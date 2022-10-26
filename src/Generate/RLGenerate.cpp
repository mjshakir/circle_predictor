//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Generate/RLGenerate.hpp"
//--------------------------------------------------------------
// Standard cpp library
//--------------------------------------------------------------
#include <algorithm>
#include <execution>
//--------------------------------------------------------------
// User Defined library
//--------------------------------------------------------------
#include "Network/Normalize.hpp"
#include "Network/DataLoader.hpp"
//--------------------------------------------------------------
// Boost library
//--------------------------------------------------------------
// #include <boost/generator_iterator.hpp> // https://www.boost.org/doc/libs/1_62_0/libs/utility/generator_iterator.htm
//--------------------------------------------------------------
RLGenerate::RLGenerate( const double& radius, 
                        const size_t& generated_points, 
                        const std::tuple<double, double>& center,
                        const size_t& batch_size) : Generate(std::move(radius), std::move(generated_points), std::move(center)){
    //--------------------------
    auto [_input, _output] = get_data();
    //--------------------------
    // Generate your data set. At this point you can add transforms to you data set, e.g. stack your
    // batches into a single tensor.
    // auto data_set = DataLoader(std::move(std::get<0>(data)), std::move(std::get<1>(data))).map(torch::data::transforms::Normalize<>(0.5, 0.25)).map(torch::data::transforms::Stack<>());
    //--------------------------
    auto data_set = DataLoader( Normalize::normalization(_input), 
                                Normalize::normalization(_output)).map(torch::data::transforms::Stack<>());
    //--------------------------
    // Generate a data loader.
    //--------------------------
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>( std::move(data_set), 
                                                                                            torch::data::DataLoaderOptions(batch_size));
    //--------------------------
    auto data_loader_size = std::distance(data_loader->begin(), data_loader->end());
    //--------------------------
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> _full_data;
    _full_data.reserve(data_loader_size);
    //--------------------------
    for (const auto& batch : *data_loader){
        //--------------------------
        auto input_data = batch.data, target_data = batch.target;
        //--------------------------
        _full_data.push_back({input_data, target_data});
        //--------------------------
    } //for (const auto& batch : *data_loader)
    //--------------------------
    m_iter = _full_data.begin();
    m_iter_end = _full_data.end();
    //--------------------------
}// end RLGenerate::RLGenerate(const double& radius, const size_t& generated_points, const std::tuple<double, double>& center)
//--------------------------------------------------------------
double RLGenerate::internal_reward_function(const torch::Tensor& real_value, const torch::Tensor& predicted_value, const long double& tolerance) const{
    //--------------------------
    auto _real_value_temp = real_value.cpu();
    auto _real_value = _real_value_temp.accessor<float, 2>();
    auto _predicted_value_temp = predicted_value.cpu();
    auto _predicted_value = _predicted_value_temp.accessor<float, 2>();
    //--------------------------
    std::vector<double> _difference;
    _difference.reserve(_real_value.size(0));
    //--------------------------
    for(int64_t i = 0; i < _real_value.size(0); ++i){
        for(int64_t j = 0; i < _real_value.size(1); ++j){
            _difference.push_back(std::abs(_real_value[i][j] - _predicted_value[i][j]) / ((_real_value[i][j] >= _predicted_value[i][j]) ? _real_value[i][j] : _predicted_value[i][j]));
        }// for(int64_t j = 0; i < _real_value.size(1); ++j)
    }// for(int64_t i = 0; i < _real_value.size(0); ++i)
    //--------------------------
    double average_difference = std::reduce(std::execution::par_unseq, _difference.begin(), _difference.end(), 0.L) / _difference.size();
    //--------------------------
    if(average_difference > tolerance){
        return average_difference*10;
    }
    //--------------------------
    return 100.f;
    //--------------------------
}// end double RLGenerate::internal_reward_function(void)
//--------------------------------------------------------------
std::tuple<torch::Tensor, double, bool> RLGenerate::internal_step_function(const torch::Tensor& actions, const long double& tolerance){
    //--------------------------
    auto [_input, _target] = *m_iter;
    //--------------------------
    auto _reward = internal_reward_function(_target, actions, tolerance);
    //--------------------------
    if(m_iter == m_iter_end){
        //--------------------------
        return {_input, _reward, true};
        //--------------------------
    }// if(m_iter.end())
    //--------------------------
    ++m_iter;
    //--------------------------
    return {_input, _reward, false};
    //--------------------------
}// end std::vector<std::tuple<torch::Tensor, double>> RLGenerate::internal_step_function()
//--------------------------------------------------------------