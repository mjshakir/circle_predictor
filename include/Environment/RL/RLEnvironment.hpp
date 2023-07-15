#pragma once
//--------------------------------------------------------------
/* From: https://stackoverflow.com/questions/14803112/short-way-to-stdbind-member-function-to-object-instance-without-binding-param 
    and https://stackoverflow.com/questions/70355767/binding-a-class-method-to-a-method-of-another-class 
    and https://stackoverflow.com/questions/28746744/passing-capturing-lambda-as-function-pointer */
//--------------------------------------------------------------
// LibTorch library
//--------------------------------------------------------------
#include <torch/torch.h>
//--------------------------------------------------------------
// User definition
//--------------------------------------------------------------
#define OUT
//--------------------------------------------------------------
namespace RL {
    //--------------------------------------------------------------
    namespace Environment {
        //--------------------------------------------------------------
        template<typename T, typename COST_OUTPUT, typename... Args>
        //--------------------------------------------------------------
        class RLEnvironment {
            //--------------------------------------------------------------
            public:
                //--------------------------------------------------------------
                RLEnvironment(void) = delete;
                //--------------------------------------------------------------
                virtual ~RLEnvironment(void) = default;
                //--------------------------------------------------------------
                /**
                 * @brief Construct to create a training environment for reinforcement learning 
                 * 
                 * @param data          [in] : Data Vector of the template type T                             
                 *                          @example: RLEnvironment<torch::Tensor,..., ...> foo(data, ...) this will results in std::vector<torch::Tensor>
                 * @param costFunction  [in] : Cost function that return the reward. needs to define the function return and parameters 
                 *                          @example: RLEnvironment<..., torch::Tensor,double, int> foo(..., [](double x, int y){return torch::tensor(x*y);})
                 * @param egreedy       [in] : The starting egreedy                        @default: 0.9
                 * @param egreedy_final [in] : The egreedy number where it will change     @default: 0.02
                 * @param egreedy_decay [in] : The egreedy exponential (e^x) decay factor  @default: 500.
                 * 
                 * @throws std::runtime_error
                 */
                explicit RLEnvironment( std::vector<T>&& data, 
                                        std::function<COST_OUTPUT(const Args&...)> costFunction,
                                        const double& egreedy = 0.9,
                                        const double& egreedy_final = 0.02,
                                        const double& egreedy_decay = 500.) :   m_data(std::move(data)),
                                                                                m_data_iter(m_data.begin()), 
                                                                                m_CostFunction(std::move(costFunction)),
                                                                                m_egreedy(egreedy),
                                                                                m_egreedy_final(egreedy_final),
                                                                                m_egreedy_decay(egreedy_decay){
                    //----------------------------
                    if(egreedy_decay == 0.){
                        //----------------------------
                        throw std::runtime_error("Egreedy Decay Cannot Be Zero. egreedy_decay Value: [" + std::to_string(egreedy_decay) + "]");
                        //----------------------------
                    }// end if(egreedy_decay == 0.)
                    //----------------------------
                }// end RLEnvironment(Dataset&& data_loader)
                //--------------------------------------------------------------
                /**
                 * @brief Construct to create a training environment for reinforcement learning 
                 * 
                 * @param data          [in] : Data Vector of the template type T                             
                 *                          @example: RLEnvironment<torch::Tensor,..., ...> foo(data, ...) this will results in std::vector<torch::Tensor>
                 * @param costFunction  [in] : Cost function that return the reward. needs to define the function return and parameters 
                 *                          @example: RLEnvironment<..., torch::Tensor,double, int> foo(..., [](double x, int y){return torch::tensor(x*y);})
                 * @param egreedy       [in] : The starting egreedy                        @default: 0.9
                 * @param egreedy_final [in] : The egreedy number where it will change     @default: 0.02
                 * @param egreedy_decay [in] : The egreedy exponential (e^x) decay factor  @default: 500.
                 * 
                 * @throws std::runtime_error
                 */
                explicit RLEnvironment( const std::vector<T>& data, 
                                        const std::function<COST_OUTPUT(const Args&...)>& costFunction,
                                        const double& egreedy = 0.9,
                                        const double& egreedy_final = 0.02,
                                        const double& egreedy_decay = 500.) :   m_data(data),
                                                                                m_data_iter(m_data.begin()), 
                                                                                m_CostFunction(costFunction),
                                                                                m_egreedy(egreedy),
                                                                                m_egreedy_final(egreedy_final),
                                                                                m_egreedy_decay(egreedy_decay){
                    //----------------------------
                    if(egreedy_decay == 0.){
                        //----------------------------
                        throw std::runtime_error("Egreedy Decay Cannot Be Zero. egreedy_decay Value: [" + std::to_string(egreedy_decay) + "]");
                        //----------------------------
                    }// end if(egreedy_decay == 0.)
                    //----------------------------
                }// end RLEnvironment(Dataset&& data_loader)
                //--------------------------------------------------------------
                //Define copy constructor explicitly
                RLEnvironment(const RLEnvironment& other) : m_data(other.m_data),
                                                            m_data_iter(other.m_data_iter),
                                                            m_CostFunction(other.m_CostFunction),
                                                            m_egreedy(other.m_egreedy),
                                                            m_egreedy_final(other.m_egreedy_final),
                                                            m_egreedy_decay(other.m_egreedy_decay) {
                    //--------------------------
                }// end RLEnvironment(const RLEnvironment& other)
                //--------------------------------------------------------------
                 //Copy assignment operator
                RLEnvironment& operator=(const RLEnvironment& other) {
                    //--------------------------
                    // Check for self-assignment
                    if (this == &other) {
                        return *this;
                    }// end if (this == &other)
                    //--------------------------
                    // Perform a deep copy of the data
                    m_data          = other.m_data;
                    m_data_iter     = other.m_data_iter;
                    m_CostFunction  = other.m_CostFunction;
                    m_egreedy       = other.m_egreedy;
                    m_egreedy_final = other.m_egreedy_final;
                    m_egreedy_decay = other.m_egreedy_decay;
                    //--------------------------
                    return *this;
                    //--------------------------
                }// end RLEnvironmentLoader& operator=(const RLEnvironmentLoader& other)
                //--------------------------------------------------------------
                /**
                 * @brief 
                 * 
                 * @param args 
                 * @return std::tuple<torch::Tensor, COST_OUTPUT, double, bool> 
                 */
                virtual std::tuple<torch::Tensor, COST_OUTPUT, double, bool> step(const Args&... args){
                    //----------------------------
                    return internal_step(args...);
                    //----------------------------
                }// std::tuple<torch::Tensor, COST_OUTPUT, double, bool> step(const Args&... args)
                //--------------------------------------------------------------
                /**
                 * @brief 
                 * 
                 * @param epsilon 
                 * @param done 
                 * @param args 
                 * @return std::tuple<torch::Tensor, COST_OUTPUT> 
                 */
                virtual std::tuple<torch::Tensor, COST_OUTPUT> step(OUT double& epsilon, OUT bool& done, const Args&... args){
                    //----------------------------
                    return internal_step(epsilon, done, args...);
                    //----------------------------
                }// std::tuple<torch::Tensor, COST_OUTPUT> step(OUT double& epsilon, OUT bool& done, const Args&... args))
                //--------------------------------------------------------------
                /**
                 * @brief Get the first object
                 * 
                 * @return std::tuple<torch::Tensor, double> 
                 */
                virtual std::tuple<torch::Tensor, double> get_first(void){
                    //----------------------------
                    return get_first_internal();
                    //----------------------------
                }// end torch::Tensor get_first(void)
                //--------------------------------------------------------------
                /**
                 * @brief Get the first object
                 * 
                 * @param epsilon 
                 * @return torch::Tensor 
                 */
                virtual torch::Tensor get_first(OUT double& epsilon){
                    //----------------------------
                    return get_first_internal(epsilon);
                    //----------------------------
                }// end torch::Tensor get_first(void)
                //--------------------------------------------------------------
                /**
                 * @brief 
                 * 
                 */
                virtual void reset(void){
                    //--------------------------
                    reset_iterator();
                    //--------------------------
                }// end void reset(void)
                //--------------------------------------------------------------
                /**
                 * @brief 
                 * 
                 */
                constexpr size_t size(void) const{
                    //--------------------------
                    return vector_size();
                    //--------------------------
                }// end constexpr size_t size(void) const
                //--------------------------------------------------------------
            protected:
                //--------------------------------------------------------------
                virtual std::tuple<torch::Tensor, COST_OUTPUT, double, bool> internal_step(const Args&... args){
                    //--------------------------
                    if (m_data_iter == m_data.end()){
                        //--------------------------
                        throw std::out_of_range("End Of The Data Iterator");
                        //--------------------------
                    }// end if (m_data_iter == m_data.end())
                    //--------------------------
                    if (m_data_iter == m_data.begin()){
                        //--------------------------
                        torch::Tensor _data;
                        double epsilon;
                        //--------------------------
                        std::tie(_data, epsilon) = get_first_internal();
                        //--------------------------
                        return {_data, torch::tensor(0), epsilon, false};
                        //--------------------------
                    }// end if (m_data_iter == m_data.begin())
                    //--------------------------
                    if(m_data_iter == m_data.end()-1){
                        //--------------------------
                        return {*m_data_iter, m_CostFunction(args...), calculate_epsilon(), true};
                        //--------------------------
                    }// if(m_data_iter == m_data.end())
                    //--------------------------
                    return {*m_data_iter++, m_CostFunction(args...), calculate_epsilon(), false};
                    //--------------------------
                }// end std::tuple<torch::Tensor, COST_OUTPUT, double, bool> internal_step(Args... args))
                //--------------------------------------------------------------
                virtual std::tuple<torch::Tensor, COST_OUTPUT> internal_step(OUT double& epsilon, OUT bool& done, const Args&... args){
                    //--------------------------
                    if (m_data_iter == m_data.end()){
                        //--------------------------
                        throw std::out_of_range("End Of The Data Iterator");
                        //--------------------------
                    }// end if (m_data_iter == m_data.end())
                    //--------------------------
                    if (m_data_iter == m_data.begin()){
                        //--------------------------
                        done = false;
                        //--------------------------
                        torch::Tensor _data = get_first_internal(epsilon);
                        //--------------------------
                        return {_data, torch::tensor(0)};
                        //--------------------------
                    }// end if (m_data_iter == m_data.begin())
                    //--------------------------
                    if(m_data_iter == m_data.end()-1){
                        //--------------------------
                        epsilon = calculate_epsilon();
                        done = true;
                        //--------------------------
                        return {*m_data_iter, m_CostFunction(args...)};
                        //--------------------------
                    }// if(m_data_iter == m_data.end())
                    //--------------------------
                    epsilon = calculate_epsilon();
                    done = false;
                    //--------------------------
                    return {*m_data_iter++, m_CostFunction(args...)};
                    //--------------------------
                }// end std::tuple<torch::Tensor, COST_OUTPUT> internal_step(double& epsilon, bool& done, Args... args)
                //--------------------------------------------------------------
                virtual std::tuple<torch::Tensor, double> get_first_internal(void){
                    //--------------------------
                    if (m_data_iter == m_data.end()){
                        //--------------------------
                        throw std::out_of_range("End Of The Data Iterator");
                        //--------------------------
                    }// end if (m_data_iter == m_data.end() or std::next(m_data_iter, batch) == m_data.end())
                    //--------------------------
                    if (m_data_iter == m_data.begin()){
                        //--------------------------
                        auto input = *m_data_iter;
                        //--------------------------
                        auto epsilon = calculate_epsilon();
                        //--------------------------
                        ++m_data_iter;
                        //--------------------------
                        return {input, epsilon};
                        //--------------------------
                    }// end if (m_data_iter == m_data.begin())
                    //--------------------------
                    return {torch::tensor(0), 0};
                    //--------------------------
                }// end torch::Tensor get_first_internal(void)
                //--------------------------------------------------------------
                virtual torch::Tensor get_first_internal(OUT double& epsilon){
                    //--------------------------
                    if (m_data_iter == m_data.end()){
                        //--------------------------
                        throw std::out_of_range("End Of The Data Iterator");
                        //--------------------------
                    }// end if (m_data_iter == m_data.end() or std::next(m_data_iter, batch) == m_data.end())
                    //--------------------------
                    if (m_data_iter == m_data.begin()){
                        //--------------------------
                        auto input = *m_data_iter;
                        //--------------------------
                        epsilon = calculate_epsilon();
                        //--------------------------
                        ++m_data_iter;
                        //--------------------------
                        return input;
                        //--------------------------
                    }// end if (m_data_iter == m_data.begin())
                    //--------------------------
                    epsilon = 0.;
                    //--------------------------
                    return torch::tensor(0);
                    //--------------------------
                }// end torch::Tensor get_first_internal(void)
                //--------------------------------------------------------------
                virtual void reset_iterator(void){
                    //--------------------------
                    m_data_iter = m_data.begin();
                    //--------------------------
                }// end void rest_iterator(void)
                //--------------------------------------------------------------
                constexpr double calculate_epsilon(void){
                    //--------------------------
                    return m_egreedy_final + (m_egreedy - m_egreedy_final) * std::exp(-1. * std::distance(m_data.begin(), m_data_iter) / m_egreedy_decay);
                    //--------------------------
                }// end double calculate_epsilon()
                //--------------------------------------------------------------
                std::vector<T>& get_data(void){
                    //--------------------------
                    return m_data;
                    //--------------------------
                }// end std::vector<T>& get_data(void)
                //--------------------------------------------------------------
                std::function<COST_OUTPUT(const Args&...)>& get_cost_function(void){
                    //--------------------------
                    return m_CostFunction;
                    //--------------------------
                }// end std::function<COST_OUTPUT(const Args&...)>& get_cost_function(void)
                //--------------------------------------------------------------
                typename std::vector<T>::iterator& get_iterator(void){
                    //--------------------------
                    return m_data_iter;
                    //--------------------------
                }// end typename std::vector<T>::iterator& get_iterator(void)
                //--------------------------------------------------------------
                constexpr size_t vector_size(void) const{
                    //--------------------------
                    return m_data.size();
                    //--------------------------
                }// end typename std::vector<T>::iterator& get_iterator(void)
                //--------------------------------------------------------------
            private:
                //--------------------------------------------------------------
                std::vector<T> m_data;
                typename std::vector<T>::iterator m_data_iter;
                //--------------------------
                std::function<COST_OUTPUT(const Args&...)> m_CostFunction;
                //--------------------------
                double m_egreedy, m_egreedy_final, m_egreedy_decay;
            //--------------------------------------------------------------
        };// end class RLEnvironment
        //--------------------------------------------------------------
    }// end namespace Environment
    //--------------------------------------------------------------
}// end namespace RL
//--------------------------------------------------------------