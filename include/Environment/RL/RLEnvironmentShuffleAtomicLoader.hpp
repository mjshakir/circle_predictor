#pragma once
//--------------------------------------------------------------
// User Defined library
//--------------------------------------------------------------
#include "Environment/RL/RLEnvironmentShuffleAtomic.hpp"
//--------------------------------------------------------------
namespace RL {
    //--------------------------------------------------------------
    namespace Environment {
        //--------------------------------------------------------------
        template<typename T, typename COST_OUTPUT, typename... Args>
        //--------------------------------------------------------------
        class RLEnvironmentShuffleAtomicLoader : public RLEnvironmentShuffleAtomic<T, COST_OUTPUT, Args...>{
            //--------------------------------------------------------------
            public:
                //--------------------------------------------------------------
                RLEnvironmentShuffleAtomicLoader(void) = delete;
                //--------------------------------------------------------------
                virtual ~RLEnvironmentShuffleAtomicLoader(void) = default;
                //--------------------------------------------------------------
                /**
                * @brief Constructs an RLEnvironmentLoader object.
                * This class represents a loader for reinforcement learning environments. 
                * It takes as input a vector of data, a cost function, 
                * and optional parameters for epsilon-greedy exploration. 
                * The data is moved into the object, and the cost function and other parameters are stored for later use.
                * 
                * @tparam data The vector of data representing the RL environment.
                * @tparam costFunction The cost function to evaluate the RL environment.
                * @param egreedy The initial value of epsilon-greedy exploration @default: 0.9.
                * @param egreedy_final The final value of epsilon-greedy exploration @default: 0.02.
                * @param egreedy_decay The decay rate of epsilon-greedy exploration @default: 500.0.
                * @param batch The size of the batch for RL training @default: 1.
                * 
                * @note The template arguments T and Args represent the data type and additional argument types required by the cost function, respectively.
                * @note The data vector is moved into the RLEnvironmentLoader object, ensuring efficient data transfer.
                * @note The RLEnvironmentLoader class is derived from RLEnvironment and initializes its base class using the given parameters.
                * 
                * @example
                * // Create a vector of data representing the RL environment
                * std::vector<int> data = {1, 2, 3, 4, 5};
                * // Define a cost function
                * auto costFunction = [](const int& state) { return state * state; };
                * // Create an RLEnvironmentLoader object
                * RLEnvironmentLoader<int, int> loader(std::move(data), costFunction, 0.9, 0.02, 500.0, 1);
                *  @throws std::out_of_range If the specified batch size is greater than or equal to half the size of the data vector.
                *          The exception message provides details about the expected range for the batch size.
                */
                explicit RLEnvironmentShuffleAtomicLoader(std::vector<T>&& data, 
                                                          std::function<COST_OUTPUT(const Args&...)>&& costFunction,
                                                          const size_t& batch = 2UL,
                                                          const double& egreedy = 0.9,
                                                          const double& egreedy_final = 0.02,
                                                          const double& egreedy_decay = 500.) : RLEnvironmentShuffleAtomic<T, COST_OUTPUT, Args...>(std::move(data),
                                                                                                                                                    std::move(costFunction),
                                                                                                                                                    egreedy,
                                                                                                                                                    egreedy_final,
                                                                                                                                                    egreedy_decay),
                                                                                                m_data(this->RLEnvironmentAtomic<T, COST_OUTPUT, Args...>::get_data()),
                                                                                                m_data_iter(this->RLEnvironmentAtomic<T, COST_OUTPUT, Args...>::get_iterator()),
                                                                                                m_CostFunction(costFunction),
                                                                                                m_distribution(this->get_distribution()),
                                                                                                m_batch(batch){
                    //----------------------------
                    if(this->get_atomic_iterator() != m_data.begin()){
                        //----------------------------
                        throw std::out_of_range("Iterator is not in the beginning");
                        //----------------------------
                    }// end if(this->get_atomic_iterator() != m_data.begin())
                    //----------------------------
                }// end RLEnvironmentShuffleAtomicLoader(Dataset&& data_loader)
                //--------------------------------------------------------------
                //Define copy constructor explicitly
                RLEnvironmentShuffleAtomicLoader(const RLEnvironmentShuffleAtomicLoader& other) :   RLEnvironmentShuffleAtomic<T, COST_OUTPUT, Args...>(other),
                                                                                                    m_data(other.m_data),
                                                                                                    m_data_iter(this->RLEnvironmentAtomic<T, COST_OUTPUT, Args...>::get_iterator()),
                                                                                                    m_CostFunction(other.m_CostFunction),
                                                                                                    m_distribution(other.m_distribution),
                                                                                                    m_batch(other.m_batch) {
                    //--------------------------
                }// end RLEnvironmentShuffleAtomicLoader(const RLEnvironmentShuffleAtomicLoader& other)
                //--------------------------------------------------------------
                //Copy assignment operator
                RLEnvironmentShuffleAtomicLoader& operator=(const RLEnvironmentShuffleAtomicLoader& other) {
                    //--------------------------
                    // Check for self-assignment
                    if (this == &other) {
                        return *this;
                    }// end if (this == &other)
                    //--------------------------
                    // Perform a deep copy of the data
                    RLEnvironmentAtomic<T, COST_OUTPUT, Args...>::operator=(other);
                    m_data          = other.m_data;
                    m_data_iter     = this->RLEnvironmentAtomic<T, COST_OUTPUT, Args...>::get_iterator();
                    m_CostFunction  = other.m_CostFunction;
                    m_distribution  = other.m_distribution;
                    m_batch         = other.m_batch;
                    //--------------------------
                    return *this;
                    //--------------------------
                }// end RLEnvironmentLoader& operator=(const RLEnvironmentLoader& other)
                //--------------------------------------------------------------
                /**
                * @brief Executes an internal step in the RL environment.
                *
                * This function performs an internal step in the RL environment, processing a batch of data
                * and returning a tuple containing the resulting tensor, cost output, epsilon value, and a flag
                * indicating whether the data iteration is complete.
                *
                * @tparam args The variadic arguments for the cost function.
                *
                * @return A tuple containing the resulting tensor, cost output, epsilon value, and a flag indicating
                *         whether the data iteration is complete.
                *
                * @throws std::out_of_range If the current data iterator is at the end or if advancing the iterator by
                *                          the batch size exceeds the end of the data.
                *
                * @note The data vector should be populated before calling this function.
                * @note The cost function should be set before calling this function.
                *
                * @example
                * size_t batchSize = 4;
                * RLEnvironmentLoader loader(data, costFunction, 0.9, 0.02, 500.0, batchSize);
                * 
                * auto result = loader.step(arg1, arg2);
                *
                * // Example usage of the returned tuple elements
                * torch::Tensor tensor = std::get<0>(result);
                * COST_OUTPUT costOutput = std::get<1>(result);
                * double epsilon = std::get<2>(result);
                * bool isComplete = std::get<3>(result);
                */
                virtual std::tuple<torch::Tensor, COST_OUTPUT, double, bool> step(const Args&... args) override {
                    //----------------------------
                    return internal_step(m_batch, args...);
                    //----------------------------
                }// std::tuple<torch::Tensor, COST_OUTPUT, double, bool> step(const Args&... args)
                //--------------------------------------------------------------
                /**
                * @brief Performs an step in the algorithm.
                *
                * This function processes a batch of data, updates the epsilon value and done flag, and returns a tuple
                * containing a torch::Tensor representing the processed data and a COST_OUTPUT representing the cost output.
                *
                * @tparam Args The template parameter pack representing the additional arguments used for cost function calculations.
                * @param[out] epsilon The updated epsilon value based on internal calculations.
                * @param[out] done Indicates whether the internal step is completed or not.
                * @param batch The size of the batch to process.
                * @param args Additional arguments of type Args used for cost function calculations.
                * @return A tuple containing a torch::Tensor and a COST_OUTPUT.
                *     - The torch::Tensor represents the processed data obtained by concatenating the tensors in _data.
                *     - The COST_OUTPUT represents the cost output calculated based on the input arguments.
                * @throws std::out_of_range If the end of the data iterator is reached before processing the full batch.
                *
                * @details
                * The `internal_step` function is a member function of the class that performs an internal step in the algorithm.
                * It processes a batch of data and performs different actions based on the current state of the data iterator and the
                * batch size. The function updates the epsilon value and done flag according to the processing results and returns a
                * tuple containing the processed data and the cost output.
                *
                * The function starts by checking if the end of the data iterator or the end of the batch is reached. If either condition
                * is true, it throws a std::out_of_range exception with an appropriate error message.
                *
                * Next, the function initializes an empty vector `_data` and reserves space for `batch` tensors to optimize memory allocation.
                *
                * If the data iterator is at the beginning and the end of the batch is not the end of the data, the function calculates the
                * epsilon value using the `calculate_epsilon()` member function, sets the done flag to false, and adds the current data
                * tensor to `_data`. It then iterates over the remaining data tensors, adding them to `_data`. Finally, it returns a tuple
                * containing the concatenated tensor of `_data` along with a tensor of value 0 as the cost output.
                *
                * If the end of the batch is the end of the data iterator, the function iterates over the remaining data tensors, adds them
                * to `_data`, calculates the epsilon value, sets the done flag to true, and returns a tuple containing the concatenated
                * tensor of `_data` along with the cost output obtained by invoking the `m_CostFunction` member function with the provided
                * arguments `args`.
                *
                * If neither of the above cases is true, the function assumes that there are remaining data tensors to process. It iterates
                * over the batch size, adding the next data tensors to `_data`. It then calculates the epsilon value, sets the done flag to
                * false, and returns a tuple containing the concatenated tensor of `_data` along with the cost output obtained by invoking
                * the `m_CostFunction` member function with the provided arguments `args`.
                *
                * @note Make sure that the data iterator is properly initialized before calling this function.
                */
                virtual std::tuple<torch::Tensor, COST_OUTPUT> step(OUT double& epsilon, OUT bool& done, const Args&... args) override {
                    //----------------------------
                    return internal_step(epsilon, done, m_batch, args...);
                    //----------------------------
                }// std::tuple<torch::Tensor, COST_OUTPUT> step(OUT double& epsilon, OUT bool& done, const Args&... args))
                //--------------------------------------------------------------
                /**
                * @brief Executes an internal step in the RL environment.
                *
                * This function performs an internal step in the RL environment, processing a batch of data
                * and returning a tuple containing the resulting tensor, cost output, epsilon value, and a flag
                * indicating whether the data iteration is complete.
                *
                * @param batch The batch size indicating the number of data elements to process.
                * @tparam args The variadic arguments for the cost function.
                *
                * @return A tuple containing the resulting tensor, cost output, epsilon value, and a flag indicating
                *         whether the data iteration is complete.
                *
                * @throws std::out_of_range If the current data iterator is at the end or if advancing the iterator by
                *                          the batch size exceeds the end of the data.
                *
                * @note The data vector should be populated before calling this function.
                *
                * @note The cost function should be set before calling this function.
                *
                * @example
                * RLEnvironmentLoader loader(data, costFunction, 0.9, 0.02, 500.0);
                * size_t batchSize = 4;
                * auto result = loader.step(batchSize, arg1, arg2);
                *
                * // Example usage of the returned tuple elements
                * torch::Tensor tensor = std::get<0>(result);
                * COST_OUTPUT costOutput = std::get<1>(result);
                * double epsilon = std::get<2>(result);
                * bool isComplete = std::get<3>(result);
                */
                std::tuple<torch::Tensor, COST_OUTPUT, double, bool> step(const size_t& batch, const Args&... args) {
                    //----------------------------
                    return internal_step(batch, args...);
                    //----------------------------
                }// std::tuple<torch::Tensor, COST_OUTPUT> step(OUT double& epsilon, OUT bool& done, const size_t& batch, const Args&... args)
                //--------------------------------------------------------------
                /**
                * @brief Perform an internal step of the class.
                *
                * This function performs an internal step of the class, which involves processing a batch of data.
                * It returns a tuple containing a Torch tensor and a custom COST_OUTPUT type. The Torch tensor
                * represents the processed data batch, while the COST_OUTPUT type represents the cost output
                * calculated for the batch.
                *
                * @param epsilon [out] The epsilon value calculated during the step.
                * @param done [out] Indicates whether the step is completed or not.
                * @param batch The size of the batch to process.
                * @tparam args Additional arguments required for the cost function.
                *
                * @return A tuple containing the Torch tensor and the cost output.
                *
                * @throw std::out_of_range If the end of the data iterator is reached before processing the batch.
                * 
                *  @details
                * The `internal_step` function is a member function of the class that performs an internal step in the algorithm.
                * It processes a batch of data and performs different actions based on the current state of the data iterator and the
                * batch size. The function updates the epsilon value and done flag according to the processing results and returns a
                * tuple containing the processed data and the cost output.
                *
                * The function starts by checking if the end of the data iterator or the end of the batch is reached. If either condition
                * is true, it throws a std::out_of_range exception with an appropriate error message.
                *
                * Next, the function initializes an empty vector `_data` and reserves space for `batch` tensors to optimize memory allocation.
                *
                * If the data iterator is at the beginning and the end of the batch is not the end of the data, the function calculates the
                * epsilon value using the `calculate_epsilon()` member function, sets the done flag to false, and adds the current data
                * tensor to `_data`. It then iterates over the remaining data tensors, adding them to `_data`. Finally, it returns a tuple
                * containing the concatenated tensor of `_data` along with a tensor of value 0 as the cost output.
                *
                * If the end of the batch is the end of the data iterator, the function iterates over the remaining data tensors, adds them
                * to `_data`, calculates the epsilon value, sets the done flag to true, and returns a tuple containing the concatenated
                * tensor of `_data` along with the cost output obtained by invoking the `m_CostFunction` member function with the provided
                * arguments `args`.
                *
                * If neither of the above cases is true, the function assumes that there are remaining data tensors to process. It iterates
                * over the batch size, adding the next data tensors to `_data`. It then calculates the epsilon value, sets the done flag to
                * false, and returns a tuple containing the concatenated tensor of `_data` along with the cost output obtained by invoking
                * the `m_CostFunction` member function with the provided arguments `args`.
                *
                * @note Make sure that the data iterator is properly initialized before calling this function.
                *
                * Example usage:
                * @code{.cpp}
                * RLEnvironmentLoader loader(data, costFunction, 0.9, 0.02, 500.0);
                * double epsilon;
                * bool done;
                * auto result = loader.step(epsilon, done, 10, arg1, arg2, arg3);
                * torch::Tensor batch = std::get<0>(result);
                * COST_OUTPUT cost = std::get<1>(result);
                * @endcode
                */
                std::tuple<torch::Tensor, COST_OUTPUT> step(OUT double& epsilon, OUT bool& done, const size_t& batch, const Args&... args) {
                    //----------------------------
                    return internal_step(epsilon, done, batch, args...);
                    //----------------------------
                }// std::tuple<torch::Tensor, COST_OUTPUT> step(OUT double& epsilon, OUT bool& done, const size_t& batch, const Args&... args)
                //--------------------------------------------------------------
                /**
                 * @brief Get the first internal data batch.
                 *
                 * This function retrieves the first internal data batch from the class. It returns a tuple
                 * containing a Torch tensor and a double value. The Torch tensor represents the concatenated
                 * data batch, while the double value represents the epsilon calculated for the batch.
                 *
                 * @return A tuple containing the Torch tensor and epsilon value.
                 *
                 * @throw std::out_of_range If the end of the data iterator is reached before retrieving the batch.
                 *
                 * Example usage:
                 * @code{.cpp}
                 * RLEnvironmentLoader loader(data, costFunction, 0.9, 0.02, 500.0, batchSize);;
                 * auto result = loader.get_first();
                 * torch::Tensor batch = std::get<0>(result);
                 * double epsilon = std::get<1>(result);
                 * @endcode
                 */
                virtual std::tuple<torch::Tensor, double> get_first(void) override {
                    //----------------------------
                    return get_first_internal(m_batch);
                    //----------------------------
                }// end torch::Tensor get_first(void)
                //--------------------------------------------------------------
                /**
                 * @brief Get the first internal data batch.
                 *
                 * This function retrieves the first internal data batch from the class. It returns a tuple
                 * containing a Torch tensor and a double value. The Torch tensor represents the concatenated
                 * data batch, while the double value represents the epsilon calculated for the batch.
                 *
                 * @param batch The size of the batch to retrieve.
                 *
                 * @return A tuple containing the Torch tensor and epsilon value.
                 *
                 * @throw std::out_of_range If the end of the data iterator is reached before retrieving the batch.
                 *
                 * Example usage:
                 * @code{.cpp}
                 * RLEnvironmentLoader loader(data, costFunction, 0.9, 0.02, 500.0, batchSize);
                 * auto result = loader.get_first(10);
                 * torch::Tensor batch = std::get<0>(result);
                 * double epsilon = std::get<1>(result);
                 * @endcode
                 */
                std::tuple<torch::Tensor, double> get_first(const size_t& batch) {
                    //----------------------------
                    return get_first_internal(batch);
                    //----------------------------
                }// end torch::Tensor get_first(void)
                //--------------------------------------------------------------
                /**
                 * @brief Get the first internal batch of data.
                 *
                 * This function retrieves the first internal batch of data from the class. It returns a Torch tensor
                 * representing the batch of data. The epsilon value calculated during the retrieval is assigned to the
                 * output parameter `epsilon`.
                 *
                 * @param epsilon [out] The epsilon value calculated during the retrieval.
                 *
                 * @return The Torch tensor representing the first internal batch of data.
                 *
                 * @throw std::out_of_range If the end of the data iterator is reached before retrieving the batch.
                 *
                 * Example usage:
                 * @code{.cpp}
                 * size_t batch_size = 10;
                 * RLEnvironmentLoader loader(data, costFunction, 0.9, 0.02, 500.0, batchSize)
                 * double epsilon;
                 * torch::Tensor batch = loader.get_first(epsilon);
                 * @endcode
                 */
                virtual torch::Tensor get_first(OUT double& epsilon) override {
                    //----------------------------
                    return get_first_internal(epsilon, m_batch);
                    //----------------------------
                }// end torch::Tensor get_first(void)
                //--------------------------------------------------------------
                /**
                 * @brief Get the first internal batch of data.
                 *
                 * This function retrieves the first internal batch of data from the class. It returns a Torch tensor
                 * representing the batch of data. The epsilon value calculated during the retrieval is assigned to the
                 * output parameter `epsilon`.
                 *
                 * @param epsilon [out] The epsilon value calculated during the retrieval.
                 * @param batch The size of the batch to retrieve.
                 *
                 * @return The Torch tensor representing the first internal batch of data.
                 *
                 * @throw std::out_of_range If the end of the data iterator is reached before retrieving the batch.
                 *
                 * Example usage:
                 * @code{.cpp}
                 * RLEnvironmentLoader loader(data, costFunction, 0.9, 0.02, 500.0, batchSize)
                 * double epsilon;
                 * size_t batch_size = 10;
                 * torch::Tensor batch = loader.get_first(epsilon, batch_size);
                 * @endcode
                 */
                torch::Tensor get_first(OUT double& epsilon, const size_t& batch) {
                    //----------------------------
                    return get_first_internal(epsilon, batch);
                    //----------------------------
                }// end torch::Tensor get_first(void)
                //--------------------------------------------------------------
            protected:
                //--------------------------------------------------------------
                using RLEnvironmentShuffleAtomic<T, COST_OUTPUT, Args...>::internal_step;  // Bring base class functions into scope
                //--------------------------
                using RLEnvironmentShuffleAtomic<T, COST_OUTPUT, Args...>::get_first_internal;  // Bring base class functions into scope
                //--------------------------------------------------------------
                std::tuple<torch::Tensor, COST_OUTPUT, double, bool> internal_step(const size_t& batch, const Args&... args) {
                    //--------------------------------------------------------------
                    std::lock_guard<std::mutex> date_lock(m_mutex);
                    //--------------------------
                    auto _data_iter = this->get_atomic_iterator();
                    //--------------------------------------------------------------
                    if (_data_iter == m_data.end() or std::next(_data_iter, batch) == m_data.end()){
                        //--------------------------
                        throw std::out_of_range("End Of The Data Iterator");
                        //--------------------------
                    }// end if (_data_iter == m_data.end() or std::next(_data_iter, batch) == m_data.end())
                    //--------------------------------------------------------------
                    if (_data_iter == m_data.begin() and std::next(_data_iter, batch) != m_data.end()-1){
                        //--------------------------
                        torch::Tensor _data;
                        double epsilon;
                        //--------------------------
                        std::tie(_data, epsilon) = get_first_internal(batch);
                        //--------------------------
                        return {_data, torch::tensor(0), epsilon, false};
                        //--------------------------
                    }// end if (_data_iter == m_data.begin())
                    //--------------------------------------------------------------
                    std::vector<torch::Tensor> _data;
                    _data.reserve(batch);
                    //--------------------------
                    auto _data_end = std::next(_data_iter, batch);
                    //--------------------------
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::discrete_distribution<> random_distribution(m_distribution.begin(), m_distribution.end());
                    //--------------------------------------------------------------
                    if(_data_end == m_data.end()-1){
                        //--------------------------
                        for(; _data_iter != _data_end; ++_data_iter) {
                            //--------------------------
                            auto _random_position = random_distribution(gen);
                            //--------------------------
                            _data.push_back(*std::next(m_data.begin(), _random_position));
                            //--------------------------
                            m_distribution.at(_random_position) = 0;
                            //--------------------------
                        }// end for(; _data_iter != _data_end; ++_data_iter)
                        //--------------------------
                        this->set_atomic_iterator(_data_iter);
                        //--------------------------
                        return {torch::cat(_data, 0), m_CostFunction(args...), this->RLEnvironmentAtomic<T, COST_OUTPUT, Args...>::calculate_epsilon(_data_iter), true};
                        //--------------------------
                    }// if(_data_iter == m_data.end())
                    //--------------------------------------------------------------
                    for(; _data_iter != _data_end; ++_data_iter) {
                        //--------------------------
                        auto _random_position = random_distribution(gen);
                        //--------------------------
                        _data.push_back(*std::next(m_data.begin(), _random_position));
                        //--------------------------
                        m_distribution.at(_random_position) = 0;
                        //--------------------------
                    }// end for(; _data_iter != _data_end; ++_data_iter)
                    //--------------------------
                    this->set_atomic_iterator(_data_iter);
                    //--------------------------
                    return {torch::cat(_data, 0), m_CostFunction(args...), this->RLEnvironmentAtomic<T, COST_OUTPUT, Args...>::calculate_epsilon(_data_iter), false};
                    //--------------------------
                }// end std::tuple<torch::Tensor, COST_OUTPUT, double, bool> internal_step(const size_t& batch, Args... args)
                //--------------------------------------------------------------
                std::tuple<torch::Tensor, COST_OUTPUT> internal_step(OUT double& epsilon, OUT bool& done, const size_t& batch, const Args&... args) {
                    //--------------------------------------------------------------
                    std::lock_guard<std::mutex> date_lock(m_mutex);
                    //--------------------------
                    auto _data_iter = this->get_atomic_iterator();
                    //--------------------------------------------------------------
                    if (_data_iter == m_data.end() or std::next(_data_iter, batch) == m_data.end()){
                        //--------------------------
                        throw std::out_of_range("End Of The Data Iterator");
                        //--------------------------
                    }// end if (_data_iter == m_data.end() or std::next(_data_iter, batch) == m_data.end())
                    //--------------------------------------------------------------
                    if (_data_iter == m_data.begin() and std::next(_data_iter, batch) != m_data.end()-1){
                        //--------------------------
                        done = false;
                        //--------------------------
                        torch::Tensor _data = get_first_internal(epsilon, batch);
                        //--------------------------
                        return {_data, torch::tensor(0)};
                        //--------------------------
                    }// end if (_data_iter == m_data.begin() and std::next(_data_iter, batch) != m_data.end()-1)
                    //--------------------------------------------------------------
                    std::vector<torch::Tensor> _data;
                    _data.reserve(batch);
                    //--------------------------
                    auto _data_end = std::next(_data_iter, batch);
                     //--------------------------
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::discrete_distribution<> random_distribution(m_distribution.begin(), m_distribution.end());
                    //--------------------------------------------------------------
                    if(_data_end == m_data.end()-1){
                        //--------------------------
                        for(; _data_iter != _data_end; ++_data_iter) {
                            //--------------------------
                            auto _random_position = random_distribution(gen);
                            //--------------------------
                            _data.push_back(*std::next(m_data.begin(), _random_position));
                            //--------------------------
                            m_distribution.at(_random_position) = 0;
                            //--------------------------
                        }// end for(; _data_iter != _data_end; ++_data_iter)
                        //--------------------------
                        this->set_atomic_iterator(_data_iter);
                        epsilon = this->RLEnvironmentAtomic<T, COST_OUTPUT, Args...>::calculate_epsilon(_data_iter);
                        done = true;
                        //--------------------------
                        return {torch::cat(_data, 0), m_CostFunction(args...)};
                        //--------------------------
                    }// if(_data_iter == m_data.end()-1)
                    //--------------------------------------------------------------
                    for(; _data_iter != _data_end; ++_data_iter) {
                        //--------------------------
                        auto _random_position = random_distribution(gen);
                        //--------------------------
                        _data.push_back(*std::next(m_data.begin(), _random_position));
                        //--------------------------
                        m_distribution.at(_random_position) = 0;
                        //--------------------------
                    }// end for(; m_data_iter != _data_end; ++m_data_iter)
                    //--------------------------
                    this->set_atomic_iterator(_data_iter);
                    epsilon = this->RLEnvironmentAtomic<T, COST_OUTPUT, Args...>::calculate_epsilon(_data_iter);
                    done = false;
                    //--------------------------
                    return {torch::cat(_data, 0), m_CostFunction(args...)};
                    //--------------------------
                }// end std::tuple<torch::Tensor, COST_OUTPUT> internal_step(double& epsilon, bool& done, const size_t& batch, Args... args)
                //--------------------------------------------------------------
                std::tuple<torch::Tensor, double> get_first_internal(const size_t& batch) {
                    //--------------------------
                    std::lock_guard<std::mutex> date_lock(m_mutex);
                    //--------------------------
                    auto _data_iter = this->get_atomic_iterator();
                    //--------------------------
                    if (_data_iter == m_data.end() or std::next(_data_iter, batch) == m_data.end()){
                        //--------------------------
                        throw std::out_of_range("End Of The Data Iterator");
                        //--------------------------
                    }// end if (_data_iter == m_data.end() or std::next(_data_iter, batch) == m_data.end())
                    //--------------------------
                    if (_data_iter == m_data.begin() and std::next(_data_iter, batch) != m_data.end()-1){
                        //--------------------------
                        std::vector<torch::Tensor> _data;
                        _data.reserve(batch);
                        //--------------------------
                        std::random_device rd;
                        std::mt19937 gen(rd());
                        std::discrete_distribution<> random_distribution(m_distribution.begin(), m_distribution.end());
                        //--------------------------
                        auto epsilon = this->RLEnvironmentAtomic<T, COST_OUTPUT, Args...>::calculate_epsilon(_data_iter);
                        //--------------------------
                        auto _data_end = std::next(_data_iter, batch);
                        //--------------------------
                        for(; _data_iter != _data_end; ++_data_iter){
                            //--------------------------
                            auto _random_position = random_distribution(gen);
                            //--------------------------
                            _data.push_back(*std::next(m_data.begin(), _random_position));
                            //--------------------------
                            m_distribution.at(_random_position) = 0;
                            //--------------------------
                        }// end for(; _data_iter != _data_end; ++_data_iter)
                        //--------------------------
                        this->set_atomic_iterator(--_data_iter);
                        //--------------------------
                        return {torch::cat(_data, 0), epsilon};
                        //--------------------------
                    }// end if (_data_iter == m_data.begin() and std::next(_data_iter, batch) != m_data.end()-1)
                    //--------------------------
                    return {torch::tensor(0), 0};
                    //--------------------------
                }// end torch::Tensor get_first_internal(void)
                //--------------------------------------------------------------
                torch::Tensor get_first_internal(OUT double& epsilon, const size_t& batch) {
                    //--------------------------
                    std::lock_guard<std::mutex> date_lock(m_mutex);
                    //--------------------------
                    auto _data_iter = this->get_atomic_iterator();
                    //--------------------------
                    if (_data_iter == m_data.end() or std::next(_data_iter, batch) == m_data.end()){
                        //--------------------------
                        throw std::out_of_range("End Of The Data Iterator");
                        //--------------------------
                    }// end if (_data_iter == m_data.end() or std::next(_data_iter, batch) == m_data.end())
                    //--------------------------
                    if (_data_iter == m_data.begin() and std::next(_data_iter, batch) != m_data.end()-1){
                        //--------------------------
                        std::vector<torch::Tensor> _data;
                        _data.reserve(batch);
                        //--------------------------
                        std::random_device rd;
                        std::mt19937 gen(rd());
                        std::discrete_distribution<> random_distribution(m_distribution.begin(), m_distribution.end());
                        //--------------------------
                        epsilon = this->RLEnvironmentAtomic<T, COST_OUTPUT, Args...>::calculate_epsilon(_data_iter);
                        //--------------------------
                        auto _data_end = std::next(_data_iter, batch);
                        //--------------------------
                        for(; _data_iter != _data_end; ++_data_iter){
                            //--------------------------
                            auto _random_position = random_distribution(gen);
                            //--------------------------
                            _data.push_back(*std::next(m_data.begin(), _random_position));
                            //--------------------------
                            m_distribution.at(_random_position) = 0;
                            //--------------------------
                        }// end for(size_t i = 0; i < batch; ++i)
                        //--------------------------
                        this->set_atomic_iterator(--_data_iter);
                        //--------------------------
                        return torch::cat(_data, 0);
                        //--------------------------
                    }// end (_data_iter == m_data.begin() and std::next(_data_iter, batch) != m_data.end()-1)
                    //--------------------------
                    epsilon = 0.;
                    //--------------------------
                    return torch::tensor(0);
                    //--------------------------
                }// end torch::Tensor get_first_internal(OUT double& epsilon, const size_t& batch)
                //--------------------------------------------------------------
            private:
                //--------------------------------------------------------------
                std::vector<T>& m_data;
                //--------------------------
                std::atomic<typename std::vector<T>::iterator> m_data_iter;
                //--------------------------
                std::function<COST_OUTPUT(const Args&...)>& m_CostFunction;
                //--------------------------
                std::vector<uint8_t>& m_distribution;
                //--------------------------
                size_t m_batch;
                //--------------------------
                std::mutex m_mutex;
            //--------------------------------------------------------------
        };// end class RLEnvironmentLoader
        //--------------------------------------------------------------
    }// end namespace Environment
    //--------------------------------------------------------------
}// end namespace RL
//--------------------------------------------------------------