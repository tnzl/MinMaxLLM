#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <models/inference_engine.h>
#include <models/qwen3model.h>
#include <stdexcept>

namespace py = pybind11;

// Wrapper class to ensure the engine stays alive while numpy array exists
class InferenceEngineWrapper
{
public:
    InferenceEngineWrapper(const std::string &model_name)
        : engine_(model_name) {}

    void load_weights(const std::string &safetensor_path, bool use_mmap = false)
    {
        engine_.load_weights(safetensor_path, use_mmap);
    }

    void reset_cache()
    {
        engine_.reset_cache();
    }

    void process_prompt_token(int token_id)
    {
        engine_.process_prompt_token(token_id);
    }

    // Get logits for zero-copy numpy array creation
    std::pair<const float*, size_t> get_logits(int token_id)
    {
        const auto &logits = engine_.predict_next_token(token_id);
        return {logits.data(), logits.size()};
    }

    std::size_t tokens_processed() const { return engine_.tokens_processed(); }
    const std::string &model_name() const { return engine_.model_name(); }

private:
    InferenceEngine engine_;
};

PYBIND11_MODULE(inference_engine, m)
{
    m.doc() = "Python bindings for InferenceEngine - Generic model inference interface";

    // Bind Qwen3Config (kept for advanced users who might need it)
    py::class_<Qwen3Config>(m, "Qwen3Config")
        .def(py::init<>())
        .def_readwrite("hidden_size", &Qwen3Config::hidden_size)
        .def_readwrite("intermediate_size", &Qwen3Config::intermediate_size)
        .def_readwrite("max_position_embeddings", &Qwen3Config::max_position_embeddings)
        .def_readwrite("max_window_layers", &Qwen3Config::max_window_layers)
        .def_readwrite("num_attention_heads", &Qwen3Config::num_attention_heads)
        .def_readwrite("num_hidden_layers", &Qwen3Config::num_hidden_layers)
        .def_readwrite("num_key_value_heads", &Qwen3Config::num_key_value_heads)
        .def_readwrite("rms_norm_eps", &Qwen3Config::rms_norm_eps)
        .def_readwrite("rope_theta", &Qwen3Config::rope_theta)
        .def_readwrite("vocab_size", &Qwen3Config::vocab_size)
        .def_readwrite("bos_token_id", &Qwen3Config::bos_token_id)
        .def_readwrite("eos_token_id", &Qwen3Config::eos_token_id);

    // Bind InferenceEngine (using wrapper for safe numpy array handling)
    py::class_<InferenceEngineWrapper>(m, "InferenceEngine")
        .def(py::init<const std::string &>(), py::arg("model_name"),
             "Create an InferenceEngine for the specified model type (e.g., 'qwen3')")
        .def("load_weights", &InferenceEngineWrapper::load_weights,
             py::arg("safetensor_path"), py::arg("use_mmap") = false,
             "Load model weights from safetensors file")
        .def("reset_cache", &InferenceEngineWrapper::reset_cache,
             "Reset the KV cache")
        .def("process_prompt_token", &InferenceEngineWrapper::process_prompt_token,
             py::arg("token_id"),
             "Process a single prompt token (without computing logits)")
        .def("predict_next_token", 
             [](InferenceEngineWrapper &self, int token_id) {
                 // Get logits data and size
                 auto [data, size] = self.get_logits(token_id);
                 
                 // Create zero-copy numpy array with capsule to keep parent alive
                 // Capture the Python object to ensure the engine stays alive
                 py::object py_self = py::cast(&self, py::return_value_policy::reference);
                 auto capsule = py::capsule(new py::object(py_self), [](void *p) {
                     delete reinterpret_cast<py::object *>(p);
                 });
                 
                 return py::array_t<float>(
                     {size},
                     {sizeof(float)},
                     const_cast<float *>(data),
                     capsule);
             },
             py::arg("token_id"),
             py::keep_alive<0, 1>(), // Keep self alive while array exists
             "Predict next token logits for the given token ID. Returns numpy array of logits.")
        .def_property_readonly("tokens_processed", &InferenceEngineWrapper::tokens_processed,
                               "Get the number of tokens processed so far")
        .def_property_readonly("model_name", &InferenceEngineWrapper::model_name,
                               "Get the model name");
}

