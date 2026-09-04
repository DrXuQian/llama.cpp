# quantize-budget: pick a quantization type per tensor under a VRAM budget.
#
# llama-quant-probe measures, for every weight tensor and every candidate type, the imatrix-weighted quantization
# error using ggml's own quantizers on a sample of rows, and writes it as JSON. quantize_budget.py (same directory)
# turns that into an allocation under a byte budget and applies it through llama-quantize's --tensor-type-file.
set(TARGET llama-quant-probe)
add_executable(${TARGET} probe.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
