#include "llama.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using llama_model_ptr   = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using llama_context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
using llama_sampler_ptr = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [-c n_ctx] [-o out.json] [prompt]\n", argv[0]);
    printf("\n");
}

// escape a UTF-8 string into a JSON string body (no surrounding quotes)
static std::string json_escape(const std::string & in) {
    std::string out;
    out.reserve(in.size() + 16);
    for (unsigned char c : in) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(c));
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // path to the model gguf file
    std::string model_path;

    // prompt to generate text from
    std::string prompt = "Hello my name is";

    // path to the output json file; empty -> print to stdout
    std::string out_json_path;

    // number of layers to offload to the GPU
    int ngl = 99;

    // number of tokens to predict
    int n_predict = 32;

    // context size, 0 -> derive from prompt + n_predict
    int n_ctx = 0;

    // parse command line arguments

    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    try {
                        n_predict = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    try {
                        ngl = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-c") == 0) {
                if (i + 1 < argc) {
                    try {
                        n_ctx = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-o") == 0) {
                if (i + 1 < argc) {
                    out_json_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                // prompt starts here
                break;
            }
        }
        if (model_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }
        if (i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
        }
    }

    // the full model output text (prompt echo + generated tokens), the only thing written to json
    std::string result_text;

    // load dynamic backends

    ggml_backend_load_all();

    // initialize the model

    llama_model_params model_params = llama_model_default_params();

    model_params.n_gpu_layers = ngl;

    llama_model_ptr model(llama_model_load_from_file(model_path.c_str(), model_params), llama_model_free);

    if (!model) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model.get());

    // tokenize the prompt

    // find the number of tokens in the prompt
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) <
        0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    // initialize the context

    llama_context_params ctx_params = llama_context_default_params();

    // n_ctx is the context size
    ctx_params.n_ctx = n_ctx > 0 ? n_ctx : (n_prompt + n_predict - 1);

    // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch = n_prompt;

    // enable performance counters
    ctx_params.no_perf = false;

    llama_context_ptr ctx(llama_init_from_model(model.get(), ctx_params), llama_free);

    if (!ctx) {
        fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
        return 1;
    }

    // initialize the sampler

    auto sparams = llama_sampler_chain_default_params();

    sparams.no_perf = false;

    llama_sampler_ptr smpl(llama_sampler_chain_init(sparams), llama_sampler_free);

    if (!smpl) {
        fprintf(stderr, "%s: error: failed to create the sampler\n", __func__);
        return 1;
    }

    llama_sampler_chain_add(smpl.get(), llama_sampler_init_greedy());

    // collect the prompt token-by-token

    for (auto id : prompt_tokens) {
        char buf[128];
        int  n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        result_text.append(buf, n);
    }

    // prepare a batch for the prompt

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    if (llama_model_has_encoder(model.get())) {
        if (llama_encode(ctx.get(), batch)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return 1;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model.get());
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }

        batch = llama_batch_get_one(&decoder_start_token_id, 1);
    }

    // main loop

    const auto t_main_start = ggml_time_us();

    int n_decode = 0;

    llama_token new_token_id;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict;) {
        // evaluate the current batch with the transformer model
        const int decode_result = llama_decode(ctx.get(), batch);
        if (decode_result != 0) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, decode_result);
            return 1;
        }

        n_pos += batch.n_tokens;

        // sample the next token
        {
            new_token_id = llama_sampler_sample(smpl.get(), ctx.get(), -1);

            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            char buf[128];
            int  n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            result_text.append(buf, n);

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);

            n_decode += 1;
        }
    }

    // write the collected output: json file when -o is set, else stdout
    if (!out_json_path.empty()) {
        std::ofstream f(out_json_path, std::ios::binary | std::ios::trunc);
        if (!f) {
            fprintf(stderr, "%s: error: failed to open output json: %s\n", __func__, out_json_path.c_str());
            return 1;
        }
        f << "{\"output\":\"" << json_escape(result_text) << "\"}\n";
    } else {
        printf("%s\n", result_text.c_str());
    }

    const auto t_main_end = ggml_time_us();

    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n", __func__, n_decode,
            (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl.get());
    llama_perf_context_print(ctx.get());
    fprintf(stderr, "\n");

    return 0;
}
