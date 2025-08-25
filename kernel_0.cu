#include "persistent_kernel.cuh"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
using json = nlohmann::json;
using namespace mirage::runtime;
size_t get_event_id(int my_gpu_id, size_t event_pos, bool nvshmem_event) {
  size_t event_id = ((static_cast<size_t>(my_gpu_id) << 32) | event_pos);
  if (nvshmem_event) {
    event_id = event_id | EVENT_NVSHMEM_TAG;
  }
  return event_id;
}

void construct_task_graph(int num_gpus,
                          int my_gpu_id,
                          std::vector<TaskDesc> &all_tasks,
                          std::vector<EventDesc> &all_events,
                          std::vector<TaskId> &first_tasks,
                          std::map<std::string, void*> const &all_tensors) {
  std::filesystem::path file_path(__FILE__);
  std::ifstream json_file(file_path.parent_path().string()+"/task_graph.json");
  nlohmann::json json_task_graph;
  json_file >> json_task_graph;
  for (json const &task : json_task_graph["all_tasks"]) {
    TaskDesc task_desc(static_cast<TaskType>(task.at("task_type")),
                task.at("variant_id"));
    if (task.at("trigger_event").is_number_integer()) {
      task_desc.trigger_event = task.at("trigger_event").get<unsigned long long int>();
    }
    else {
      assert(false);
    }
    if (task.at("dependent_event").is_number_integer()) {
      task_desc.dependent_event = task.at("dependent_event").get<unsigned long long int>();
    }
    else {
      assert(false);
    }
    task_desc.num_inputs = 0;
    for (json const &tensor : task["inputs"]) {
      TensorDesc input;
      std::string name = tensor.at("base_ptr").get<std::string>();
      assert(all_tensors.find(name) != all_tensors.end());
      off_t offset = tensor.at("offset").get<off_t>();
      input.base_ptr = static_cast<char*>(all_tensors.at(name))+offset;
      assert(tensor.at("dims").size() == tensor.at("strides").size());
      input.num_dims = tensor.at("dims").size();
      input.data_type = tensor.at("data_type").get<int>();
      for (int i = 0; i < input.num_dims; i++) {
        input.dim[i] = tensor["dims"][i].get<int>();
        input.stride[i] = tensor["strides"][i].get<int>();
      }
      task_desc.inputs[task_desc.num_inputs++] = input;
    }
    task_desc.num_outputs = 0;
    for (json const &tensor : task["outputs"]) {
      TensorDesc output;
      std::string name = tensor.at("base_ptr").get<std::string>();
      assert(all_tensors.find(name) != all_tensors.end());
      off_t offset = tensor.at("offset").get<off_t>();
      output.base_ptr = static_cast<char*>(all_tensors.at(name))+offset;
      assert(tensor.at("dims").size() == tensor.at("strides").size());
      output.num_dims = tensor.at("dims").size();
      output.data_type = tensor.at("data_type").get<int>();
      for (int i = 0; i < output.num_dims; i++) {
        output.dim[i] = tensor["dims"][i];
        output.stride[i] = tensor["strides"][i];
      }
      task_desc.outputs[task_desc.num_outputs++] = output;
    }
    all_tasks.push_back(task_desc);
  }
  for (json const &e : json_task_graph["all_events"]) {
    EventType event_type = static_cast<EventType>(e.at("event_type").get<int>());
    int num_triggers = e.at("num_triggers").get<int>();
    int first_task_id = e.at("first_task_id").get<int>();
    int last_task_id = e.at("last_task_id").get<int>();
    all_events.push_back(EventDesc(event_type, num_triggers, first_task_id, last_task_id));
  }
  for (json const &t : json_task_graph["first_tasks"]) {
    first_tasks.push_back(t.get<int>());
  }
}

static void _init_persistent_kernel(std::vector<TaskDesc> &all_tasks,
                                    std::vector<EventDesc> &all_events,
                                  std::vector<TaskId> &first_tasks,
                                  int num_gpus,
                                  int my_gpu_id) {
  assert(num_gpus = 1);
  std::map<std::string, void*> all_tensors;
  char *input_token = (char*)(0x75036a520c00);
  all_tensors["input_token"] = input_token;
  char *cos_position_embedding = (char*)(0x7502af020000);
  all_tensors["cos_position_embedding"] = cos_position_embedding;
  char *sin_position_embedding = (char*)(0x7502af820000);
  all_tensors["sin_position_embedding"] = sin_position_embedding;
  void *embed_out;
  cudaMalloc(&embed_out, 8192);
  all_tensors["embed_out"] = embed_out;
  void *attn_in;
  cudaMalloc(&attn_in, 12288);
  all_tensors["attn_in"] = attn_in;
  void *attn_out;
  cudaMalloc(&attn_out, 8192);
  all_tensors["attn_out"] = attn_out;
  void *attn_proj_out;
  cudaMalloc(&attn_proj_out, 8192);
  all_tensors["attn_proj_out"] = attn_proj_out;
  void *all_reduce_buf;
  cudaMalloc(&all_reduce_buf, 8192);
  all_tensors["all_reduce_buf"] = all_reduce_buf;
  void *attn_allreduce_out;
  cudaMalloc(&attn_allreduce_out, 8192);
  all_tensors["attn_allreduce_out"] = attn_allreduce_out;
  void *mlp_mid;
  cudaMalloc(&mlp_mid, 49152);
  all_tensors["mlp_mid"] = mlp_mid;
  void *mlp_out;
  cudaMalloc(&mlp_out, 8192);
  all_tensors["mlp_out"] = mlp_out;
  void *mlp_final;
  cudaMalloc(&mlp_final, 8192);
  all_tensors["mlp_final"] = mlp_final;
  void *argmax_in;
  cudaMalloc(&argmax_in, 307200);
  all_tensors["argmax_in"] = argmax_in;
  void *argmax_part_value;
  cudaMalloc(&argmax_part_value, 192);
  all_tensors["argmax_part_value"] = argmax_part_value;
  void *argmax_part_index;
  cudaMalloc(&argmax_part_index, 768);
  all_tensors["argmax_part_index"] = argmax_part_index;
  void *argmax_out;
  cudaMalloc(&argmax_out, 8);
  all_tensors["argmax_out"] = argmax_out;
  char *embed_tokens = (char*)(0x750320000000);
  all_tensors["embed_tokens"] = embed_tokens;
  char *layer_0_input_layernorm = (char*)(0x75036a404e00);
  all_tensors["layer_0_input_layernorm"] = layer_0_input_layernorm;
  char *layer_0_q_proj = (char*)(0x74ff84800000);
  all_tensors["layer_0_q_proj"] = layer_0_q_proj;
  char *layer_0_k_proj = (char*)(0x74ff82000000);
  all_tensors["layer_0_k_proj"] = layer_0_k_proj;
  char *layer_0_v_proj = (char*)(0x74ff86800000);
  all_tensors["layer_0_v_proj"] = layer_0_v_proj;
  char *layer_0_q_norm = (char*)(0x75036a408e00);
  all_tensors["layer_0_q_norm"] = layer_0_q_norm;
  char *layer_0_k_norm = (char*)(0x75036a400200);
  all_tensors["layer_0_k_norm"] = layer_0_k_norm;
  char *layer_0_k_cache = (char*)(0x750382000000);
  all_tensors["layer_0_k_cache"] = layer_0_k_cache;
  char *layer_0_v_cache = (char*)(0x75036e000000);
  all_tensors["layer_0_v_cache"] = layer_0_v_cache;
  char *layer_0_o_proj = (char*)(0x74ff82800000);
  all_tensors["layer_0_o_proj"] = layer_0_o_proj;
  char *layer_0_post_attn_layernorm = (char*)(0x75036a406e00);
  all_tensors["layer_0_post_attn_layernorm"] = layer_0_post_attn_layernorm;
  char *layer_0_gate_proj = (char*)(0x74ff76000000);
  all_tensors["layer_0_gate_proj"] = layer_0_gate_proj;
  char *layer_0_up_proj = (char*)(0x74ff7c000000);
  all_tensors["layer_0_up_proj"] = layer_0_up_proj;
  char *layer_0_down_proj = (char*)(0x74ff70000000);
  all_tensors["layer_0_down_proj"] = layer_0_down_proj;
  char *layer_1_input_layernorm = (char*)(0x75036a409000);
  all_tensors["layer_1_input_layernorm"] = layer_1_input_layernorm;
  char *layer_1_q_proj = (char*)(0x74ff9b800000);
  all_tensors["layer_1_q_proj"] = layer_1_q_proj;
  char *layer_1_k_proj = (char*)(0x74ff99000000);
  all_tensors["layer_1_k_proj"] = layer_1_k_proj;
  char *layer_1_v_proj = (char*)(0x74ff9d800000);
  all_tensors["layer_1_v_proj"] = layer_1_v_proj;
  char *layer_1_q_norm = (char*)(0x75036a40d200);
  all_tensors["layer_1_q_norm"] = layer_1_q_norm;
  char *layer_1_k_norm = (char*)(0x75036a40d000);
  all_tensors["layer_1_k_norm"] = layer_1_k_norm;
  char *layer_1_k_cache = (char*)(0x750382800000);
  all_tensors["layer_1_k_cache"] = layer_1_k_cache;
  char *layer_1_v_cache = (char*)(0x75036e800000);
  all_tensors["layer_1_v_cache"] = layer_1_v_cache;
  char *layer_1_o_proj = (char*)(0x74ff99800000);
  all_tensors["layer_1_o_proj"] = layer_1_o_proj;
  char *layer_1_post_attn_layernorm = (char*)(0x75036a40b000);
  all_tensors["layer_1_post_attn_layernorm"] = layer_1_post_attn_layernorm;
  char *layer_1_gate_proj = (char*)(0x74ff8d000000);
  all_tensors["layer_1_gate_proj"] = layer_1_gate_proj;
  char *layer_1_up_proj = (char*)(0x74ff93000000);
  all_tensors["layer_1_up_proj"] = layer_1_up_proj;
  char *layer_1_down_proj = (char*)(0x74ff87000000);
  all_tensors["layer_1_down_proj"] = layer_1_down_proj;
  char *layer_2_input_layernorm = (char*)(0x75036a40d400);
  all_tensors["layer_2_input_layernorm"] = layer_2_input_layernorm;
  char *layer_2_q_proj = (char*)(0x74ffb2800000);
  all_tensors["layer_2_q_proj"] = layer_2_q_proj;
  char *layer_2_k_proj = (char*)(0x74ffb0000000);
  all_tensors["layer_2_k_proj"] = layer_2_k_proj;
  char *layer_2_v_proj = (char*)(0x74ffb4800000);
  all_tensors["layer_2_v_proj"] = layer_2_v_proj;
  char *layer_2_q_norm = (char*)(0x75036a411600);
  all_tensors["layer_2_q_norm"] = layer_2_q_norm;
  char *layer_2_k_norm = (char*)(0x75036a411400);
  all_tensors["layer_2_k_norm"] = layer_2_k_norm;
  char *layer_2_k_cache = (char*)(0x750383000000);
  all_tensors["layer_2_k_cache"] = layer_2_k_cache;
  char *layer_2_v_cache = (char*)(0x75036f000000);
  all_tensors["layer_2_v_cache"] = layer_2_v_cache;
  char *layer_2_o_proj = (char*)(0x74ffb0800000);
  all_tensors["layer_2_o_proj"] = layer_2_o_proj;
  char *layer_2_post_attn_layernorm = (char*)(0x75036a40f400);
  all_tensors["layer_2_post_attn_layernorm"] = layer_2_post_attn_layernorm;
  char *layer_2_gate_proj = (char*)(0x74ffa4000000);
  all_tensors["layer_2_gate_proj"] = layer_2_gate_proj;
  char *layer_2_up_proj = (char*)(0x74ffaa000000);
  all_tensors["layer_2_up_proj"] = layer_2_up_proj;
  char *layer_2_down_proj = (char*)(0x74ff9e000000);
  all_tensors["layer_2_down_proj"] = layer_2_down_proj;
  char *layer_3_input_layernorm = (char*)(0x75036a411800);
  all_tensors["layer_3_input_layernorm"] = layer_3_input_layernorm;
  char *layer_3_q_proj = (char*)(0x74ffc9800000);
  all_tensors["layer_3_q_proj"] = layer_3_q_proj;
  char *layer_3_k_proj = (char*)(0x74ffc7000000);
  all_tensors["layer_3_k_proj"] = layer_3_k_proj;
  char *layer_3_v_proj = (char*)(0x74ffcb800000);
  all_tensors["layer_3_v_proj"] = layer_3_v_proj;
  char *layer_3_q_norm = (char*)(0x75036a415a00);
  all_tensors["layer_3_q_norm"] = layer_3_q_norm;
  char *layer_3_k_norm = (char*)(0x75036a415800);
  all_tensors["layer_3_k_norm"] = layer_3_k_norm;
  char *layer_3_k_cache = (char*)(0x750383800000);
  all_tensors["layer_3_k_cache"] = layer_3_k_cache;
  char *layer_3_v_cache = (char*)(0x75036f800000);
  all_tensors["layer_3_v_cache"] = layer_3_v_cache;
  char *layer_3_o_proj = (char*)(0x74ffc7800000);
  all_tensors["layer_3_o_proj"] = layer_3_o_proj;
  char *layer_3_post_attn_layernorm = (char*)(0x75036a413800);
  all_tensors["layer_3_post_attn_layernorm"] = layer_3_post_attn_layernorm;
  char *layer_3_gate_proj = (char*)(0x74ffbb000000);
  all_tensors["layer_3_gate_proj"] = layer_3_gate_proj;
  char *layer_3_up_proj = (char*)(0x74ffc1000000);
  all_tensors["layer_3_up_proj"] = layer_3_up_proj;
  char *layer_3_down_proj = (char*)(0x74ffb5000000);
  all_tensors["layer_3_down_proj"] = layer_3_down_proj;
  char *layer_4_input_layernorm = (char*)(0x75036a415c00);
  all_tensors["layer_4_input_layernorm"] = layer_4_input_layernorm;
  char *layer_4_q_proj = (char*)(0x74ffe0800000);
  all_tensors["layer_4_q_proj"] = layer_4_q_proj;
  char *layer_4_k_proj = (char*)(0x74ffde000000);
  all_tensors["layer_4_k_proj"] = layer_4_k_proj;
  char *layer_4_v_proj = (char*)(0x74ffe2800000);
  all_tensors["layer_4_v_proj"] = layer_4_v_proj;
  char *layer_4_q_norm = (char*)(0x75036a419e00);
  all_tensors["layer_4_q_norm"] = layer_4_q_norm;
  char *layer_4_k_norm = (char*)(0x75036a419c00);
  all_tensors["layer_4_k_norm"] = layer_4_k_norm;
  char *layer_4_k_cache = (char*)(0x750384000000);
  all_tensors["layer_4_k_cache"] = layer_4_k_cache;
  char *layer_4_v_cache = (char*)(0x750370000000);
  all_tensors["layer_4_v_cache"] = layer_4_v_cache;
  char *layer_4_o_proj = (char*)(0x74ffde800000);
  all_tensors["layer_4_o_proj"] = layer_4_o_proj;
  char *layer_4_post_attn_layernorm = (char*)(0x75036a417c00);
  all_tensors["layer_4_post_attn_layernorm"] = layer_4_post_attn_layernorm;
  char *layer_4_gate_proj = (char*)(0x74ffd2000000);
  all_tensors["layer_4_gate_proj"] = layer_4_gate_proj;
  char *layer_4_up_proj = (char*)(0x74ffd8000000);
  all_tensors["layer_4_up_proj"] = layer_4_up_proj;
  char *layer_4_down_proj = (char*)(0x74ffcc000000);
  all_tensors["layer_4_down_proj"] = layer_4_down_proj;
  char *layer_5_input_layernorm = (char*)(0x75036a41a000);
  all_tensors["layer_5_input_layernorm"] = layer_5_input_layernorm;
  char *layer_5_q_proj = (char*)(0x74fff7800000);
  all_tensors["layer_5_q_proj"] = layer_5_q_proj;
  char *layer_5_k_proj = (char*)(0x74fff5000000);
  all_tensors["layer_5_k_proj"] = layer_5_k_proj;
  char *layer_5_v_proj = (char*)(0x74fff9800000);
  all_tensors["layer_5_v_proj"] = layer_5_v_proj;
  char *layer_5_q_norm = (char*)(0x75036a41e200);
  all_tensors["layer_5_q_norm"] = layer_5_q_norm;
  char *layer_5_k_norm = (char*)(0x75036a41e000);
  all_tensors["layer_5_k_norm"] = layer_5_k_norm;
  char *layer_5_k_cache = (char*)(0x750384800000);
  all_tensors["layer_5_k_cache"] = layer_5_k_cache;
  char *layer_5_v_cache = (char*)(0x750370800000);
  all_tensors["layer_5_v_cache"] = layer_5_v_cache;
  char *layer_5_o_proj = (char*)(0x74fff5800000);
  all_tensors["layer_5_o_proj"] = layer_5_o_proj;
  char *layer_5_post_attn_layernorm = (char*)(0x75036a41c000);
  all_tensors["layer_5_post_attn_layernorm"] = layer_5_post_attn_layernorm;
  char *layer_5_gate_proj = (char*)(0x74ffe9000000);
  all_tensors["layer_5_gate_proj"] = layer_5_gate_proj;
  char *layer_5_up_proj = (char*)(0x74ffef000000);
  all_tensors["layer_5_up_proj"] = layer_5_up_proj;
  char *layer_5_down_proj = (char*)(0x74ffe3000000);
  all_tensors["layer_5_down_proj"] = layer_5_down_proj;
  char *layer_6_input_layernorm = (char*)(0x75036a41e400);
  all_tensors["layer_6_input_layernorm"] = layer_6_input_layernorm;
  char *layer_6_q_proj = (char*)(0x75000e800000);
  all_tensors["layer_6_q_proj"] = layer_6_q_proj;
  char *layer_6_k_proj = (char*)(0x75000c000000);
  all_tensors["layer_6_k_proj"] = layer_6_k_proj;
  char *layer_6_v_proj = (char*)(0x750010800000);
  all_tensors["layer_6_v_proj"] = layer_6_v_proj;
  char *layer_6_q_norm = (char*)(0x75036a422600);
  all_tensors["layer_6_q_norm"] = layer_6_q_norm;
  char *layer_6_k_norm = (char*)(0x75036a422400);
  all_tensors["layer_6_k_norm"] = layer_6_k_norm;
  char *layer_6_k_cache = (char*)(0x750385000000);
  all_tensors["layer_6_k_cache"] = layer_6_k_cache;
  char *layer_6_v_cache = (char*)(0x750371000000);
  all_tensors["layer_6_v_cache"] = layer_6_v_cache;
  char *layer_6_o_proj = (char*)(0x75000c800000);
  all_tensors["layer_6_o_proj"] = layer_6_o_proj;
  char *layer_6_post_attn_layernorm = (char*)(0x75036a420400);
  all_tensors["layer_6_post_attn_layernorm"] = layer_6_post_attn_layernorm;
  char *layer_6_gate_proj = (char*)(0x750000000000);
  all_tensors["layer_6_gate_proj"] = layer_6_gate_proj;
  char *layer_6_up_proj = (char*)(0x750006000000);
  all_tensors["layer_6_up_proj"] = layer_6_up_proj;
  char *layer_6_down_proj = (char*)(0x74fffa000000);
  all_tensors["layer_6_down_proj"] = layer_6_down_proj;
  char *layer_7_input_layernorm = (char*)(0x75036a440800);
  all_tensors["layer_7_input_layernorm"] = layer_7_input_layernorm;
  char *layer_7_q_proj = (char*)(0x750011800000);
  all_tensors["layer_7_q_proj"] = layer_7_q_proj;
  char *layer_7_k_proj = (char*)(0x750011000000);
  all_tensors["layer_7_k_proj"] = layer_7_k_proj;
  char *layer_7_v_proj = (char*)(0x750013800000);
  all_tensors["layer_7_v_proj"] = layer_7_v_proj;
  char *layer_7_q_norm = (char*)(0x75036a444a00);
  all_tensors["layer_7_q_norm"] = layer_7_q_norm;
  char *layer_7_k_norm = (char*)(0x75036a444800);
  all_tensors["layer_7_k_norm"] = layer_7_k_norm;
  char *layer_7_k_cache = (char*)(0x750385800000);
  all_tensors["layer_7_k_cache"] = layer_7_k_cache;
  char *layer_7_v_cache = (char*)(0x750371800000);
  all_tensors["layer_7_v_cache"] = layer_7_v_cache;
  char *layer_7_o_proj = (char*)(0x7500d2000000);
  all_tensors["layer_7_o_proj"] = layer_7_o_proj;
  char *layer_7_post_attn_layernorm = (char*)(0x75036a442800);
  all_tensors["layer_7_post_attn_layernorm"] = layer_7_post_attn_layernorm;
  char *layer_7_gate_proj = (char*)(0x7500c6000000);
  all_tensors["layer_7_gate_proj"] = layer_7_gate_proj;
  char *layer_7_up_proj = (char*)(0x7500cc000000);
  all_tensors["layer_7_up_proj"] = layer_7_up_proj;
  char *layer_7_down_proj = (char*)(0x7500c0000000);
  all_tensors["layer_7_down_proj"] = layer_7_down_proj;
  char *layer_8_input_layernorm = (char*)(0x75036a444c00);
  all_tensors["layer_8_input_layernorm"] = layer_8_input_layernorm;
  char *layer_8_q_proj = (char*)(0x7500e8800000);
  all_tensors["layer_8_q_proj"] = layer_8_q_proj;
  char *layer_8_k_proj = (char*)(0x7500e6000000);
  all_tensors["layer_8_k_proj"] = layer_8_k_proj;
  char *layer_8_v_proj = (char*)(0x7500ea800000);
  all_tensors["layer_8_v_proj"] = layer_8_v_proj;
  char *layer_8_q_norm = (char*)(0x75036a448e00);
  all_tensors["layer_8_q_norm"] = layer_8_q_norm;
  char *layer_8_k_norm = (char*)(0x75036a448c00);
  all_tensors["layer_8_k_norm"] = layer_8_k_norm;
  char *layer_8_k_cache = (char*)(0x750386000000);
  all_tensors["layer_8_k_cache"] = layer_8_k_cache;
  char *layer_8_v_cache = (char*)(0x750372000000);
  all_tensors["layer_8_v_cache"] = layer_8_v_cache;
  char *layer_8_o_proj = (char*)(0x7500e6800000);
  all_tensors["layer_8_o_proj"] = layer_8_o_proj;
  char *layer_8_post_attn_layernorm = (char*)(0x75036a446c00);
  all_tensors["layer_8_post_attn_layernorm"] = layer_8_post_attn_layernorm;
  char *layer_8_gate_proj = (char*)(0x7500da000000);
  all_tensors["layer_8_gate_proj"] = layer_8_gate_proj;
  char *layer_8_up_proj = (char*)(0x7500e0000000);
  all_tensors["layer_8_up_proj"] = layer_8_up_proj;
  char *layer_8_down_proj = (char*)(0x7500d4000000);
  all_tensors["layer_8_down_proj"] = layer_8_down_proj;
  char *layer_9_input_layernorm = (char*)(0x75036a449000);
  all_tensors["layer_9_input_layernorm"] = layer_9_input_layernorm;
  char *layer_9_q_proj = (char*)(0x7500ff800000);
  all_tensors["layer_9_q_proj"] = layer_9_q_proj;
  char *layer_9_k_proj = (char*)(0x7500fd000000);
  all_tensors["layer_9_k_proj"] = layer_9_k_proj;
  char *layer_9_v_proj = (char*)(0x750101800000);
  all_tensors["layer_9_v_proj"] = layer_9_v_proj;
  char *layer_9_q_norm = (char*)(0x75036a44d200);
  all_tensors["layer_9_q_norm"] = layer_9_q_norm;
  char *layer_9_k_norm = (char*)(0x75036a44d000);
  all_tensors["layer_9_k_norm"] = layer_9_k_norm;
  char *layer_9_k_cache = (char*)(0x750386800000);
  all_tensors["layer_9_k_cache"] = layer_9_k_cache;
  char *layer_9_v_cache = (char*)(0x750372800000);
  all_tensors["layer_9_v_cache"] = layer_9_v_cache;
  char *layer_9_o_proj = (char*)(0x7500fd800000);
  all_tensors["layer_9_o_proj"] = layer_9_o_proj;
  char *layer_9_post_attn_layernorm = (char*)(0x75036a44b000);
  all_tensors["layer_9_post_attn_layernorm"] = layer_9_post_attn_layernorm;
  char *layer_9_gate_proj = (char*)(0x7500f1000000);
  all_tensors["layer_9_gate_proj"] = layer_9_gate_proj;
  char *layer_9_up_proj = (char*)(0x7500f7000000);
  all_tensors["layer_9_up_proj"] = layer_9_up_proj;
  char *layer_9_down_proj = (char*)(0x7500eb000000);
  all_tensors["layer_9_down_proj"] = layer_9_down_proj;
  char *layer_10_input_layernorm = (char*)(0x75036a422800);
  all_tensors["layer_10_input_layernorm"] = layer_10_input_layernorm;
  char *layer_10_q_proj = (char*)(0x750028800000);
  all_tensors["layer_10_q_proj"] = layer_10_q_proj;
  char *layer_10_k_proj = (char*)(0x750026000000);
  all_tensors["layer_10_k_proj"] = layer_10_k_proj;
  char *layer_10_v_proj = (char*)(0x75002a800000);
  all_tensors["layer_10_v_proj"] = layer_10_v_proj;
  char *layer_10_q_norm = (char*)(0x75036a426a00);
  all_tensors["layer_10_q_norm"] = layer_10_q_norm;
  char *layer_10_k_norm = (char*)(0x75036a426800);
  all_tensors["layer_10_k_norm"] = layer_10_k_norm;
  char *layer_10_k_cache = (char*)(0x750387000000);
  all_tensors["layer_10_k_cache"] = layer_10_k_cache;
  char *layer_10_v_cache = (char*)(0x750373000000);
  all_tensors["layer_10_v_cache"] = layer_10_v_cache;
  char *layer_10_o_proj = (char*)(0x750026800000);
  all_tensors["layer_10_o_proj"] = layer_10_o_proj;
  char *layer_10_post_attn_layernorm = (char*)(0x75036a424800);
  all_tensors["layer_10_post_attn_layernorm"] = layer_10_post_attn_layernorm;
  char *layer_10_gate_proj = (char*)(0x75001a000000);
  all_tensors["layer_10_gate_proj"] = layer_10_gate_proj;
  char *layer_10_up_proj = (char*)(0x750020000000);
  all_tensors["layer_10_up_proj"] = layer_10_up_proj;
  char *layer_10_down_proj = (char*)(0x750014000000);
  all_tensors["layer_10_down_proj"] = layer_10_down_proj;
  char *layer_11_input_layernorm = (char*)(0x75036a426c00);
  all_tensors["layer_11_input_layernorm"] = layer_11_input_layernorm;
  char *layer_11_q_proj = (char*)(0x75003f800000);
  all_tensors["layer_11_q_proj"] = layer_11_q_proj;
  char *layer_11_k_proj = (char*)(0x75003d000000);
  all_tensors["layer_11_k_proj"] = layer_11_k_proj;
  char *layer_11_v_proj = (char*)(0x750041800000);
  all_tensors["layer_11_v_proj"] = layer_11_v_proj;
  char *layer_11_q_norm = (char*)(0x75036a42ae00);
  all_tensors["layer_11_q_norm"] = layer_11_q_norm;
  char *layer_11_k_norm = (char*)(0x75036a42ac00);
  all_tensors["layer_11_k_norm"] = layer_11_k_norm;
  char *layer_11_k_cache = (char*)(0x750387800000);
  all_tensors["layer_11_k_cache"] = layer_11_k_cache;
  char *layer_11_v_cache = (char*)(0x750373800000);
  all_tensors["layer_11_v_cache"] = layer_11_v_cache;
  char *layer_11_o_proj = (char*)(0x75003d800000);
  all_tensors["layer_11_o_proj"] = layer_11_o_proj;
  char *layer_11_post_attn_layernorm = (char*)(0x75036a428c00);
  all_tensors["layer_11_post_attn_layernorm"] = layer_11_post_attn_layernorm;
  char *layer_11_gate_proj = (char*)(0x750031000000);
  all_tensors["layer_11_gate_proj"] = layer_11_gate_proj;
  char *layer_11_up_proj = (char*)(0x750037000000);
  all_tensors["layer_11_up_proj"] = layer_11_up_proj;
  char *layer_11_down_proj = (char*)(0x75002b000000);
  all_tensors["layer_11_down_proj"] = layer_11_down_proj;
  char *layer_12_input_layernorm = (char*)(0x75036a42b000);
  all_tensors["layer_12_input_layernorm"] = layer_12_input_layernorm;
  char *layer_12_q_proj = (char*)(0x750056800000);
  all_tensors["layer_12_q_proj"] = layer_12_q_proj;
  char *layer_12_k_proj = (char*)(0x750054000000);
  all_tensors["layer_12_k_proj"] = layer_12_k_proj;
  char *layer_12_v_proj = (char*)(0x750058800000);
  all_tensors["layer_12_v_proj"] = layer_12_v_proj;
  char *layer_12_q_norm = (char*)(0x75036a42f200);
  all_tensors["layer_12_q_norm"] = layer_12_q_norm;
  char *layer_12_k_norm = (char*)(0x75036a42f000);
  all_tensors["layer_12_k_norm"] = layer_12_k_norm;
  char *layer_12_k_cache = (char*)(0x750388000000);
  all_tensors["layer_12_k_cache"] = layer_12_k_cache;
  char *layer_12_v_cache = (char*)(0x750374000000);
  all_tensors["layer_12_v_cache"] = layer_12_v_cache;
  char *layer_12_o_proj = (char*)(0x750054800000);
  all_tensors["layer_12_o_proj"] = layer_12_o_proj;
  char *layer_12_post_attn_layernorm = (char*)(0x75036a42d000);
  all_tensors["layer_12_post_attn_layernorm"] = layer_12_post_attn_layernorm;
  char *layer_12_gate_proj = (char*)(0x750048000000);
  all_tensors["layer_12_gate_proj"] = layer_12_gate_proj;
  char *layer_12_up_proj = (char*)(0x75004e000000);
  all_tensors["layer_12_up_proj"] = layer_12_up_proj;
  char *layer_12_down_proj = (char*)(0x750042000000);
  all_tensors["layer_12_down_proj"] = layer_12_down_proj;
  char *layer_13_input_layernorm = (char*)(0x75036a42f400);
  all_tensors["layer_13_input_layernorm"] = layer_13_input_layernorm;
  char *layer_13_q_proj = (char*)(0x75006d800000);
  all_tensors["layer_13_q_proj"] = layer_13_q_proj;
  char *layer_13_k_proj = (char*)(0x75006b000000);
  all_tensors["layer_13_k_proj"] = layer_13_k_proj;
  char *layer_13_v_proj = (char*)(0x75006f800000);
  all_tensors["layer_13_v_proj"] = layer_13_v_proj;
  char *layer_13_q_norm = (char*)(0x75036a433600);
  all_tensors["layer_13_q_norm"] = layer_13_q_norm;
  char *layer_13_k_norm = (char*)(0x75036a433400);
  all_tensors["layer_13_k_norm"] = layer_13_k_norm;
  char *layer_13_k_cache = (char*)(0x750388800000);
  all_tensors["layer_13_k_cache"] = layer_13_k_cache;
  char *layer_13_v_cache = (char*)(0x750374800000);
  all_tensors["layer_13_v_cache"] = layer_13_v_cache;
  char *layer_13_o_proj = (char*)(0x75006b800000);
  all_tensors["layer_13_o_proj"] = layer_13_o_proj;
  char *layer_13_post_attn_layernorm = (char*)(0x75036a431400);
  all_tensors["layer_13_post_attn_layernorm"] = layer_13_post_attn_layernorm;
  char *layer_13_gate_proj = (char*)(0x75005f000000);
  all_tensors["layer_13_gate_proj"] = layer_13_gate_proj;
  char *layer_13_up_proj = (char*)(0x750065000000);
  all_tensors["layer_13_up_proj"] = layer_13_up_proj;
  char *layer_13_down_proj = (char*)(0x750059000000);
  all_tensors["layer_13_down_proj"] = layer_13_down_proj;
  char *layer_14_input_layernorm = (char*)(0x75036a433800);
  all_tensors["layer_14_input_layernorm"] = layer_14_input_layernorm;
  char *layer_14_q_proj = (char*)(0x750084800000);
  all_tensors["layer_14_q_proj"] = layer_14_q_proj;
  char *layer_14_k_proj = (char*)(0x750082000000);
  all_tensors["layer_14_k_proj"] = layer_14_k_proj;
  char *layer_14_v_proj = (char*)(0x750086800000);
  all_tensors["layer_14_v_proj"] = layer_14_v_proj;
  char *layer_14_q_norm = (char*)(0x75036a437a00);
  all_tensors["layer_14_q_norm"] = layer_14_q_norm;
  char *layer_14_k_norm = (char*)(0x75036a437800);
  all_tensors["layer_14_k_norm"] = layer_14_k_norm;
  char *layer_14_k_cache = (char*)(0x750389000000);
  all_tensors["layer_14_k_cache"] = layer_14_k_cache;
  char *layer_14_v_cache = (char*)(0x750375000000);
  all_tensors["layer_14_v_cache"] = layer_14_v_cache;
  char *layer_14_o_proj = (char*)(0x750082800000);
  all_tensors["layer_14_o_proj"] = layer_14_o_proj;
  char *layer_14_post_attn_layernorm = (char*)(0x75036a435800);
  all_tensors["layer_14_post_attn_layernorm"] = layer_14_post_attn_layernorm;
  char *layer_14_gate_proj = (char*)(0x750076000000);
  all_tensors["layer_14_gate_proj"] = layer_14_gate_proj;
  char *layer_14_up_proj = (char*)(0x75007c000000);
  all_tensors["layer_14_up_proj"] = layer_14_up_proj;
  char *layer_14_down_proj = (char*)(0x750070000000);
  all_tensors["layer_14_down_proj"] = layer_14_down_proj;
  char *layer_15_input_layernorm = (char*)(0x75036a437c00);
  all_tensors["layer_15_input_layernorm"] = layer_15_input_layernorm;
  char *layer_15_q_proj = (char*)(0x75009b800000);
  all_tensors["layer_15_q_proj"] = layer_15_q_proj;
  char *layer_15_k_proj = (char*)(0x750099000000);
  all_tensors["layer_15_k_proj"] = layer_15_k_proj;
  char *layer_15_v_proj = (char*)(0x75009d800000);
  all_tensors["layer_15_v_proj"] = layer_15_v_proj;
  char *layer_15_q_norm = (char*)(0x75036a43be00);
  all_tensors["layer_15_q_norm"] = layer_15_q_norm;
  char *layer_15_k_norm = (char*)(0x75036a43bc00);
  all_tensors["layer_15_k_norm"] = layer_15_k_norm;
  char *layer_15_k_cache = (char*)(0x750389800000);
  all_tensors["layer_15_k_cache"] = layer_15_k_cache;
  char *layer_15_v_cache = (char*)(0x750375800000);
  all_tensors["layer_15_v_cache"] = layer_15_v_cache;
  char *layer_15_o_proj = (char*)(0x750099800000);
  all_tensors["layer_15_o_proj"] = layer_15_o_proj;
  char *layer_15_post_attn_layernorm = (char*)(0x75036a439c00);
  all_tensors["layer_15_post_attn_layernorm"] = layer_15_post_attn_layernorm;
  char *layer_15_gate_proj = (char*)(0x75008d000000);
  all_tensors["layer_15_gate_proj"] = layer_15_gate_proj;
  char *layer_15_up_proj = (char*)(0x750093000000);
  all_tensors["layer_15_up_proj"] = layer_15_up_proj;
  char *layer_15_down_proj = (char*)(0x750087000000);
  all_tensors["layer_15_down_proj"] = layer_15_down_proj;
  char *layer_16_input_layernorm = (char*)(0x75036a43c000);
  all_tensors["layer_16_input_layernorm"] = layer_16_input_layernorm;
  char *layer_16_q_proj = (char*)(0x7500b2800000);
  all_tensors["layer_16_q_proj"] = layer_16_q_proj;
  char *layer_16_k_proj = (char*)(0x7500b0000000);
  all_tensors["layer_16_k_proj"] = layer_16_k_proj;
  char *layer_16_v_proj = (char*)(0x7500b4800000);
  all_tensors["layer_16_v_proj"] = layer_16_v_proj;
  char *layer_16_q_norm = (char*)(0x75036a440200);
  all_tensors["layer_16_q_norm"] = layer_16_q_norm;
  char *layer_16_k_norm = (char*)(0x75036a440000);
  all_tensors["layer_16_k_norm"] = layer_16_k_norm;
  char *layer_16_k_cache = (char*)(0x75038a000000);
  all_tensors["layer_16_k_cache"] = layer_16_k_cache;
  char *layer_16_v_cache = (char*)(0x750376000000);
  all_tensors["layer_16_v_cache"] = layer_16_v_cache;
  char *layer_16_o_proj = (char*)(0x7500b0800000);
  all_tensors["layer_16_o_proj"] = layer_16_o_proj;
  char *layer_16_post_attn_layernorm = (char*)(0x75036a43e000);
  all_tensors["layer_16_post_attn_layernorm"] = layer_16_post_attn_layernorm;
  char *layer_16_gate_proj = (char*)(0x7500a4000000);
  all_tensors["layer_16_gate_proj"] = layer_16_gate_proj;
  char *layer_16_up_proj = (char*)(0x7500aa000000);
  all_tensors["layer_16_up_proj"] = layer_16_up_proj;
  char *layer_16_down_proj = (char*)(0x75009e000000);
  all_tensors["layer_16_down_proj"] = layer_16_down_proj;
  char *layer_17_input_layernorm = (char*)(0x75036a44d400);
  all_tensors["layer_17_input_layernorm"] = layer_17_input_layernorm;
  char *layer_17_q_proj = (char*)(0x7500bd800000);
  all_tensors["layer_17_q_proj"] = layer_17_q_proj;
  char *layer_17_k_proj = (char*)(0x7500bb000000);
  all_tensors["layer_17_k_proj"] = layer_17_k_proj;
  char *layer_17_v_proj = (char*)(0x7500bf800000);
  all_tensors["layer_17_v_proj"] = layer_17_v_proj;
  char *layer_17_q_norm = (char*)(0x75036a440600);
  all_tensors["layer_17_q_norm"] = layer_17_q_norm;
  char *layer_17_k_norm = (char*)(0x75036a440400);
  all_tensors["layer_17_k_norm"] = layer_17_k_norm;
  char *layer_17_k_cache = (char*)(0x75038a800000);
  all_tensors["layer_17_k_cache"] = layer_17_k_cache;
  char *layer_17_v_cache = (char*)(0x750376800000);
  all_tensors["layer_17_v_cache"] = layer_17_v_cache;
  char *layer_17_o_proj = (char*)(0x7500bb800000);
  all_tensors["layer_17_o_proj"] = layer_17_o_proj;
  char *layer_17_post_attn_layernorm = (char*)(0x75036a44f400);
  all_tensors["layer_17_post_attn_layernorm"] = layer_17_post_attn_layernorm;
  char *layer_17_gate_proj = (char*)(0x7500b5000000);
  all_tensors["layer_17_gate_proj"] = layer_17_gate_proj;
  char *layer_17_up_proj = (char*)(0x750108000000);
  all_tensors["layer_17_up_proj"] = layer_17_up_proj;
  char *layer_17_down_proj = (char*)(0x750102000000);
  all_tensors["layer_17_down_proj"] = layer_17_down_proj;
  char *layer_18_input_layernorm = (char*)(0x75036a451400);
  all_tensors["layer_18_input_layernorm"] = layer_18_input_layernorm;
  char *layer_18_q_proj = (char*)(0x750122800000);
  all_tensors["layer_18_q_proj"] = layer_18_q_proj;
  char *layer_18_k_proj = (char*)(0x750120000000);
  all_tensors["layer_18_k_proj"] = layer_18_k_proj;
  char *layer_18_v_proj = (char*)(0x750124800000);
  all_tensors["layer_18_v_proj"] = layer_18_v_proj;
  char *layer_18_q_norm = (char*)(0x75036a455600);
  all_tensors["layer_18_q_norm"] = layer_18_q_norm;
  char *layer_18_k_norm = (char*)(0x75036a455400);
  all_tensors["layer_18_k_norm"] = layer_18_k_norm;
  char *layer_18_k_cache = (char*)(0x75038b000000);
  all_tensors["layer_18_k_cache"] = layer_18_k_cache;
  char *layer_18_v_cache = (char*)(0x750377000000);
  all_tensors["layer_18_v_cache"] = layer_18_v_cache;
  char *layer_18_o_proj = (char*)(0x750120800000);
  all_tensors["layer_18_o_proj"] = layer_18_o_proj;
  char *layer_18_post_attn_layernorm = (char*)(0x75036a453400);
  all_tensors["layer_18_post_attn_layernorm"] = layer_18_post_attn_layernorm;
  char *layer_18_gate_proj = (char*)(0x750114000000);
  all_tensors["layer_18_gate_proj"] = layer_18_gate_proj;
  char *layer_18_up_proj = (char*)(0x75011a000000);
  all_tensors["layer_18_up_proj"] = layer_18_up_proj;
  char *layer_18_down_proj = (char*)(0x75010e000000);
  all_tensors["layer_18_down_proj"] = layer_18_down_proj;
  char *layer_19_input_layernorm = (char*)(0x75036a455800);
  all_tensors["layer_19_input_layernorm"] = layer_19_input_layernorm;
  char *layer_19_q_proj = (char*)(0x750139800000);
  all_tensors["layer_19_q_proj"] = layer_19_q_proj;
  char *layer_19_k_proj = (char*)(0x750137000000);
  all_tensors["layer_19_k_proj"] = layer_19_k_proj;
  char *layer_19_v_proj = (char*)(0x75013b800000);
  all_tensors["layer_19_v_proj"] = layer_19_v_proj;
  char *layer_19_q_norm = (char*)(0x75036a459a00);
  all_tensors["layer_19_q_norm"] = layer_19_q_norm;
  char *layer_19_k_norm = (char*)(0x75036a459800);
  all_tensors["layer_19_k_norm"] = layer_19_k_norm;
  char *layer_19_k_cache = (char*)(0x75038b800000);
  all_tensors["layer_19_k_cache"] = layer_19_k_cache;
  char *layer_19_v_cache = (char*)(0x750377800000);
  all_tensors["layer_19_v_cache"] = layer_19_v_cache;
  char *layer_19_o_proj = (char*)(0x750137800000);
  all_tensors["layer_19_o_proj"] = layer_19_o_proj;
  char *layer_19_post_attn_layernorm = (char*)(0x75036a457800);
  all_tensors["layer_19_post_attn_layernorm"] = layer_19_post_attn_layernorm;
  char *layer_19_gate_proj = (char*)(0x75012b000000);
  all_tensors["layer_19_gate_proj"] = layer_19_gate_proj;
  char *layer_19_up_proj = (char*)(0x750131000000);
  all_tensors["layer_19_up_proj"] = layer_19_up_proj;
  char *layer_19_down_proj = (char*)(0x750125000000);
  all_tensors["layer_19_down_proj"] = layer_19_down_proj;
  char *layer_20_input_layernorm = (char*)(0x75036a459c00);
  all_tensors["layer_20_input_layernorm"] = layer_20_input_layernorm;
  char *layer_20_q_proj = (char*)(0x750150800000);
  all_tensors["layer_20_q_proj"] = layer_20_q_proj;
  char *layer_20_k_proj = (char*)(0x75014e000000);
  all_tensors["layer_20_k_proj"] = layer_20_k_proj;
  char *layer_20_v_proj = (char*)(0x750152800000);
  all_tensors["layer_20_v_proj"] = layer_20_v_proj;
  char *layer_20_q_norm = (char*)(0x75036a45de00);
  all_tensors["layer_20_q_norm"] = layer_20_q_norm;
  char *layer_20_k_norm = (char*)(0x75036a45dc00);
  all_tensors["layer_20_k_norm"] = layer_20_k_norm;
  char *layer_20_k_cache = (char*)(0x75038c000000);
  all_tensors["layer_20_k_cache"] = layer_20_k_cache;
  char *layer_20_v_cache = (char*)(0x750378000000);
  all_tensors["layer_20_v_cache"] = layer_20_v_cache;
  char *layer_20_o_proj = (char*)(0x75014e800000);
  all_tensors["layer_20_o_proj"] = layer_20_o_proj;
  char *layer_20_post_attn_layernorm = (char*)(0x75036a45bc00);
  all_tensors["layer_20_post_attn_layernorm"] = layer_20_post_attn_layernorm;
  char *layer_20_gate_proj = (char*)(0x750142000000);
  all_tensors["layer_20_gate_proj"] = layer_20_gate_proj;
  char *layer_20_up_proj = (char*)(0x750148000000);
  all_tensors["layer_20_up_proj"] = layer_20_up_proj;
  char *layer_20_down_proj = (char*)(0x75013c000000);
  all_tensors["layer_20_down_proj"] = layer_20_down_proj;
  char *layer_21_input_layernorm = (char*)(0x75036a45e000);
  all_tensors["layer_21_input_layernorm"] = layer_21_input_layernorm;
  char *layer_21_q_proj = (char*)(0x750167800000);
  all_tensors["layer_21_q_proj"] = layer_21_q_proj;
  char *layer_21_k_proj = (char*)(0x750165000000);
  all_tensors["layer_21_k_proj"] = layer_21_k_proj;
  char *layer_21_v_proj = (char*)(0x750169800000);
  all_tensors["layer_21_v_proj"] = layer_21_v_proj;
  char *layer_21_q_norm = (char*)(0x75036a462200);
  all_tensors["layer_21_q_norm"] = layer_21_q_norm;
  char *layer_21_k_norm = (char*)(0x75036a462000);
  all_tensors["layer_21_k_norm"] = layer_21_k_norm;
  char *layer_21_k_cache = (char*)(0x75038c800000);
  all_tensors["layer_21_k_cache"] = layer_21_k_cache;
  char *layer_21_v_cache = (char*)(0x750378800000);
  all_tensors["layer_21_v_cache"] = layer_21_v_cache;
  char *layer_21_o_proj = (char*)(0x750165800000);
  all_tensors["layer_21_o_proj"] = layer_21_o_proj;
  char *layer_21_post_attn_layernorm = (char*)(0x75036a460000);
  all_tensors["layer_21_post_attn_layernorm"] = layer_21_post_attn_layernorm;
  char *layer_21_gate_proj = (char*)(0x750159000000);
  all_tensors["layer_21_gate_proj"] = layer_21_gate_proj;
  char *layer_21_up_proj = (char*)(0x75015f000000);
  all_tensors["layer_21_up_proj"] = layer_21_up_proj;
  char *layer_21_down_proj = (char*)(0x750153000000);
  all_tensors["layer_21_down_proj"] = layer_21_down_proj;
  char *layer_22_input_layernorm = (char*)(0x75036a462400);
  all_tensors["layer_22_input_layernorm"] = layer_22_input_layernorm;
  char *layer_22_q_proj = (char*)(0x75017e800000);
  all_tensors["layer_22_q_proj"] = layer_22_q_proj;
  char *layer_22_k_proj = (char*)(0x75017c000000);
  all_tensors["layer_22_k_proj"] = layer_22_k_proj;
  char *layer_22_v_proj = (char*)(0x750180800000);
  all_tensors["layer_22_v_proj"] = layer_22_v_proj;
  char *layer_22_q_norm = (char*)(0x75036a466600);
  all_tensors["layer_22_q_norm"] = layer_22_q_norm;
  char *layer_22_k_norm = (char*)(0x75036a466400);
  all_tensors["layer_22_k_norm"] = layer_22_k_norm;
  char *layer_22_k_cache = (char*)(0x75038d000000);
  all_tensors["layer_22_k_cache"] = layer_22_k_cache;
  char *layer_22_v_cache = (char*)(0x750379000000);
  all_tensors["layer_22_v_cache"] = layer_22_v_cache;
  char *layer_22_o_proj = (char*)(0x75017c800000);
  all_tensors["layer_22_o_proj"] = layer_22_o_proj;
  char *layer_22_post_attn_layernorm = (char*)(0x75036a464400);
  all_tensors["layer_22_post_attn_layernorm"] = layer_22_post_attn_layernorm;
  char *layer_22_gate_proj = (char*)(0x750170000000);
  all_tensors["layer_22_gate_proj"] = layer_22_gate_proj;
  char *layer_22_up_proj = (char*)(0x750176000000);
  all_tensors["layer_22_up_proj"] = layer_22_up_proj;
  char *layer_22_down_proj = (char*)(0x75016a000000);
  all_tensors["layer_22_down_proj"] = layer_22_down_proj;
  char *layer_23_input_layernorm = (char*)(0x75036a466800);
  all_tensors["layer_23_input_layernorm"] = layer_23_input_layernorm;
  char *layer_23_q_proj = (char*)(0x750195800000);
  all_tensors["layer_23_q_proj"] = layer_23_q_proj;
  char *layer_23_k_proj = (char*)(0x750193000000);
  all_tensors["layer_23_k_proj"] = layer_23_k_proj;
  char *layer_23_v_proj = (char*)(0x750197800000);
  all_tensors["layer_23_v_proj"] = layer_23_v_proj;
  char *layer_23_q_norm = (char*)(0x75036a46aa00);
  all_tensors["layer_23_q_norm"] = layer_23_q_norm;
  char *layer_23_k_norm = (char*)(0x75036a46a800);
  all_tensors["layer_23_k_norm"] = layer_23_k_norm;
  char *layer_23_k_cache = (char*)(0x75038d800000);
  all_tensors["layer_23_k_cache"] = layer_23_k_cache;
  char *layer_23_v_cache = (char*)(0x750379800000);
  all_tensors["layer_23_v_cache"] = layer_23_v_cache;
  char *layer_23_o_proj = (char*)(0x750193800000);
  all_tensors["layer_23_o_proj"] = layer_23_o_proj;
  char *layer_23_post_attn_layernorm = (char*)(0x75036a468800);
  all_tensors["layer_23_post_attn_layernorm"] = layer_23_post_attn_layernorm;
  char *layer_23_gate_proj = (char*)(0x750187000000);
  all_tensors["layer_23_gate_proj"] = layer_23_gate_proj;
  char *layer_23_up_proj = (char*)(0x75018d000000);
  all_tensors["layer_23_up_proj"] = layer_23_up_proj;
  char *layer_23_down_proj = (char*)(0x750181000000);
  all_tensors["layer_23_down_proj"] = layer_23_down_proj;
  char *layer_24_input_layernorm = (char*)(0x75036a46ac00);
  all_tensors["layer_24_input_layernorm"] = layer_24_input_layernorm;
  char *layer_24_q_proj = (char*)(0x7501ac800000);
  all_tensors["layer_24_q_proj"] = layer_24_q_proj;
  char *layer_24_k_proj = (char*)(0x7501aa000000);
  all_tensors["layer_24_k_proj"] = layer_24_k_proj;
  char *layer_24_v_proj = (char*)(0x7501ae800000);
  all_tensors["layer_24_v_proj"] = layer_24_v_proj;
  char *layer_24_q_norm = (char*)(0x75036a46ee00);
  all_tensors["layer_24_q_norm"] = layer_24_q_norm;
  char *layer_24_k_norm = (char*)(0x75036a46ec00);
  all_tensors["layer_24_k_norm"] = layer_24_k_norm;
  char *layer_24_k_cache = (char*)(0x75038e000000);
  all_tensors["layer_24_k_cache"] = layer_24_k_cache;
  char *layer_24_v_cache = (char*)(0x75037a000000);
  all_tensors["layer_24_v_cache"] = layer_24_v_cache;
  char *layer_24_o_proj = (char*)(0x7501aa800000);
  all_tensors["layer_24_o_proj"] = layer_24_o_proj;
  char *layer_24_post_attn_layernorm = (char*)(0x75036a46cc00);
  all_tensors["layer_24_post_attn_layernorm"] = layer_24_post_attn_layernorm;
  char *layer_24_gate_proj = (char*)(0x75019e000000);
  all_tensors["layer_24_gate_proj"] = layer_24_gate_proj;
  char *layer_24_up_proj = (char*)(0x7501a4000000);
  all_tensors["layer_24_up_proj"] = layer_24_up_proj;
  char *layer_24_down_proj = (char*)(0x750198000000);
  all_tensors["layer_24_down_proj"] = layer_24_down_proj;
  char *layer_25_input_layernorm = (char*)(0x75036a46f000);
  all_tensors["layer_25_input_layernorm"] = layer_25_input_layernorm;
  char *layer_25_q_proj = (char*)(0x7501c3800000);
  all_tensors["layer_25_q_proj"] = layer_25_q_proj;
  char *layer_25_k_proj = (char*)(0x7501c1000000);
  all_tensors["layer_25_k_proj"] = layer_25_k_proj;
  char *layer_25_v_proj = (char*)(0x7501c5800000);
  all_tensors["layer_25_v_proj"] = layer_25_v_proj;
  char *layer_25_q_norm = (char*)(0x75036a473200);
  all_tensors["layer_25_q_norm"] = layer_25_q_norm;
  char *layer_25_k_norm = (char*)(0x75036a473000);
  all_tensors["layer_25_k_norm"] = layer_25_k_norm;
  char *layer_25_k_cache = (char*)(0x75038e800000);
  all_tensors["layer_25_k_cache"] = layer_25_k_cache;
  char *layer_25_v_cache = (char*)(0x75037a800000);
  all_tensors["layer_25_v_cache"] = layer_25_v_cache;
  char *layer_25_o_proj = (char*)(0x7501c1800000);
  all_tensors["layer_25_o_proj"] = layer_25_o_proj;
  char *layer_25_post_attn_layernorm = (char*)(0x75036a471000);
  all_tensors["layer_25_post_attn_layernorm"] = layer_25_post_attn_layernorm;
  char *layer_25_gate_proj = (char*)(0x7501b5000000);
  all_tensors["layer_25_gate_proj"] = layer_25_gate_proj;
  char *layer_25_up_proj = (char*)(0x7501bb000000);
  all_tensors["layer_25_up_proj"] = layer_25_up_proj;
  char *layer_25_down_proj = (char*)(0x7501af000000);
  all_tensors["layer_25_down_proj"] = layer_25_down_proj;
  char *layer_26_input_layernorm = (char*)(0x75036a473400);
  all_tensors["layer_26_input_layernorm"] = layer_26_input_layernorm;
  char *layer_26_q_proj = (char*)(0x7501da800000);
  all_tensors["layer_26_q_proj"] = layer_26_q_proj;
  char *layer_26_k_proj = (char*)(0x7501d8000000);
  all_tensors["layer_26_k_proj"] = layer_26_k_proj;
  char *layer_26_v_proj = (char*)(0x7501dc800000);
  all_tensors["layer_26_v_proj"] = layer_26_v_proj;
  char *layer_26_q_norm = (char*)(0x75036a477600);
  all_tensors["layer_26_q_norm"] = layer_26_q_norm;
  char *layer_26_k_norm = (char*)(0x75036a477400);
  all_tensors["layer_26_k_norm"] = layer_26_k_norm;
  char *layer_26_k_cache = (char*)(0x75038f000000);
  all_tensors["layer_26_k_cache"] = layer_26_k_cache;
  char *layer_26_v_cache = (char*)(0x75037b000000);
  all_tensors["layer_26_v_cache"] = layer_26_v_cache;
  char *layer_26_o_proj = (char*)(0x7501d8800000);
  all_tensors["layer_26_o_proj"] = layer_26_o_proj;
  char *layer_26_post_attn_layernorm = (char*)(0x75036a475400);
  all_tensors["layer_26_post_attn_layernorm"] = layer_26_post_attn_layernorm;
  char *layer_26_gate_proj = (char*)(0x7501cc000000);
  all_tensors["layer_26_gate_proj"] = layer_26_gate_proj;
  char *layer_26_up_proj = (char*)(0x7501d2000000);
  all_tensors["layer_26_up_proj"] = layer_26_up_proj;
  char *layer_26_down_proj = (char*)(0x7501c6000000);
  all_tensors["layer_26_down_proj"] = layer_26_down_proj;
  char *layer_27_input_layernorm = (char*)(0x75036a477c00);
  all_tensors["layer_27_input_layernorm"] = layer_27_input_layernorm;
  char *layer_27_q_proj = (char*)(0x7501eb800000);
  all_tensors["layer_27_q_proj"] = layer_27_q_proj;
  char *layer_27_k_proj = (char*)(0x7501e9000000);
  all_tensors["layer_27_k_proj"] = layer_27_k_proj;
  char *layer_27_v_proj = (char*)(0x7501ed800000);
  all_tensors["layer_27_v_proj"] = layer_27_v_proj;
  char *layer_27_q_norm = (char*)(0x75036a477a00);
  all_tensors["layer_27_q_norm"] = layer_27_q_norm;
  char *layer_27_k_norm = (char*)(0x75036a477800);
  all_tensors["layer_27_k_norm"] = layer_27_k_norm;
  char *layer_27_k_cache = (char*)(0x75038f800000);
  all_tensors["layer_27_k_cache"] = layer_27_k_cache;
  char *layer_27_v_cache = (char*)(0x75037b800000);
  all_tensors["layer_27_v_cache"] = layer_27_v_cache;
  char *layer_27_o_proj = (char*)(0x7501e9800000);
  all_tensors["layer_27_o_proj"] = layer_27_o_proj;
  char *layer_27_post_attn_layernorm = (char*)(0x75036a479c00);
  all_tensors["layer_27_post_attn_layernorm"] = layer_27_post_attn_layernorm;
  char *layer_27_gate_proj = (char*)(0x7501dd000000);
  all_tensors["layer_27_gate_proj"] = layer_27_gate_proj;
  char *layer_27_up_proj = (char*)(0x7501e3000000);
  all_tensors["layer_27_up_proj"] = layer_27_up_proj;
  char *layer_27_down_proj = (char*)(0x7501ee000000);
  all_tensors["layer_27_down_proj"] = layer_27_down_proj;
  char *layer_28_input_layernorm = (char*)(0x75036a47bc00);
  all_tensors["layer_28_input_layernorm"] = layer_28_input_layernorm;
  char *layer_28_q_proj = (char*)(0x750208800000);
  all_tensors["layer_28_q_proj"] = layer_28_q_proj;
  char *layer_28_k_proj = (char*)(0x750206000000);
  all_tensors["layer_28_k_proj"] = layer_28_k_proj;
  char *layer_28_v_proj = (char*)(0x75020a800000);
  all_tensors["layer_28_v_proj"] = layer_28_v_proj;
  char *layer_28_q_norm = (char*)(0x75036a47fe00);
  all_tensors["layer_28_q_norm"] = layer_28_q_norm;
  char *layer_28_k_norm = (char*)(0x75036a47fc00);
  all_tensors["layer_28_k_norm"] = layer_28_k_norm;
  char *layer_28_k_cache = (char*)(0x750390000000);
  all_tensors["layer_28_k_cache"] = layer_28_k_cache;
  char *layer_28_v_cache = (char*)(0x75037c000000);
  all_tensors["layer_28_v_cache"] = layer_28_v_cache;
  char *layer_28_o_proj = (char*)(0x750206800000);
  all_tensors["layer_28_o_proj"] = layer_28_o_proj;
  char *layer_28_post_attn_layernorm = (char*)(0x75036a47dc00);
  all_tensors["layer_28_post_attn_layernorm"] = layer_28_post_attn_layernorm;
  char *layer_28_gate_proj = (char*)(0x7501fa000000);
  all_tensors["layer_28_gate_proj"] = layer_28_gate_proj;
  char *layer_28_up_proj = (char*)(0x750200000000);
  all_tensors["layer_28_up_proj"] = layer_28_up_proj;
  char *layer_28_down_proj = (char*)(0x7501f4000000);
  all_tensors["layer_28_down_proj"] = layer_28_down_proj;
  char *layer_29_input_layernorm = (char*)(0x75036a480000);
  all_tensors["layer_29_input_layernorm"] = layer_29_input_layernorm;
  char *layer_29_q_proj = (char*)(0x75021f800000);
  all_tensors["layer_29_q_proj"] = layer_29_q_proj;
  char *layer_29_k_proj = (char*)(0x75021d000000);
  all_tensors["layer_29_k_proj"] = layer_29_k_proj;
  char *layer_29_v_proj = (char*)(0x750221800000);
  all_tensors["layer_29_v_proj"] = layer_29_v_proj;
  char *layer_29_q_norm = (char*)(0x75036a484200);
  all_tensors["layer_29_q_norm"] = layer_29_q_norm;
  char *layer_29_k_norm = (char*)(0x75036a484000);
  all_tensors["layer_29_k_norm"] = layer_29_k_norm;
  char *layer_29_k_cache = (char*)(0x750390800000);
  all_tensors["layer_29_k_cache"] = layer_29_k_cache;
  char *layer_29_v_cache = (char*)(0x75037c800000);
  all_tensors["layer_29_v_cache"] = layer_29_v_cache;
  char *layer_29_o_proj = (char*)(0x75021d800000);
  all_tensors["layer_29_o_proj"] = layer_29_o_proj;
  char *layer_29_post_attn_layernorm = (char*)(0x75036a482000);
  all_tensors["layer_29_post_attn_layernorm"] = layer_29_post_attn_layernorm;
  char *layer_29_gate_proj = (char*)(0x750211000000);
  all_tensors["layer_29_gate_proj"] = layer_29_gate_proj;
  char *layer_29_up_proj = (char*)(0x750217000000);
  all_tensors["layer_29_up_proj"] = layer_29_up_proj;
  char *layer_29_down_proj = (char*)(0x75020b000000);
  all_tensors["layer_29_down_proj"] = layer_29_down_proj;
  char *layer_30_input_layernorm = (char*)(0x75036a484400);
  all_tensors["layer_30_input_layernorm"] = layer_30_input_layernorm;
  char *layer_30_q_proj = (char*)(0x750236800000);
  all_tensors["layer_30_q_proj"] = layer_30_q_proj;
  char *layer_30_k_proj = (char*)(0x750234000000);
  all_tensors["layer_30_k_proj"] = layer_30_k_proj;
  char *layer_30_v_proj = (char*)(0x750238800000);
  all_tensors["layer_30_v_proj"] = layer_30_v_proj;
  char *layer_30_q_norm = (char*)(0x75036a488600);
  all_tensors["layer_30_q_norm"] = layer_30_q_norm;
  char *layer_30_k_norm = (char*)(0x75036a488400);
  all_tensors["layer_30_k_norm"] = layer_30_k_norm;
  char *layer_30_k_cache = (char*)(0x750391000000);
  all_tensors["layer_30_k_cache"] = layer_30_k_cache;
  char *layer_30_v_cache = (char*)(0x75037d000000);
  all_tensors["layer_30_v_cache"] = layer_30_v_cache;
  char *layer_30_o_proj = (char*)(0x750234800000);
  all_tensors["layer_30_o_proj"] = layer_30_o_proj;
  char *layer_30_post_attn_layernorm = (char*)(0x75036a486400);
  all_tensors["layer_30_post_attn_layernorm"] = layer_30_post_attn_layernorm;
  char *layer_30_gate_proj = (char*)(0x750228000000);
  all_tensors["layer_30_gate_proj"] = layer_30_gate_proj;
  char *layer_30_up_proj = (char*)(0x75022e000000);
  all_tensors["layer_30_up_proj"] = layer_30_up_proj;
  char *layer_30_down_proj = (char*)(0x750222000000);
  all_tensors["layer_30_down_proj"] = layer_30_down_proj;
  char *layer_31_input_layernorm = (char*)(0x75036a488800);
  all_tensors["layer_31_input_layernorm"] = layer_31_input_layernorm;
  char *layer_31_q_proj = (char*)(0x75024d800000);
  all_tensors["layer_31_q_proj"] = layer_31_q_proj;
  char *layer_31_k_proj = (char*)(0x75024b000000);
  all_tensors["layer_31_k_proj"] = layer_31_k_proj;
  char *layer_31_v_proj = (char*)(0x75024f800000);
  all_tensors["layer_31_v_proj"] = layer_31_v_proj;
  char *layer_31_q_norm = (char*)(0x75036a48ca00);
  all_tensors["layer_31_q_norm"] = layer_31_q_norm;
  char *layer_31_k_norm = (char*)(0x75036a48c800);
  all_tensors["layer_31_k_norm"] = layer_31_k_norm;
  char *layer_31_k_cache = (char*)(0x750391800000);
  all_tensors["layer_31_k_cache"] = layer_31_k_cache;
  char *layer_31_v_cache = (char*)(0x75037d800000);
  all_tensors["layer_31_v_cache"] = layer_31_v_cache;
  char *layer_31_o_proj = (char*)(0x75024b800000);
  all_tensors["layer_31_o_proj"] = layer_31_o_proj;
  char *layer_31_post_attn_layernorm = (char*)(0x75036a48a800);
  all_tensors["layer_31_post_attn_layernorm"] = layer_31_post_attn_layernorm;
  char *layer_31_gate_proj = (char*)(0x75023f000000);
  all_tensors["layer_31_gate_proj"] = layer_31_gate_proj;
  char *layer_31_up_proj = (char*)(0x750245000000);
  all_tensors["layer_31_up_proj"] = layer_31_up_proj;
  char *layer_31_down_proj = (char*)(0x750239000000);
  all_tensors["layer_31_down_proj"] = layer_31_down_proj;
  char *layer_32_input_layernorm = (char*)(0x75036a48cc00);
  all_tensors["layer_32_input_layernorm"] = layer_32_input_layernorm;
  char *layer_32_q_proj = (char*)(0x750264800000);
  all_tensors["layer_32_q_proj"] = layer_32_q_proj;
  char *layer_32_k_proj = (char*)(0x750262000000);
  all_tensors["layer_32_k_proj"] = layer_32_k_proj;
  char *layer_32_v_proj = (char*)(0x750266800000);
  all_tensors["layer_32_v_proj"] = layer_32_v_proj;
  char *layer_32_q_norm = (char*)(0x75036a490e00);
  all_tensors["layer_32_q_norm"] = layer_32_q_norm;
  char *layer_32_k_norm = (char*)(0x75036a490c00);
  all_tensors["layer_32_k_norm"] = layer_32_k_norm;
  char *layer_32_k_cache = (char*)(0x750392000000);
  all_tensors["layer_32_k_cache"] = layer_32_k_cache;
  char *layer_32_v_cache = (char*)(0x75037e000000);
  all_tensors["layer_32_v_cache"] = layer_32_v_cache;
  char *layer_32_o_proj = (char*)(0x750262800000);
  all_tensors["layer_32_o_proj"] = layer_32_o_proj;
  char *layer_32_post_attn_layernorm = (char*)(0x75036a48ec00);
  all_tensors["layer_32_post_attn_layernorm"] = layer_32_post_attn_layernorm;
  char *layer_32_gate_proj = (char*)(0x750256000000);
  all_tensors["layer_32_gate_proj"] = layer_32_gate_proj;
  char *layer_32_up_proj = (char*)(0x75025c000000);
  all_tensors["layer_32_up_proj"] = layer_32_up_proj;
  char *layer_32_down_proj = (char*)(0x750250000000);
  all_tensors["layer_32_down_proj"] = layer_32_down_proj;
  char *layer_33_input_layernorm = (char*)(0x75036a491000);
  all_tensors["layer_33_input_layernorm"] = layer_33_input_layernorm;
  char *layer_33_q_proj = (char*)(0x75027b800000);
  all_tensors["layer_33_q_proj"] = layer_33_q_proj;
  char *layer_33_k_proj = (char*)(0x750279000000);
  all_tensors["layer_33_k_proj"] = layer_33_k_proj;
  char *layer_33_v_proj = (char*)(0x75027d800000);
  all_tensors["layer_33_v_proj"] = layer_33_v_proj;
  char *layer_33_q_norm = (char*)(0x75036a495200);
  all_tensors["layer_33_q_norm"] = layer_33_q_norm;
  char *layer_33_k_norm = (char*)(0x75036a495000);
  all_tensors["layer_33_k_norm"] = layer_33_k_norm;
  char *layer_33_k_cache = (char*)(0x750392800000);
  all_tensors["layer_33_k_cache"] = layer_33_k_cache;
  char *layer_33_v_cache = (char*)(0x75037e800000);
  all_tensors["layer_33_v_cache"] = layer_33_v_cache;
  char *layer_33_o_proj = (char*)(0x750279800000);
  all_tensors["layer_33_o_proj"] = layer_33_o_proj;
  char *layer_33_post_attn_layernorm = (char*)(0x75036a493000);
  all_tensors["layer_33_post_attn_layernorm"] = layer_33_post_attn_layernorm;
  char *layer_33_gate_proj = (char*)(0x75026d000000);
  all_tensors["layer_33_gate_proj"] = layer_33_gate_proj;
  char *layer_33_up_proj = (char*)(0x750273000000);
  all_tensors["layer_33_up_proj"] = layer_33_up_proj;
  char *layer_33_down_proj = (char*)(0x750267000000);
  all_tensors["layer_33_down_proj"] = layer_33_down_proj;
  char *layer_34_input_layernorm = (char*)(0x75036a495400);
  all_tensors["layer_34_input_layernorm"] = layer_34_input_layernorm;
  char *layer_34_q_proj = (char*)(0x750292800000);
  all_tensors["layer_34_q_proj"] = layer_34_q_proj;
  char *layer_34_k_proj = (char*)(0x750290000000);
  all_tensors["layer_34_k_proj"] = layer_34_k_proj;
  char *layer_34_v_proj = (char*)(0x750294800000);
  all_tensors["layer_34_v_proj"] = layer_34_v_proj;
  char *layer_34_q_norm = (char*)(0x75036a499600);
  all_tensors["layer_34_q_norm"] = layer_34_q_norm;
  char *layer_34_k_norm = (char*)(0x75036a499400);
  all_tensors["layer_34_k_norm"] = layer_34_k_norm;
  char *layer_34_k_cache = (char*)(0x750393000000);
  all_tensors["layer_34_k_cache"] = layer_34_k_cache;
  char *layer_34_v_cache = (char*)(0x75037f000000);
  all_tensors["layer_34_v_cache"] = layer_34_v_cache;
  char *layer_34_o_proj = (char*)(0x750290800000);
  all_tensors["layer_34_o_proj"] = layer_34_o_proj;
  char *layer_34_post_attn_layernorm = (char*)(0x75036a497400);
  all_tensors["layer_34_post_attn_layernorm"] = layer_34_post_attn_layernorm;
  char *layer_34_gate_proj = (char*)(0x750284000000);
  all_tensors["layer_34_gate_proj"] = layer_34_gate_proj;
  char *layer_34_up_proj = (char*)(0x75028a000000);
  all_tensors["layer_34_up_proj"] = layer_34_up_proj;
  char *layer_34_down_proj = (char*)(0x75027e000000);
  all_tensors["layer_34_down_proj"] = layer_34_down_proj;
  char *layer_35_input_layernorm = (char*)(0x75036a499800);
  all_tensors["layer_35_input_layernorm"] = layer_35_input_layernorm;
  char *layer_35_q_proj = (char*)(0x7502a9800000);
  all_tensors["layer_35_q_proj"] = layer_35_q_proj;
  char *layer_35_k_proj = (char*)(0x7502a7000000);
  all_tensors["layer_35_k_proj"] = layer_35_k_proj;
  char *layer_35_v_proj = (char*)(0x7502ab800000);
  all_tensors["layer_35_v_proj"] = layer_35_v_proj;
  char *layer_35_q_norm = (char*)(0x75036a49da00);
  all_tensors["layer_35_q_norm"] = layer_35_q_norm;
  char *layer_35_k_norm = (char*)(0x75036a49d800);
  all_tensors["layer_35_k_norm"] = layer_35_k_norm;
  char *layer_35_k_cache = (char*)(0x750393800000);
  all_tensors["layer_35_k_cache"] = layer_35_k_cache;
  char *layer_35_v_cache = (char*)(0x75037f800000);
  all_tensors["layer_35_v_cache"] = layer_35_v_cache;
  char *layer_35_o_proj = (char*)(0x7502a7800000);
  all_tensors["layer_35_o_proj"] = layer_35_o_proj;
  char *layer_35_post_attn_layernorm = (char*)(0x75036a49b800);
  all_tensors["layer_35_post_attn_layernorm"] = layer_35_post_attn_layernorm;
  char *layer_35_gate_proj = (char*)(0x75029b000000);
  all_tensors["layer_35_gate_proj"] = layer_35_gate_proj;
  char *layer_35_up_proj = (char*)(0x7502a1000000);
  all_tensors["layer_35_up_proj"] = layer_35_up_proj;
  char *layer_35_down_proj = (char*)(0x750295000000);
  all_tensors["layer_35_down_proj"] = layer_35_down_proj;
  char *model_norm_weight = (char*)(0x75036a49dc00);
  all_tensors["model_norm_weight"] = model_norm_weight;
  char *lm_head = (char*)(0x74f984000000);
  all_tensors["lm_head"] = lm_head;
  all_tensors["nullptr"] = nullptr;
  construct_task_graph(num_gpus, my_gpu_id, all_tasks, all_events, first_tasks, all_tensors);
}

__device__ __forceinline__
void _execute_task(TaskDesc const& task_desc,
                   RuntimeConfig const &runtime_config) {
  if (task_desc.task_type == TASK_EMBEDDING && task_desc.variant_id == 0) {
      kernel::embedding_kernel<bfloat16, 1, 64, 4096>(
      runtime_config.tokens + runtime_config.step[0], 
      task_desc.inputs[1].base_ptr,
      task_desc.outputs[0].base_ptr);

  }
  else if (task_desc.task_type == TASK_RMS_NORM_LINEAR && task_desc.variant_id == 0) {
      kernel::norm_linear_task_impl<bfloat16, 1, 64, 4096, 6144>(
      task_desc.inputs[0].base_ptr,
      task_desc.inputs[1].base_ptr,
      task_desc.inputs[2].base_ptr,
      1e-6f,
      task_desc.outputs[0].base_ptr);

  }
  else if (task_desc.task_type == TASK_RMS_NORM_LINEAR && task_desc.variant_id == 1) {
      kernel::norm_linear_task_impl<bfloat16, 1, 256, 4096, 24576>(
      task_desc.inputs[0].base_ptr,
      task_desc.inputs[1].base_ptr,
      task_desc.inputs[2].base_ptr,
      1e-6f,
      task_desc.outputs[0].base_ptr);

  }
  else if (task_desc.task_type == TASK_RMS_NORM_LINEAR && task_desc.variant_id == 2) {
      kernel::norm_linear_task_impl<bfloat16, 1, 1600, 4096, 153600>(
      task_desc.inputs[0].base_ptr,
      task_desc.inputs[1].base_ptr,
      task_desc.inputs[2].base_ptr,
      1e-6f,
      task_desc.outputs[0].base_ptr);

  }
  else if (task_desc.task_type == TASK_ATTENTION_1 && task_desc.variant_id == 0) {
      kernel::single_batch_decoding_kernel<bfloat16, 4, 1, 128, 1024>(
      task_desc.inputs[0].base_ptr,
      task_desc.inputs[1].base_ptr,
      task_desc.inputs[2].base_ptr,
      task_desc.outputs[0].base_ptr,
      runtime_config.step[0] + 1,
      true,
      true,
      task_desc.inputs[3].base_ptr,
      task_desc.inputs[4].base_ptr,
      task_desc.inputs[5].base_ptr,
      task_desc.inputs[6].base_ptr,
      1e-6f,
      1e-6f);

  }
  else if (task_desc.task_type == TASK_SILU_MUL_LINEAR_WITH_RESIDUAL && task_desc.variant_id == 0) {
      kernel::silu_mul_linear_task_impl<bfloat16, 1, 64, 12288, 4096>(
      task_desc.inputs[0].base_ptr,
      task_desc.inputs[1].base_ptr,
      task_desc.inputs[2].base_ptr,
      task_desc.outputs[0].base_ptr,
      runtime_config.my_gpu_id == 0);

  }
  else if (task_desc.task_type == TASK_LINEAR_WITH_RESIDUAL && task_desc.variant_id == 0) {
      kernel::linear_kernel<bfloat16, 1, 64, 4096, 4096>(
      task_desc.inputs[0].base_ptr,
      task_desc.inputs[1].base_ptr,
      task_desc.inputs[2].base_ptr,
      task_desc.outputs[0].base_ptr,
      runtime_config.my_gpu_id == 0);

  }
  else if (task_desc.task_type == TASK_ARGMAX_PARTIAL && task_desc.variant_id == 0) {
      kernel::argmax_partial_kernel<bfloat16, 1, 1600, 1>(
      task_desc.inputs[0].base_ptr,
      task_desc.outputs[0].base_ptr,
      task_desc.outputs[1].base_ptr);

  }
  else if (task_desc.task_type == TASK_ARGMAX_REDUCE && task_desc.variant_id == 0) {
      kernel::argmax_reduce_kernel<bfloat16, 1600, 96>(
      task_desc.inputs[0].base_ptr,
      task_desc.inputs[1].base_ptr,
      task_desc.outputs[0].base_ptr,
      runtime_config.step[0],
      runtime_config.tokens);

  }
}
