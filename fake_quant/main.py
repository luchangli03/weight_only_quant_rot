import sys
import torch
import utils
import model_utils
import data_utils
import transformers
import quant_utils
import rotation_utils
import gptq_utils
import eval_utils
import hadamard_utils
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":
    model_path_in = sys.argv[1]
    model_path_out = sys.argv[2]

    model = AutoModelForCausalLM.from_pretrained(
        model_path_in, device_map="auto", torch_dtype=torch.float32, trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    rotation_utils.fuse_layer_norms(model)
    rotation_utils.rotate_model(model, args)

    model.save_pretrained(model_path_out)
    tokenizer.save_pretrained(model_path_out)
