import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig


def init_model_pipe(model_name, model_adapter=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_adapter:
        peft_config = PeftConfig.from_pretrained(model_adapter)
        model = PeftModel.from_pretrained(model, model_adapter)
        model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return pipe

def get_model_response(pipe, prompt, max_new_token, temperature, top_p):
    kwargs = {
        "text_inputs": [{"role": "user", "content": prompt}],
        "max_new_tokens": max_new_token,
        "top_p": top_p
    }
    if temperature == 0.0:
        kwargs["do_sample"] = False
    else:
        kwargs["temperature"] = temperature

    return pipe(**kwargs)[0]["generated_text"][-1]["content"].strip()

def get_api_response(client, prompt, model_engine, max_new_token, temperature, top_p):
    response = client.chat.completions.create(
        model=model_engine,
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_new_token,
        temperature=temperature,
        top_p=top_p,
    )
    return response.choices[0].message.content.strip()