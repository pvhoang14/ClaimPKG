from openai import OpenAI

DEFAULT_MODEL = "Llama-3.3-70B-Instruct"
SERVER_HOST = "http://0.0.0.0:8148"
TEMPERATURE = 0.0
TOP_P = 0.9
MAX_TOKENS = 1024


def get_completion_vllm(
    input_prompt,
    system_prompt=None,
    model=DEFAULT_MODEL,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_tokens=MAX_TOKENS,
    server_host=SERVER_HOST,
):
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"{server_host}/v1",
    )
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": input_prompt})
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            seed=0,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return None
