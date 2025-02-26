import json
import re
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
os.environ['HF_TOKEN'] = "<your_hugging_face_token>"


def extract_tool_call(generated_text):
    # Extract the content between <tool_call> tags
    pattern = r"(?<=<tool_call>).*?(?=</tool_call>)"
    match = re.findall(pattern, generated_text, re.DOTALL | re.MULTILINE)

    if not match:
        return None

    # Get the content inside tool_call tags
    tool_call_content = match[2]

    try:
        # Replace single quotes with double quotes for JSON compatibility
        tool_call_json = tool_call_content.replace("'", '"')

        # Parse the JSON
        tool_data = json.loads(tool_call_json)
        return tool_data

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True)

username = "publicfax"
output_dir = "gemma-2-2B-it-thinking-function_calling-V0"
peft_model_id = f"{username}/{output_dir}"


config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model, peft_model_id)
model.to(torch.bfloat16)
model.eval()


def get_current_temperature(location: str):
    """
    A tool that gets the temperature at a given location.

    Args:
        location: A string representing a valid city (e.g., 'New York').
    """
    if (location == "Copenhagen"):
        return -7000.2
    return 20.0


tools = [get_current_temperature]

user_prompt = """
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.
You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.
Here are the available tools:
<tools> 
[
    {
        'type': 'function', 
        'function': {'name': 'get_current_temperature', 'description': 'A tool that gets the temperature at a given location', 
        'parameters': {'type': 'object', 'properties': {'location': {'type': 'string', 'description': 'A string representing a valid city (e.g., 'New York')'}}, 
        'required': ['location']
    }
] 
</tools>
Use the following pydantic model json schema for each tool call you will make: {'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']}
For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{tool_call}
</tool_call>
Also, before making a call to a function take the time to plan the function to take. Make that thinking process between <think>{your thoughts}</think>

Hi, I need to know what the weather is like in Copenhagen. Can you help me with that?
"""

chat = [
    {"role": "user", "content": user_prompt}
]

tool_prompt = tokenizer.apply_chat_template(
    chat,
    tools=tools,
    add_generation_prompt=True,
    return_tensors="pt"
)
tool_prompt = tool_prompt.to(model.device)
out = model.generate(tool_prompt, max_new_tokens=300)
generated_text = tokenizer.decode(out[0])
# generated_text = """ < bos > <start_of_turn > user\nYou are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.\nYou may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.\nHere are the available tools:\n<tools> \n[\n    {\n        'type': 'function', \n        'function': {'name': 'get_current_temperature', 'description': 'A tool that gets the temperature at a given location', \n        'parameters': {'type': 'object', 'properties': {'location': {'type': 'string', 'description': 'A string representing a valid city (e.g., 'New York')'}}, \n        'required': ['location']\n    }\n] \n</tools>\nUse the following pydantic model json schema for each tool call you will make: {'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']}\nFor each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n<tool_call>\n{tool_call}\n</tool_call>\nAlso, before making a call to a function take the time to plan the function to take. Make that thinking process between <think>{your thoughts}</think>\n\nHi, I need to know what the weather is like in Copenhagen. Can you help me with that?<end_of_turn><eos>\n<start_of_turn>model\n<think>Okay, so the user is asking about the weather in Copenhagen. I need to figure out how to respond. Looking at the available tools, there's a function called get_current_temperature that can provide the current temperature at a specific location. The user provided the city, so I can use that as the location parameter. I should call this function to get the information the user is looking for.\n</think><tool_call>\n{'name': 'get_current_temperature', 'arguments': {'location': 'Copenhagen'}}\n</tool_call><end_of_turn>"""

tool = extract_tool_call(generated_text)

result = None

if tool:

    try:
        # Map model tool call name to the corresponding function
        if tool["name"] == "get_current_temperature":
            location = tool["arguments"].get("location", "")
            # Call the weather tool function with the provided location
            result = get_current_temperature(location)
            print("Tool call result:", result)
        else:
            print("No mapping for tool:", tool["name"])
    except Exception as e:
        print("Failed to parse tool call:", e)
else:
    print("No tool call found in model output.")

chat.append({"role": "tool", "content": f"{result}"})
user_prompt = tokenizer.apply_chat_template(
    chat,
    add_generation_prompt=True,
    return_tensors="pt"
)
user_prompt = user_prompt.to(model.device)
outputs = model.generate(user_prompt, max_new_tokens=300)
print(final_output := tokenizer.decode(outputs[0]))
