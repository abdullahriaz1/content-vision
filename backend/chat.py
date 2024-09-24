from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

def model(input):
  model_id = "CohereForAI/c4ai-command-r-plus"
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForCausalLM.from_pretrained(model_id)

  # Format message with the command-r-plus chat template
  messages = [{"role": "user", "content": input}]
  input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
  ## <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello, how are you?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>

  gen_tokens = model.generate(
      input_ids, 
      max_new_tokens=100, 
      do_sample=True, 
      temperature=0.3,
      )

  gen_text = tokenizer.decode(gen_tokens[0])
  return gen_text
  
def main():
  prompt = ""
  for line in sys.argv:
    prompt += " " + line
  print(model(prompt))

if __name__ == "__main__":
  main()

