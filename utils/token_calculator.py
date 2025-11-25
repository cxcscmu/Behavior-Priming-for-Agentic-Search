from transformers import AutoTokenizer

model_name = "Qwen/Qwen3-8B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(input):
    tokens = tokenizer.encode(input)
    token_length = len(tokens)
    return token_length


def main():
    input = "Hello, world!"
    print(tokenize(input))


if __name__ == "__main__":
    main()



