from typing import Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://localhost:9000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# bert-base-uncased
# bert-large-uncased
# bert-base-cased
# bert-large-cased
# bert-base-multilingual-uncased
# bert-base-multilingual-cased
# bert-base-chinese
# bert-base-german-cased
# bert-large-uncased-whole-word-masking
# bert-large-cased-whole-word-masking
# bert-large-uncased-whole-word-masking-finetuned-squad
# bert-large-cased-whole-word-masking-finetuned-squad
# bert-base-cased-finetuned-mrpc
# bert-base-german-dbmdz-cased
# bert-base-german-dbmdz-uncased
# bert-base-japanese
# bert-base-japanese-whole-word-masking
# bert-base-japanese-char
# bert-base-japanese-char-whole-word-masking
# bert-base-finnish-cased-v1
# bert-base-finnish-uncased-v1
# bert-base-dutch-cased
# openai-gpt
# gpt2
# gpt2-medium
# gpt2-large
# gpt2-xl
# transfo-xl-wt103
# xlnet-base-cased
# xlnet-large-cased
# xlm-mlm-en-2048
# xlm-mlm-ende-1024
# xlm-mlm-enfr-1024
# xlm-mlm-enro-1024
# xlm-mlm-xnli15-1024
# xlm-mlm-tlm-xnli15-1024
# xlm-clm-enfr-1024
# xlm-clm-ende-1024
# xlm-mlm-17-1280
# xlm-mlm-100-1280
# roberta-base
# roberta-large
# roberta-large-mnli
# distilroberta-base
# roberta-base-openai-detector
# roberta-large-openai-detector
# distilbert-base-uncased
# distilbert-base-uncased-distilled-squad
# distilbert-base-cased
# distilbert-base-cased-distilled-squad
# distilgpt2
# distilbert-base-german-cased
# distilbert-base-multilingual-cased
# ctrl
# camembert-base
# albert-base-v1
# albert-large-v1
# albert-xlarge-v1
# albert-xxlarge-v1
# albert-base-v2
# albert-large-v2
# albert-xlarge-v2
# albert-xxlarge-v2
# t5-small
# t5-base
# t5-large
# t5-3B
# t5-11B
# xlm-roberta-base
# xlm-roberta-large
# flaubert-small-cased
# flaubert-base-uncased
# flaubert-base-cased
# flaubert-large-cased
# bart-large
# bart-large-mnli
# bart-large-cnn
# mbart-large-en-ro
# DialoGPT-small
# DialoGPT-medium
# DialoGPT-large

@app.get("/get_filter_options")
@app.post("/get_filter_options")
def get_filter_options():
    return {
        "models": [
            "distilbert-base-cased-distilled-squad",
            "t5-large",
            "t5-small",
            "t5-base",
            "t5-3b",
            "t5-11b",
            "google/mt5-base",
            "google/mt5-small",
            "google/mt5-large",
            "google/mt5-xxl",
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large",
            "microsoft/DialoGPT-small",
            "bigscience/bloom-560m",
            "bigscience/bloom-1b1",
            "bigscience/bloom-1b7",
            "bigscience/bloom-3b",
            "bigscience/bloom-7b1",
            "bigscience/bloom", 
            "bigscience/bloomz-560m",
            "bigscience/bloomz-1b1",
            "bigscience/bloomz-1b7",
            "bigscience/bloomz-3b",
            "bigscience/bloomz-7b1",
            "bigscience/bloomz", 
        ],
        "devices": [
            "cpu",
            "cuda"
        ]
    }

from pydantic import BaseModel

class MessageOptions(BaseModel):
    model: str
    device: str
    message: str
    seed: int

chat_history_ids=[]

@app.post("/process_chat_message")
def process_chat_message(oArgs: MessageOptions):
    try:
        print(oArgs)

        if (oArgs.model.startswith("microsoft/DialoGPT-")):
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            tokenizer = AutoTokenizer.from_pretrained(oArgs.model)
            model = AutoModelForCausalLM.from_pretrained(oArgs.model)

            sR = ""
            # Let's chat for 5 lines
            # encode the new user input, add the eos_token and return a tensor in Pytorch
            new_user_input_ids = tokenizer.encode(oArgs.message + tokenizer.eos_token, return_tensors='pt')
            # append the new user input tokens to the chat history
            # bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) 
            # if step > 0 else new_user_input_ids
            # generated a response while limiting the total chat history to 1000 tokens,
            chat_history_ids = model.generate(
                new_user_input_ids, 
                max_length=1000, 
                pad_token_id=tokenizer.eos_token_id
            )
            # pretty print last ouput tokens from bot
            # sR = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

            sR = tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)

            oR = { "message": sR }
        elif (oArgs.model.startswith("distilbert-")):
            from transformers import pipeline
            question_answerer = pipeline("question-answering", model=oArgs.model)

            aM = oArgs.message.split("\n")
            sQ = aM.pop()
            sC ="\n".join(aM)
            result = question_answerer(question=sQ, context=sC)

            oR = { "message": result['answer'] }
        elif (oArgs.model.startswith("google/mt5-")):
            # Import generic wrappers
            from transformers import AutoModel, AutoTokenizer 

            # Download pytorch model
            model = AutoModel.from_pretrained(oArgs.model)
            tokenizer = AutoTokenizer.from_pretrained(oArgs.model)

            input_ids = tokenizer.encode(oArgs.message, return_tensors='pt')

            outputs = model.generate(input_ids=input_ids)

            sR = tokenizer.decode(outputs[0], skip_special_tokens=True)

            oR = { "message": sR }
        elif (oArgs.model.startswith("t5-")):
            from transformers import T5Tokenizer, T5ForConditionalGeneration

            # Initialize the T5 tokenizer and model
            tokenizer = T5Tokenizer.from_pretrained('t5-base')
            model = T5ForConditionalGeneration.from_pretrained('t5-base')

            # Tokenize the input text
            input_ids = tokenizer.encode(oArgs.message, return_tensors='pt')

            # Generate a response
            outputs = model.generate(input_ids=input_ids)

            # Print the response
            sR = tokenizer.decode(outputs[0], skip_special_tokens=True)

            oR = { "message": sR }

            """
            import torch
            from transformers import AutoTokenizer, AutoModelWithLMHead

            tokenizer = AutoTokenizer.from_pretrained(oArgs.model)
            model = AutoModelWithLMHead.from_pretrained(oArgs.model, return_dict=True)

            inputs = tokenizer.encode(
                oArgs.message,
                return_tensors='pt',
                max_length=512,
                truncation=True
            )

            summary_ids = model.generate(
                inputs, 
                max_length=600, 
                num_beams=2
            )

            sR = tokenizer.decode(summary_ids[0])

            oR = { "message": sR }
            """
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
            import torch

            device = torch.device(oArgs.device)

            # torch.set_default_tensor_type(torch.cuda.FloatTensor)
            model = AutoModelForCausalLM.from_pretrained(oArgs.model, use_cache=True).to(oArgs.device)
            tokenizer = AutoTokenizer.from_pretrained(oArgs.model)

            # set_seed(oArgs.seed)

            # print("size: "+len(oArgs.message))

            message = oArgs.message # "Q: "+oArgs.message+"\nA:"
            input_ids = tokenizer(message, return_tensors="pt").to(oArgs.device)
            sample = model.generate(
                **input_ids, 
                max_length=1000, 
                top_k=1, 
                temperature=0.9, 
                repetition_penalty = 2.0
            )

            oR = { "message": tokenizer.decode(sample[0], skip_special_tokens=True) }

        print(oR)

        return oR
    except Exception as e: # work on python 3.x
        return { "type": "error", "message": str(e) }


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}