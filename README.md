# End-to-end-Medical-Chatbot-using-Llama2

## Steps to run the project

### STEP 01 - Create a conda envirnment after opening the repository

```bash
conda create -n mchatbot python=3.8 -y
```

```bash
conda activate mchatbot
```

### STEP 02 - Install the required dependencies

```bash
pip install -r requirements.txt
```

### Create a `.env` file in the root directory and add your Pinecone credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
PINECONE_API_ENV = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Download the quantize model from the link provided in model folder & keep the model in the model directory:

```ini
## Download the Llama 2 Model:

llama-2-7b-chat.ggmlv3.q4_0.bin


## From the following link:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
```

### To process the PDF file into text chunks and store it as embeddings in Pinecone run the following:
```bash
# run the following command
python store_index.py
```

### To run the app run:
```bash
# Finally run the following command
python app.py
```

### Open the app
```bash
open up localhost:
```


## Techstack Used:

- Python
- LangChain
- Flask
- Meta Llama2
- Pinecone