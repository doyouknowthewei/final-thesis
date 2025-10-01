# backend.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import torch
import torch.nn as nn
import pickle
import re
import numpy as np
from textblob import TextBlob
from transformers import BertTokenizer, BertModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import time
import nltk
from nltk.stem import WordNetLemmatizer

# --- Downloads for NLTK (only once) ---
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# --- Initialize FastAPI ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Constants ---
MAX_LEN = 128
VOCAB_SIZE = 30000
EMBEDDING_DIM = 300
KERAS_MODEL_PATH = "best_toxic_model.h5"
KERAS_TOKENIZER_PATH = "original_tokenizer.pickle"
HYBRID_MODEL_PATH = "best_hybrid_model.pt"
LSTM_TOKENIZER_PATH = "lstm_tokenizer.pickle"

IDENTITY_TERMS = [
    # Race / Ethnicity
    "muslim", "jew", "jewish", "black", "white", "asian", "latino", "latina", "hispanic",
    "african", "indian", "arab", "middle eastern", "asian american", "african american",
    "native american", "indigenous", "pacific islander", "caribbean", "european", "south asian",
    "east asian", "west african", "north african", "mestizo", "creole", "aboriginal",

    # Nationalities (important for identity recognition)
    "american", "mexican", "canadian", "chinese", "japanese", "korean", "vietnamese", "filipino",
    "indonesian", "malaysian", "pakistani", "bangladeshi", "nigerian", "ethiopian", "somali",
    "egyptian", "saudi", "iranian", "iraqi", "syrian", "turkish", "brazilian", "argentinian",
    "colombian", "chilean", "peruvian", "venezuelan", "german", "french", "british", "spanish",
    "italian", "russian", "ukrainian", "polish", "greek", "swedish", "norwegian", "dutch",

    # Religion / Belief systems
    "christian", "catholic", "protestant", "orthodox", "mormon", "buddhist", "hindu", "sikh",
    "atheist", "agnostic", "pagan", "wiccan", "spiritualist", "evangelical", "lutheran", "presbyterian",
    "shinto", "taoist", "zoroastrian", "scientologist", "secular", "humanist", "sunni", "shia",

    # Gender and Sexual Identity
    "gay", "lesbian", "bisexual", "queer", "trans", "transgender", "nonbinary", "genderqueer",
    "genderfluid", "intersex", "asexual", "demisexual", "pansexual", "heterosexual", "straight",
    "cisgender", "two-spirit",

    # Family / Gendered terms
    "women", "woman", "men", "man", "girl", "boy", "female", "male",
    "mother", "father", "daughter", "son", "brother", "sister", "wife", "husband",
    "grandmother", "grandfather", "grandson", "granddaughter", "parent", "child", "children",

    # Disability / Health Status
    "disabled", "handicapped", "differently abled", "autistic", "aspergers", "deaf", "blind",
    "hard of hearing", "mute", "speech impaired", "paralyzed", "quadriplegic", "amputee",
    "wheelchair user", "mental illness", "schizophrenic", "bipolar", "depressed", "anxious",
    "adhd", "ptsd", "cancer survivor", "addict", "alcoholic", "recovering addict",

    # Age groups
    "elderly", "senior citizen", "youth", "teenager", "child", "kids", "young adult", "middle aged",
    "baby", "toddler", "infant", "adolescent", "geriatric",

    # Socioeconomic / Other community identifiers
    "immigrant", "refugee", "migrant", "asylum seeker", "veteran", "military spouse",
    "homeless", "poor", "low income", "working class", "middle class", "wealthy", "rich",
    "orphan", "foster child", "widow", "widower"
]


TOXIC_WORDS = {
    # General toxic words
    "abuse", "idiot", "hate", "stupid", "kill", "moron", "nazi", "terrorist",
    "trash", "dumb", "racist", "sexist", "retard", "freak", "scum", "ignorant",
    "whore", "slut", "loser", "ugly", "bastard", "dirtbag", "asshole",
    "jerk", "pussy", "faggot","fag", "bitch", "douchebag", "fat", "coward", "creep",
    "psycho", "evil", "satan", "scumbag", "retarded", "dumbass", "sicko", "filthy",
    "stinker", "maniac", "lunatic", "disease", "vermin", "clown", "imbecile",
    "idiotic", "delusional", "monster", "disgusting", "horrible", "gross",
    "disgrace", "failure", "degenerate", "toxic", "worthless", "subhuman",
    "parasite", "dirt", "worm", "cockroach", "rodent", "backstabber", "traitor",
    "liar", "cheater", "snake", "villain", "nuisance", "menace", "harass",
    "low-life", "bottomfeeder", "pathetic", "imbecile", "airhead", "bonehead",
    "halfwit", "simpleton", "nitwit", "nincompoop", "butthead", "blockhead",

    # Threatening words
    "threat", "die", "death", "murder", "attack", "explode", "shoot",
    "stab", "hang", "lynch", "rape", "molest", "assault", "cripple", "destroy",
    "ruin", "terrorize", "fear", "violence", "bomb", "kidnap", "execute", "ambush",
    "annihilate", "slaughter", "strangle", "behead", "decapitate", "gun", "bullet",
    "raped", "kill", "poison", "torture", "abduct", "assassinate", "slay",
    "execute", "eradicate", "obliterate", "massacre", "slaughter", "bloodbath",

    # Identity-related slurs (⚠️ offensive, listed for filtering purposes only)
    "tranny", "kike", "chink", "gook", "spic", "beaner", "wetback", "raghead",
    "sandnigger", "coon", "porchmonkey", "redskin", "dyke", "queer", "fairy",
    "shemale", "jap", "paki", "nigger", "nigga", "camel jockey",

    # Curse words (profanity)
    "fuck", "fucking", "fucker", "motherfucker", "shit", "bullshit", "piss", "pissed",
    "dick", "dickhead", "prick", "cock", "cockhead", "cocksucker", "bastard",
    "ass", "asshat", "asswipe", "jackass", "twat", "wanker", "cum", "cumdumpster",
    "cunt", "goddamn", "damn", "hell", "bitchass", "shitty", "dipshit", "whorebag",
    "son of a bitch", "sonofabitch", "slutbag", "shithead", "fuckhead", "shitface",
    "fucksake", "douche", "douchelord", "bastardized", "fubar",

    # Miscellaneous harsh/violent words
    "slimeball", "homewrecker", "twit", "yob", "git", "tosser", "muppet",
    "loserface", "uggo", "dimwit", "scumbucket", "arsehole", "arse", "arsewipe"
}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

lemmatizer = WordNetLemmatizer()

class TextOnlyInput(BaseModel):
    text: str

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|\#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.lower()

def compute_subjectivity(text):
    return float(TextBlob(text).sentiment.subjectivity)

def has_identity_term(text):
    text_lower = text.lower()
    return int(any(term in text_lower for term in IDENTITY_TERMS))

def detect_toxic_words(text):
    tokens = re.findall(r"\b\w+\b", text.lower())
    toxic_matches = []
    toxic_lemmas = {lemmatizer.lemmatize(word): word for word in TOXIC_WORDS}

    for token in tokens:
        lemma_token = lemmatizer.lemmatize(token)
        if lemma_token in toxic_lemmas:
            toxic_matches.append(token)

    return list(set(toxic_matches))

def load_keras_model_and_tokenizer():
    keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    with open(KERAS_TOKENIZER_PATH, 'rb') as handle:
        keras_tokenizer = pickle.load(handle)
    return keras_model, keras_tokenizer

class HybridLSTM_BERT(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, lstm_hidden=256, bert_model_name='bert-base-uncased', num_classes=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden, batch_first=True, dropout=dropout)
        self.bert = BertModel.from_pretrained(bert_model_name)
        combined_dim = lstm_hidden + self.bert.config.hidden_size + 2
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(combined_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_seq, input_ids, attention_mask, subjectivity, identity_flag):
        emb = self.embedding(lstm_seq)
        emb = self.lstm_input_dropout(emb)
        lstm_out, _ = self.lstm(emb)
        lstm_out = lstm_out[:, -1, :]

        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        combined = torch.cat([lstm_out, bert_out, subjectivity, identity_flag], dim=1)
        x = self.dropout(combined)
        logits = self.classifier(x)
        probs = self.softmax(logits)
        return probs

def load_hybrid_model_and_tokenizers():
    model = HybridLSTM_BERT(vocab_size=VOCAB_SIZE).to(device)
    state_dict = torch.load(HYBRID_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    with open(LSTM_TOKENIZER_PATH, 'rb') as handle:
        lstm_tokenizer = pickle.load(handle)

    return model, bert_tokenizer, lstm_tokenizer

keras_model, keras_tokenizer = load_keras_model_and_tokenizer()
hybrid_model, bert_tokenizer, lstm_tokenizer = load_hybrid_model_and_tokenizers()

def predict_with_keras(text):
    cleaned = clean_text(text)
    sequence = keras_tokenizer.texts_to_sequences([cleaned])
    sequence = pad_sequences(sequence, maxlen=MAX_LEN)
    probs = keras_model.predict(sequence)[0]
    return probs

def predict_with_hybrid(text):
    cleaned = clean_text(text)
    lstm_seq = lstm_tokenizer.texts_to_sequences([cleaned])
    lstm_seq = pad_sequences(lstm_seq, maxlen=MAX_LEN)
    lstm_seq = torch.tensor(lstm_seq, dtype=torch.long).to(device)

    bert_enc = bert_tokenizer(cleaned, return_tensors="pt", padding='max_length', truncation=True, max_length=MAX_LEN)
    input_ids = bert_enc['input_ids'].to(device)
    attention_mask = bert_enc['attention_mask'].to(device)

    subjectivity = torch.tensor([[compute_subjectivity(text)]], dtype=torch.float).to(device)
    identity_flag = torch.tensor([[has_identity_term(text)]], dtype=torch.float).to(device)

    with torch.no_grad():
        hybrid_model.to(device)
        probs = hybrid_model(lstm_seq, input_ids, attention_mask, subjectivity, identity_flag)

    return probs.squeeze().cpu().numpy()

def classify(probs):
    return int(np.argmax(probs))

def format_percentages(probs):
    return f"Non-toxic: {probs[0]*100:.2f}%, Toxic: {probs[1]*100:.2f}%"

@app.post("/predict")
async def predict_both_models(input_data: TextOnlyInput):
    text = input_data.text

    start_lstm = time.time()
    lstm_probs = predict_with_keras(text)
    lstm_time = time.time() - start_lstm
    lstm_toxic_words = detect_toxic_words(text)

    start_hybrid = time.time()
    hybrid_probs = predict_with_hybrid(text)
    hybrid_time = time.time() - start_hybrid
    hybrid_toxic_words = detect_toxic_words(text)

    return {
        "baseline_model": "Keras LSTM",
        "baseline_probabilities": lstm_probs.tolist(),
        "baseline_toxicity_status": classify(lstm_probs),
        "baseline_percentages": format_percentages(lstm_probs),
        "baseline_inference_time": lstm_time,
        "baseline_toxic_words": lstm_toxic_words,

        "proposed_model": "Hybrid LSTM+BERT",
        "proposed_probabilities": hybrid_probs.tolist(),
        "proposed_toxicity_status": classify(hybrid_probs),
        "proposed_percentages": format_percentages(hybrid_probs),
        "proposed_inference_time": hybrid_time,
        "proposed_toxic_words": hybrid_toxic_words
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5001)
