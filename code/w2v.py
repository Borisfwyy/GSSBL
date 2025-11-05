import time
import json
from gensim.models import Word2Vec

def train_cbow_model(data, vector_size=100, window=5, min_count=1, sg=0,
                     negative=5, alpha=0.025, workers=4, epochs=5,
                     model_filename: str = 'model/cbow/cbow00.model',
                     vectors_filename: str = 'model/cbow/cbow00.json',
                     filter_stopwords: bool = False):

    if not data:
        raise ValueError("Training data is empty!")

    if filter_stopwords:
        stopwords = {"none", "<STOP>", "<PAD>"}  
        data = [[word for word in sentence if word not in stopwords] for sentence in data]

    start_time = time.time()

    model = Word2Vec(
        sentences=data,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        negative=negative,
        alpha=alpha,
        workers=workers,
        epochs=epochs
    )

    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds.")

    save_model(model, model_filename, vectors_filename)

    return model

def save_model(model, model_filename: str = 'cbow_model.model', vectors_filename: str = 'word_vectors.json'):

    model.save(model_filename)
    print(f"Model saved to {model_filename}")

    word_vectors = {word: model.wv[word].tolist() for word in model.wv.index_to_key}
    with open(vectors_filename, 'w', encoding='utf-8') as f:
        json.dump(word_vectors, f, ensure_ascii=False, indent=4)
    print(f"Word vectors saved to {vectors_filename}")
