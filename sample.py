import pickle

with open('sentiment_api/model/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Example usage
texts = [
    "I love this hockey game! It's so exciting.",
    "I hate the weather today.",
    "The sun is not chill"]

# proba = model.predict_proba([texts[1]])[0]
# print(proba)

# pred = model.predict([texts[1]])[0]
# print(pred)

for text in texts:
    pred = model.predict([text])[0]
    label = "positive" if pred == 1 else "negative"
    print(f"Text: '{text}' -> Sentiment: {label}")

# try on other data
# Download IMDB dataset (50k real sentiment reviews)
# import urllib.request, tarfile

# URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
# if not os.path.exists("aclImdb"):
#     print("Downloading IMDB dataset (~80MB)...")
#     urllib.request.urlretrieve(URL, "aclImdb_v1.tar.gz")
#     print("Extracting...")
#     with tarfile.open("aclImdb_v1.tar.gz") as f:
#         f.extractall()
#     os.remove("aclImdb_v1.tar.gz")

# # Load train split
# print("Loading data...")
# data = load_files("aclImdb/train", categories=["pos", "neg"], encoding="utf-8", decode_error="replace")
# X_train, X_test, y_train, y_test = train_test_split(
#     data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
# )