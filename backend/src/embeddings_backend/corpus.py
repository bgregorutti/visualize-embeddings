from datasets import load_dataset

def contains_specials(word, specials):
    return set(word).intersection(set(specials))

def get_tokens(corpus):
    to_words = lambda sentence: {word.strip() for word in sentence.strip().split(" ")}

    stop_words = {
        "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for", "with",
        "from", "by", "of", "is", "are", "was", "were", "be", "been", "being", "that",
        "this", "these", "those", "as", "it", "its", "he", "she", "they", "them", "his",
        "her", "their", "we", "us", "you", "your", "i", "me", "my", "mine", "not", "so",
        "such", "than", "too", "very", "can", "could", "should", "would"
    }

    specials = (".", ",", ";", "/", "'", "'s", "-", "´", "(", ")", "$", "ä", "ď", '"', ":", "?", "!")

    words = set()
    for sentence in corpus:
        sentence = " ".join(token.lower() for token in sentence.strip().split() if token.lower() not in stop_words)
        words = words.union(to_words(sentence))
    
    without_specials = set()
    for word in words:
        if not contains_specials(word, specials) and not contains_specials(word, "0123456789"):
            without_specials.add(word)

    return without_specials

if __name__ == "__main__":
    dataset = load_dataset("sentence-transformers/stsb")
    train = dataset.get("train").to_dict().get("sentence1")
    test =  dataset.get("test").to_dict().get("sentence1")
    validation =  dataset.get("validation").to_dict().get("sentence1")
    tokens = get_tokens(train+test+validation)
    print(len(tokens))
    
    with open("tmp.txt", "w") as f:
        f.write(";".join(tokens))
