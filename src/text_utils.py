import re
from collections import Counter
from functools import lru_cache

import torch
from torch.utils.data import Dataset

from config import NONIID_LABELS, TEXT_CHUNK_OVERLAP, TEXT_CHUNK_SIZE

MAX_VOCAB_SIZE = 10000
MAX_SEQ_LEN = 50

LOCAL_TEXT_CORPUS = {
    1: [
        "Parliament approved a fiscal package after a week of negotiations between coalition leaders, regional governors, and public sector unions. The proposal keeps core services funded while shifting money toward digital infrastructure and municipal planning.",
        "Government officials announced a cross border agreement to share environmental data, strengthen port inspections, and coordinate emergency procurement. Analysts said the accord reduces duplication and gives local agencies faster access to operational information.",
        "City administrators opened a public consultation on transport reform, asking residents to comment on bus priority corridors, station redesign, and night service coverage. The mayor argued that the plan will reduce delays and improve access to jobs and schools.",
        "A committee in the senate released a report on cybersecurity standards for public institutions, recommending annual audits, stronger vendor controls, and mandatory incident disclosure. Opposition members supported the recommendations but requested stricter enforcement deadlines.",
        "Diplomatic teams met for a second round of talks focused on supply resilience, border logistics, and academic exchange programs. The joint statement emphasized predictable rules, transparent reporting, and long term cooperation between neighboring states.",
    ],
    2: [
        "A manufacturing group expanded its distribution network after demand for industrial sensors increased across logistics centers and energy plants. Executives said the investment should shorten delivery times and improve coordination with regional suppliers.",
        "Retailers reported steady consumer spending despite tighter credit conditions, helped by stronger online subscriptions and more efficient inventory planning. Several chains said disciplined pricing allowed them to protect margins without aggressive discount campaigns.",
        "A payments company launched a settlement platform for small exporters, offering automated invoices, currency conversion, and shipment tracking in one interface. Business clients said the service reduces paperwork and provides faster confirmation for cross border orders.",
        "Energy markets reacted to a revised storage forecast as utilities increased purchases ahead of winter. Economists noted that long term contracts and more flexible demand programs helped moderate volatility compared with earlier quarters.",
        "An agricultural cooperative invested in cold chain upgrades, warehouse analytics, and route optimization software to reduce waste during peak harvest weeks. Managers expect the changes to improve product quality and increase revenue stability for member farms.",
    ],
    3: [
        "Engineers presented a language compression prototype that converts long documents into compact latent vectors before reconstructing text on a remote node. The team said the design favors semantic preservation over exact word by word copying.",
        "Researchers trained a lightweight text model to summarize sensor events, maintenance reports, and operator notes in industrial environments. Their evaluation showed that chunk based encoding can keep the central meaning even when communication bandwidth is limited.",
        "A laboratory published results on robust federated learning for edge devices, combining noisy channels, partial updates, and asynchronous aggregation. The study found that careful batching and regular checkpointing reduced instability during collaborative training.",
        "Software developers released a toolkit for document completion that masks sections of text and predicts plausible continuations from the available context. The project targets enterprise support systems where incomplete forms and fragmented reports are common.",
        "Scientists monitoring coastal ecosystems used automated text pipelines to merge ship logs, weather bulletins, and field observations into unified reports. The workflow reduced manual transcription effort and improved consistency across research teams.",
    ],
    4: [
        "The coaching staff changed its training calendar after reviewing player recovery data, travel load, and match intensity. Trainers said shorter sessions with more tactical drills improved concentration late in the week.",
        "A championship final turned on patient possession and disciplined defending as the underdog club absorbed pressure before scoring twice on fast transitions. Commentators praised the team for balancing risk and control in the decisive moments.",
        "Organizers of a city marathon introduced staggered start times, revised hydration points, and real time route alerts for participants. Runners reported smoother movement through crowded areas and fewer interruptions at support stations.",
        "A youth academy expanded its scouting network and added performance analysis sessions focused on passing patterns, spatial awareness, and recovery runs. Club directors believe the program will create a more sustainable path to the senior squad.",
        "Medical staff cleared a veteran captain to return after a structured rehabilitation plan that tracked strength, mobility, and workload tolerance. The player said transparent communication with coaches helped manage expectations throughout recovery.",
    ],
}


class SimpleVocab:
    def __init__(self, token_counter, max_tokens=MAX_VOCAB_SIZE, specials=None):
        specials = specials or ["<unk>", "<pad>", "<bos>", "<eos>"]
        self.itos = list(specials)
        for token, _ in token_counter.most_common(max(0, max_tokens - len(self.itos))):
            if token not in specials:
                self.itos.append(token)
        self.stoi = {token: index for index, token in enumerate(self.itos)}
        self.default_index = self.stoi["<unk>"]

    def __call__(self, tokens):
        return [self.stoi.get(token, self.default_index) for token in tokens]

    def __getitem__(self, token):
        return self.stoi[token]

    def __len__(self):
        return len(self.itos)

    def get_itos(self):
        return self.itos

    def set_default_index(self, index):
        self.default_index = index


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def basic_tokenize(text):
    return re.findall(r"[a-z0-9']+", normalize_whitespace(text).lower())


def get_corpus_documents(train=True):
    documents = []
    for label, texts in LOCAL_TEXT_CORPUS.items():
        split_index = max(1, len(texts) - 1)
        selected = texts[:split_index] if train else texts[split_index:]
        for text in selected:
            documents.append((label, normalize_whitespace(text)))
    return documents


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield basic_tokenize(text)


def pad_token_ids(token_ids, vocab, seq_len=MAX_SEQ_LEN):
    trimmed = token_ids[:seq_len]
    if len(trimmed) < seq_len:
        trimmed = trimmed + [vocab["<pad>"]] * (seq_len - len(trimmed))
    return trimmed


def chunk_token_ids(token_ids, chunk_size=None, overlap=None):
    chunk_size = chunk_size or TEXT_CHUNK_SIZE or MAX_SEQ_LEN
    overlap = TEXT_CHUNK_OVERLAP if overlap is None else overlap
    chunk_size = max(1, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size - 1))

    if not token_ids:
        return [[]]

    step = max(1, chunk_size - overlap)
    return [token_ids[start:start + chunk_size] for start in range(0, len(token_ids), step)]


@lru_cache(maxsize=1)
def build_or_load_vocab():
    """Build a local vocabulary from the bundled text corpus."""
    counter = Counter()
    for tokens in yield_tokens(get_corpus_documents(train=True) + get_corpus_documents(train=False)):
        counter.update(tokens)
    vocab = SimpleVocab(counter, max_tokens=MAX_VOCAB_SIZE)
    vocab.set_default_index(vocab["<unk>"])
    return vocab


class TextDataset(Dataset):
    def __init__(self, data_iter, vocab, max_seq_len=MAX_SEQ_LEN, allowed_labels=None):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.allowed_labels = set(allowed_labels or [])
        self.data = []
        sequence_overlap = max(0, min(10, max_seq_len - 1))

        for label, text in data_iter:
            if self.allowed_labels and label not in self.allowed_labels:
                continue

            token_ids = self.vocab(basic_tokenize(text))
            for chunk in chunk_token_ids(token_ids, chunk_size=self.max_seq_len, overlap=sequence_overlap):
                padded_ids = pad_token_ids(chunk, vocab, seq_len=self.max_seq_len)
                self.data.append((torch.tensor(padded_ids, dtype=torch.long), int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_tensor, label = self.data[idx]
        return text_tensor, text_tensor, label


def load_text_dataset(train=True, vocab=None, allowed_labels=None):
    """Load the local bundled text corpus and return tokenized sequences."""
    if vocab is None:
        vocab = build_or_load_vocab()

    dataset = TextDataset(get_corpus_documents(train=train), vocab, allowed_labels=allowed_labels)
    return dataset, vocab

def get_random_batch_text(dataset, batch_size=32):
    """Returns a random batch of tokenized text sequences"""
    indices = torch.randperm(len(dataset))[:batch_size]
    inputs = torch.stack([dataset[i][0] for i in indices])
    targets = torch.stack([dataset[i][1] for i in indices])
    return inputs, targets


def get_random_sample_text(dataset):
    index = torch.randint(0, len(dataset), (1,)).item()
    return dataset[index]


def encode_text(text, vocab):
    normalized = normalize_whitespace(text)
    return vocab(basic_tokenize(normalized)) if normalized else []


def chunk_text(text, vocab, chunk_size=None, overlap=None, tensor_seq_len=None):
    token_ids = encode_text(text, vocab)
    tensor_seq_len = tensor_seq_len or MAX_SEQ_LEN
    chunks = chunk_token_ids(token_ids, chunk_size=chunk_size, overlap=overlap)
    return [pad_token_ids(chunk, vocab, seq_len=tensor_seq_len) for chunk in chunks]


def text_to_chunk_tensors(text, vocab, chunk_size=None, overlap=None, tensor_seq_len=None):
    chunk_size = chunk_size or TEXT_CHUNK_SIZE or MAX_SEQ_LEN
    tensor_seq_len = tensor_seq_len or MAX_SEQ_LEN
    chunk_ids = chunk_text(text, vocab, chunk_size=chunk_size, overlap=overlap, tensor_seq_len=tensor_seq_len)
    tensors = [torch.tensor(chunk, dtype=torch.long) for chunk in chunk_ids]
    return tensors, chunk_ids


def decode_tokens(token_ids, vocab):
    """Converts a list of token IDs back to a string."""
    itos = vocab.get_itos()
    words = []
    special_tokens = {vocab["<pad>"], vocab["<bos>"], vocab["<eos>"]}
    for idx in token_ids:
        if idx in special_tokens:
            continue
        if 0 <= idx < len(itos):
            words.append(itos[idx])
    return normalize_whitespace(" ".join(words))


def stitch_text_chunks(chunk_texts):
    cleaned_chunks = [normalize_whitespace(chunk) for chunk in chunk_texts if normalize_whitespace(chunk)]
    return normalize_whitespace(" ".join(cleaned_chunks))


def mask_chunk_tokens(chunk_tensor, vocab, strategy="truncate", mask_ratio=0.5):
    if strategy == "random":
        return mask_text_random(chunk_tensor, vocab, mask_ratio=mask_ratio)
    keep_ratio = max(0.0, 1.0 - mask_ratio)
    return truncate_text(chunk_tensor, vocab, keep_ratio=keep_ratio)


def build_masked_document(text, vocab, strategy="truncate", mask_ratio=0.5, chunk_size=None, overlap=None):
    tensors, _ = text_to_chunk_tensors(text, vocab, chunk_size=chunk_size, overlap=overlap)
    masked_tensors = [mask_chunk_tokens(tensor, vocab, strategy=strategy, mask_ratio=mask_ratio) for tensor in tensors]
    masked_text = stitch_text_chunks([decode_tokens(tensor.tolist(), vocab) for tensor in masked_tensors])
    return masked_tensors, masked_text


def get_label_subset(node_id):
    if "noniid" in node_id:
        return NONIID_LABELS
    return None

# --- Text Corruption/Masking Functions ---

def mask_text_random(text_tensor, vocab, mask_ratio=0.5):
    """
    Masks random words in the sequence.
    Replaces tokens with <unk> or <pad> to simulate loss.
    """
    mask = (torch.rand_like(text_tensor.float()) > mask_ratio)
    masked = text_tensor.clone()
    masked[~mask] = vocab["<pad>"] 
    return masked

def truncate_text(text_tensor, vocab, keep_ratio=0.5):
    """
    Truncates the sentence, keeping only the first X%.
    Simulates sending only part of the message.
    """
    masked = text_tensor.clone()
    seq_len = masked.shape[-1]
    cut = int(seq_len * keep_ratio)
    if masked.dim() == 2:
        masked[:, cut:] = vocab["<pad>"]
    else:
        masked[cut:] = vocab["<pad>"]
    return masked

# --- NLP Metrics ---

def compute_accuracy(original, reconstructed_logits):
    """
    Calculates Token-level Accuracy.
    """
    predictions = torch.argmax(reconstructed_logits, dim=-1)
    correct = (predictions == original).float()
    return torch.mean(correct).item()

def compute_cross_entropy(original, reconstructed_logits):
    """
    Calculates Cross Entropy Loss.
    """
    criterion = torch.nn.CrossEntropyLoss()
    logits_flat = reconstructed_logits.view(-1, reconstructed_logits.size(-1))
    target_flat = original.view(-1)
    loss = criterion(logits_flat, target_flat)
    return loss.item()
