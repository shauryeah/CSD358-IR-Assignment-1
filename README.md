# CSD358 Information Retrieval — Assignment 1

A small clothing search engine built for **CSD358 Information Retrieval Assignment-1** using the supplied 100-document clothing corpus.

## What this project implements

- Corpus parsing and text preprocessing
- Lowercase normalization and punctuation removal
- Porter stemming
- Stop-word retention for positional/phrase searching
- Inverted index with document frequency and `(docID, tf)` postings
- Ranked free-text retrieval using **lnc.ltc** weighting and cosine similarity
- Positional index storing `(docID, tf, positions)`
- Exact phrase search
- Ordered proximity search using `term1 WITHIN/k term2`
- Streamlit interface with three search modes
- Automatic generation of `inverted_index.txt` and `positional_index.txt`

## Project structure

```text
CSD358-IR-Assignment-1/
├── dashboard.py
├── corpus_100.txt
├── inverted_index.txt
├── positional_index.txt
├── requirements.txt
├── README.md
├── screenshots/
│   ├── part_e_page3_img1.png
│   ├── part_e_page3_img2.png
│   ├── part_e_page3_img3.png
│   ├── part_e_page5_img1.png
│   ├── part_e_page5_img2.png
│   ├── part_e_page5_img3.png
│   ├── part_e_page7_img1.png
│   └── part_e_page7_img2.png
└── docs/
    ├── IR_Code_Documentation.pdf
    └── Part_E_Information_Retrieval.pdf
```

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the application

```bash
streamlit run dashboard.py
```

The app reads `corpus_100.txt` from the project folder and builds the inverted and positional indexes when it starts.

## Search modes

### Free Text (VSM)
Uses lnc.ltc weighting and cosine similarity. Results are ranked by decreasing similarity, with document ID used as the tie-breaker.

Example:
```text
cotton shirt
```

### Phrase Search
Uses the positional index and requires the terms to occur consecutively and in the specified order.

Example:
```text
cotton shirt
```

### Proximity Search
Uses the positional index and requires the first term to occur before the second term within `k` token positions.

Example:
```text
cotton WITHIN/3 shirt
```

## Testing

The assignment testing includes:

- 10+ free-text queries
- 5 exact phrase queries
- 3+ proximity queries with different `k` values
- A query containing a term not present in the corpus
- Comparisons showing how positional information changes retrieval results

The detailed test results are in `docs/Part_E_Information_Retrieval.pdf`.

## Stop-word policy

Stop-word removal was not applied. Common English words were retained because the assignment also requires phrase and proximity queries, where retaining positional information for all terms can help preserve phrase structure. The same preprocessing policy is applied consistently to both documents and queries.

## Team

- Shaurya Sharma — 2410110614
- Suvigya Khare — 2410110482
