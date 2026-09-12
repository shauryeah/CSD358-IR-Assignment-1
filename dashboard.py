import re
from collections import Counter
from math import log10, sqrt
from nltk.stem import PorterStemmer
import streamlit as st


CORPUS_FILE = "corpus_100.txt"

# Porter Stemmer is used to reduce related words to a common stem.
stemmer = PorterStemmer()


# ---------------- PART A ----------------

def read_corpus(filename):
    # Read the complete corpus file at once.
    with open(filename, "r", encoding="utf-8") as file:
        data = file.read()

    documents = []

    # Each document in the corpus is enclosed inside <DOC>...</DOC>.
    blocks = re.findall(r"<DOC>(.*?)</DOC>", data, re.DOTALL)

    for block in blocks:
        # Extract the basic information stored for each document.
        docid = re.search(r"<DOCID>(.*?)</DOCID>", block).group(1)
        category = re.search(r"<CATEGORY>(.*?)</CATEGORY>", block).group(1)
        title = re.search(r"<TITLE>(.*?)</TITLE>", block).group(1)
        text = re.search(r"<TEXT>(.*?)</TEXT>", block).group(1)

        documents.append({
            "docid": docid,
            "category": category,
            "title": title,
            "text": text
        })

    return documents


def preprocess(text):
    # Case folding makes words like "Cotton" and "cotton" the same term.
    text = text.lower()

    # Keep alphabetic words and allow apostrophes inside words.
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text)

    # Remove apostrophes before continuing with the tokens.
    tokens = [token.replace("'", "") for token in tokens]

    # Apply Porter stemming so related word forms share the same stem.
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens


def build_inverted_index(documents):
    index = {}

    for document in documents:

        # Title and description are both treated as searchable content.
        text = document["title"] + " " + document["text"]

        tokens = preprocess(text)

        # Count how many times each term occurs in this document.
        term_frequency = Counter(tokens)

        for term, tf in term_frequency.items():

            # Create a new entry the first time we see a term.
            if term not in index:
                index[term] = {
                    "df": 0,
                    "postings": []
                }

            # Since the term occurs in this document, increase its df.
            index[term]["df"] += 1

            # Store the document ID and term frequency in the postings list.
            index[term]["postings"].append(
                (document["docid"], tf)
            )

    return index


def save_inverted_index(index, filename="inverted_index.txt"):

    # Save the complete inverted index so it can be submitted separately.
    with open(filename, "w", encoding="utf-8") as file:

        # Sorting makes the output easier to read and inspect.
        for term in sorted(index):
            df = index[term]["df"]
            postings = index[term]["postings"]

            posting_text = ", ".join(
                f"{docid}:{tf}" for docid, tf in postings
            )

            file.write(
                f"{term}\tDF={df}\tPostings=[{posting_text}]\n"
            )


# ---------------- PART B ----------------

def build_document_vectors(index):
    document_vectors = {}
    document_lengths = {}

    for term, data in index.items():

        for docid, tf in data["postings"]:

            # Document weight follows the lnc scheme:
            # wd,t = 1 + log10(tf)
            weight = 1 + log10(tf)

            if docid not in document_vectors:
                document_vectors[docid] = {}

            document_vectors[docid][term] = weight

    # Calculate the length of each document vector for cosine normalization.
    for docid, vector in document_vectors.items():

        length = sqrt(
            sum(weight ** 2 for weight in vector.values())
        )

        document_lengths[docid] = length

    return document_vectors, document_lengths


def search(query, index, document_vectors, document_lengths, top_k=10):

    # The query goes through the same preprocessing as the documents.
    query_terms = preprocess(query)

    # Count how often each query term occurs.
    query_tf = Counter(query_terms)

    query_vector = {}

    for term, tf in query_tf.items():

        # Terms that are not present in the corpus are simply ignored.
        if term in index:

            df = index[term]["df"]

            # Query weight follows the ltc scheme:
            # wq,t = (1 + log10(tf)) * log10(N/df)
            weight = (
                (1 + log10(tf))
                * log10(100 / df)
            )

            query_vector[term] = weight

    # Find the length of the query vector for cosine normalization.
    query_length = sqrt(
        sum(weight ** 2 for weight in query_vector.values())
    )

    # If there is no usable query term, there is nothing to rank.
    if query_length == 0:
        return []

    scores = {}

    # Compare the query vector against every document vector.
    for docid, doc_vector in document_vectors.items():

        dot_product = 0

        for term, query_weight in query_vector.items():

            if term in doc_vector:

                # Add the contribution of this shared term to the dot product.
                dot_product += (
                    query_weight * doc_vector[term]
                )

        if dot_product > 0:

            # Normalize the dot product to get cosine similarity.
            cosine_similarity = (
                dot_product
                / (query_length * document_lengths[docid])
            )

            scores[docid] = cosine_similarity

    # Rank by highest cosine score.
    # If scores are equal, use the smaller document ID first.
    results = sorted(
        scores.items(),
        key=lambda x: (-x[1], int(x[0][1:]))
    )

    return results[:top_k]


# ---------------- PART C ----------------

def build_positional_index(documents):
    positional_index = {}

    for document in documents:

        # Use the same searchable text and preprocessing as the inverted index.
        text = document["title"] + " " + document["text"]

        tokens = preprocess(text)

        term_positions = {}

        # Store every position at which each term occurs in this document.
        for position, term in enumerate(tokens):

            if term not in term_positions:
                term_positions[term] = []

            term_positions[term].append(position)

        for term, positions in term_positions.items():

            # Create the term's entry the first time it appears.
            if term not in positional_index:
                positional_index[term] = {
                    "df": 0,
                    "postings": []
                }

            positional_index[term]["df"] += 1

            # Store document ID, term frequency, and all positions.
            positional_index[term]["postings"].append(
                (
                    document["docid"],
                    len(positions),
                    positions
                )
            )

    return positional_index


def save_positional_index(
    positional_index,
    filename="positional_index.txt"
):

    # Save the positional index so the complete structure can be submitted.
    with open(filename, "w", encoding="utf-8") as file:

        for term in sorted(positional_index):

            df = positional_index[term]["df"]
            postings = positional_index[term]["postings"]

            posting_text = ", ".join(
                f"{docid}:{tf}:{positions}"
                for docid, tf, positions in postings
            )

            file.write(
                f"{term}\tDF={df}\tPostings=[{posting_text}]\n"
            )


def phrase_search(query, positional_index):

    # Preprocess the phrase using the same rules as the corpus.
    query_terms = preprocess(query)

    if not query_terms:
        return []

    # Start by finding documents containing the first query term.
    first_term = query_terms[0]

    if first_term not in positional_index:
        return []

    results = []

    first_postings = positional_index[first_term]["postings"]

    for docid, tf, positions in first_postings:

        # These positions are possible starting points for the phrase.
        matching_positions = positions

        for i in range(1, len(query_terms)):

            term = query_terms[i]

            # If any query term is missing, this document cannot match.
            if term not in positional_index:
                matching_positions = []
                break

            term_postings = positional_index[term]["postings"]

            position_dict = {}

            # Find the positions of this term in the current document.
            for posting_docid, posting_tf, posting_positions in term_postings:

                if posting_docid == docid:
                    position_dict = posting_positions
                    break

            if not position_dict:
                matching_positions = []
                break

            new_matches = []

            # For an exact phrase, the next term must be exactly one
            # position after the previous term.
            for position in matching_positions:

                if position + 1 in position_dict:
                    new_matches.append(position + 1)

            matching_positions = new_matches

            if not matching_positions:
                break

        if matching_positions:
            results.append(
                (docid, matching_positions)
            )

    return results


def proximity_search(query, positional_index):

    # Expected format: term1 WITHIN/k term2
    match = re.match(
        r"(.+?)\s+WITHIN/(\d+)\s+(.+)",
        query,
        re.IGNORECASE
    )

    if not match:
        return []

    term1 = preprocess(match.group(1))
    k = int(match.group(2))
    term2 = preprocess(match.group(3))

    # This implementation expects one term on each side of WITHIN/k.
    if len(term1) != 1 or len(term2) != 1:
        return []

    term1 = term1[0]
    term2 = term2[0]

    # Both terms must exist in the positional index.
    if term1 not in positional_index or term2 not in positional_index:
        return []

    postings1 = positional_index[term1]["postings"]
    postings2 = positional_index[term2]["postings"]

    # Convert the postings into dictionaries so positions can be
    # accessed directly using the document ID.
    positions1 = {
        docid: positions
        for docid, tf, positions in postings1
    }

    positions2 = {
        docid: positions
        for docid, tf, positions in postings2
    }

    results = []

    # Only documents containing both terms can satisfy the query.
    common_docs = set(positions1.keys()) & set(positions2.keys())

    for docid in common_docs:

        matching_pairs = []

        for p1 in positions1[docid]:

            for p2 in positions2[docid]:

                # The first term must come before the second term,
                # and the two positions must be within k tokens.
                if 0 < p2 - p1 <= k:
                    matching_pairs.append((p1, p2))

        if matching_pairs:
            results.append(
                (docid, matching_pairs)
            )

    # Display documents in increasing document ID order.
    results.sort(
        key=lambda x: int(x[0][1:])
    )

    return results


# ---------------- PART D ----------------

def create_document_lookup(documents):

    lookup = {}

    for document in documents:

        # Keep the information needed to display search results.
        lookup[document["docid"]] = {
            "title": document["title"],
            "category": document["category"]
        }

    return lookup


# ---------------- STREAMLIT INTERFACE ----------------

# Basic Streamlit page configuration.
st.set_page_config(
    page_title="Clothing Information Retrieval",
    page_icon="🔎",
    layout="wide"
)


# Read the supplied clothing corpus.
documents = read_corpus(CORPUS_FILE)

# Build the normal inverted index used for ranked retrieval.
index = build_inverted_index(documents)

# Build document vectors and their lengths for cosine similarity.
document_vectors, document_lengths = build_document_vectors(index)

# Build the positional index used for phrase and proximity searches.
positional_index = build_positional_index(documents)

# Generate the index files automatically when the application runs.
save_inverted_index(index)
save_positional_index(positional_index)

# Create a quick lookup for titles and categories shown in the results.
document_lookup = create_document_lookup(documents)


# ---------------- SIDEBAR ----------------

st.sidebar.title("Information Retrieval")

st.sidebar.markdown("### Corpus Information")

st.sidebar.write(
    f"**Documents:** {len(documents)}"
)

st.sidebar.write(
    f"**Unique terms:** {len(index)}"
)

st.sidebar.write(
    f"**Positional terms:** {len(positional_index)}"
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Preprocessing")

st.sidebar.write("• Lowercase")
st.sidebar.write("• Punctuation removal")
st.sidebar.write("• Porter stemming")
st.sidebar.write("• Stopwords retained")

st.sidebar.markdown("---")

st.sidebar.markdown("### Index Files")

st.sidebar.write("✓ inverted_index.txt")
st.sidebar.write("✓ positional_index.txt")


# ---------------- MAIN PAGE ----------------

st.title("Clothing Information Retrieval System")

st.write(
    "Search the clothing corpus using Vector Space Model "
    "or positional search."
)


# ---------------- SEARCH MODE ----------------

# Let the user choose which type of retrieval they want to perform.
search_mode = st.radio(
    "Select search mode:",
    [
        "Free Text (VSM)",
        "Phrase Search",
        "Proximity Search"
    ],
    horizontal=True
)


# ---------------- FREE TEXT SEARCH ----------------

if search_mode == "Free Text (VSM)":

    st.subheader("Free-Text Ranked Retrieval")

    query = st.text_input(
        "Enter your clothing query:",
        placeholder="Example: cotton shirt"
    )

    st.caption(
        "Uses lnc.ltc cosine similarity and returns the top 10 results."
    )

    if query:

        # Run the query through the VSM ranking function.
        results = search(
            query,
            index,
            document_vectors,
            document_lengths,
            top_k=10
        )

        if results:

            result_data = []

            # Convert the search results into a table for Streamlit.
            for rank, (docid, score) in enumerate(results, start=1):

                result_data.append({
                    "Rank": rank,
                    "DocID": docid,
                    "Title": document_lookup[docid]["title"],
                    "Category": document_lookup[docid]["category"],
                    "Cosine Score": round(score, 4)
                })

            st.subheader("Top 10 Results")

            st.dataframe(
                result_data,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "No results found. The query may contain terms "
                "that are not present in the corpus or have zero IDF."
            )


# ---------------- PHRASE SEARCH ----------------

elif search_mode == "Phrase Search":

    st.subheader("Phrase Search")

    query = st.text_input(
        "Enter a phrase:",
        placeholder="Example: cotton shirt"
    )

    st.caption(
        "Finds documents where the query terms occur consecutively "
        "using the positional index."
    )

    if query:

        # Search for the exact phrase using stored term positions.
        results = phrase_search(
            query,
            positional_index
        )

        if results:

            # Only the first 10 matching documents are displayed.
            results = results[:10]

            result_data = []

            for rank, (docid, positions) in enumerate(
                results,
                start=1
            ):

                result_data.append({
                    "Rank": rank,
                    "DocID": docid,
                    "Title": document_lookup[docid]["title"],
                    "Category": document_lookup[docid]["category"],
                    "Matching Ending Positions": str(positions)
                })

            st.subheader("Matching Documents")

            st.dataframe(
                result_data,
                use_container_width=True,
                hide_index=True
            )

            st.success(
                "Positional index used successfully. "
                "The positions shown are the ending positions "
                "of matching phrase occurrences."
            )

        else:

            st.warning(
                "No exact phrase matches found."
            )


# ---------------- PROXIMITY SEARCH ----------------

else:

    st.subheader("Proximity Search")

    query = st.text_input(
        "Enter a proximity query:",
        placeholder="Example: cotton WITHIN/3 shirt"
    )

    st.caption(
        "Format: term1 WITHIN/k term2"
    )

    if query:

        # Check that the user has entered the expected WITHIN/k format.
        if not re.match(
            r"(.+?)\s+WITHIN/(\d+)\s+(.+)",
            query,
            re.IGNORECASE
        ):

            st.error(
                "Invalid format. Use: term1 WITHIN/k term2"
            )

        else:

            # Run the ordered proximity search using the positional index.
            results = proximity_search(
                query,
                positional_index
            )

            if results:

                # Display at most the first 10 matching documents.
                results = results[:10]

                result_data = []

                for rank, (docid, positions) in enumerate(
                    results,
                    start=1
                ):

                    result_data.append({
                        "Rank": rank,
                        "DocID": docid,
                        "Title": document_lookup[docid]["title"],
                        "Category": document_lookup[docid]["category"],
                        "Matching Position Pairs": str(positions)
                    })

                st.subheader("Matching Documents")

                st.dataframe(
                    result_data,
                    use_container_width=True,
                    hide_index=True
                )

                st.success(
                    "Positional index used successfully. "
                    "Each position pair represents the positions "
                    "of the two query terms."
                )

            else:

                st.warning(
                    "No documents satisfy the proximity condition."
                )


# ---------------- FOOTER ----------------

st.markdown("---")

st.caption(
    "CSD358 Information Retrieval | Clothing Corpus | "
    "VSM + Positional Retrieval"
)
