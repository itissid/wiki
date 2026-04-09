// A chunk of wiki content with metadata
class WikiChunk {
    text string
    page string
    heading string
    chunk_index int
}

// Result from hybrid retrieval
class RetrievalResult {
    chunk WikiChunk
    bm25_rank int?
    vector_rank int?
    rrf_score float
}

// The full answer returned by the MCP tool
class WikiAnswer {
    answer string
    repo string
    question string
    chunks_used int
    retrieval_time_ms float
    generation_time_ms float
}
