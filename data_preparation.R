library(igraph)
library(udpipe)

dir.create("data", showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)

languages <- c("Arabic", "Basque", "Catalan", "Chinese", "Czech", "English",
               "Greek", "Hungarian", "Italian", "Turkish", "Spanish", "Polish")

# Build the undirected syntactic network for a language and return its summary stats
build_undirected_network <- function(treebanks_path, language_name) {
  all_dirs  <- list.dirs(treebanks_path, recursive = FALSE)
  lang_dirs <- all_dirs[grepl(paste0("^UD_", language_name, "-"), basename(all_dirs))]
  cat(language_name, ":", basename(lang_dirs), "\n")
  
  conllu_files <- list.files(lang_dirs, pattern = "\\.conllu$", full.names = TRUE)
  
  # Read all CoNLL-U files for the language
  all_rows <- do.call(rbind, lapply(conllu_files, function(f) {
    rows <- udpipe_read_conllu(f)
    rows$file_id <- basename(f)
    rows
  }))
  
  # Match each token to its head strictly within the same sentence
  all_rows$sentence_uid <- paste(all_rows$file_id, all_rows$doc_id, all_rows$sentence_id, sep = "_")
  head_lookup <- data.frame(
    sentence_uid  = all_rows$sentence_uid,
    head_token_id = all_rows$token_id,
    head_token    = all_rows$token,
    stringsAsFactors = FALSE
  )
  edges_data <- merge(all_rows, head_lookup, by = c("sentence_uid", "head_token_id"))
  
  # Normalize words to lowercase and replace internal spaces in multi-part tokens
  from_words <- gsub("[[:punct:]]", "", gsub("\\s+", "_", tolower(trimws(edges_data$head_token))))
  to_words   <- gsub("[[:punct:]]", "", gsub("\\s+", "_", tolower(trimws(edges_data$token))))
  
  # Filter out missing words (_) and self-loops (u == v)
  valid <- from_words != "_" & to_words != "_" &
    from_words != ""  & to_words != ""  &
    from_words != to_words
  edges_list <- data.frame(from = from_words[valid], to = to_words[valid], stringsAsFactors = FALSE)
  
  # Build undirected network without multiedges or loops
  g <- graph_from_data_frame(edges_list, directed = FALSE)
  g <- simplify(g, remove.multiple = TRUE, remove.loops = TRUE)
  
  N <- as.numeric(vcount(g))
  E <- as.numeric(ecount(g))
  
  # Export network: header "N E" followed by the edge list
  out_file <- paste0("data/", language_name, "_dependency_network.txt")
  con <- file(out_file, open = "wt", encoding = "UTF-8")
  writeLines(paste(vcount(g), ecount(g)), con)
  edge_df <- as_data_frame(g, what = "edges")
  writeLines(paste(edge_df$from, edge_df$to), con)
  close(con)
  
  # Compute Table 1 metrics
  data.frame(
    language    = language_name,
    N           = vcount(g),
    E           = ecount(g),
    mean_degree = (2 * E) / N,
    density     = (2 * E) / (N * (N - 1)),
    stringsAsFactors = FALSE
  )
}

# Process all 12 languages and save Table 1
table_1 <- do.call(rbind, lapply(languages, function(lang) {
  build_undirected_network("ud-treebanks-v2.18", lang)
}))

print(table_1)
write.csv(table_1, "results/table_1_summary.csv", row.names = FALSE)
