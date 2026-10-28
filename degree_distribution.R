# Plot degree distributions for the dependency networks in data/.

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg)) {
  script_path <- normalizePath(sub("^--file=", "", script_arg[1]), winslash = "/")
  setwd(dirname(script_path))
}

data_dir <- "data"
output_dir <- file.path("results", "degree_distribution")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

network_files <- sort(list.files(
  data_dir,
  pattern = "_dependency_network\\.txt$",
  full.names = TRUE
))
if (!length(network_files)) {
  stop("No *_dependency_network.txt files found in ", data_dir)
}

read_degrees <- function(path) {
  language <- sub("_dependency_network\\.txt$", "", basename(path))
  header <- scan(path, what = integer(), nlines = 1, quiet = TRUE)
  if (length(header) != 2L) stop("Invalid network header in ", path)

  edges <- read.table(path, skip = 1, header = FALSE, sep = "",
                      quote = "", comment.char = "", stringsAsFactors = FALSE)
  if (nrow(edges) && ncol(edges) < 2L) stop("Invalid edge list in ", path)

  # Count both endpoints; the input networks are undirected and have no loops.
  degree_table <- table(c(as.character(edges[[1]]), as.character(edges[[2]])))
  degrees <- as.integer(degree_table)
  if (length(degrees) != header[1]) {
    warning(language, ": header reports ", header[1], " vertices, but ",
            length(degrees), " occur in the edge list")
  }
  list(language = language, degrees = degrees)
}

networks <- lapply(network_files, read_degrees)

plot_distribution <- function(network, file) {
  degrees <- network$degrees
  counts <- table(degrees)
  x <- as.numeric(names(counts))
  y <- as.numeric(counts)
  keep <- x > 0 & y > 0 # zero-degree vertices cannot appear on log axes

  if (!any(keep)) stop("No positive degree values for ", network$language)
  plot(x[keep], y[keep], log = "xy", pch = 19, cex = 0.75,
       xlab = "Degree (k)", ylab = "Number of vertices with degree k",
       main = paste(network$language, "degree distribution"))
}

# One PNG per language for easy inspection or inclusion in a report.
for (network in networks) {
  png(file.path(output_dir, paste0(network$language, "_degree_distribution.png")),
      width = 1200, height = 900, res = 150)
  plot_distribution(network)
  dev.off()
}

# A single multi-page PDF contains the same per-language plots.
pdf(file.path(output_dir, "degree_distributions_by_language.pdf"),
    width = 8, height = 6)
for (network in networks) plot_distribution(network)
dev.off()

message("Wrote degree distribution plots to ", normalizePath(output_dir))
