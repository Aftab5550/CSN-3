library(ggplot2)
library(gridExtra)

dir.create("figures", showWarnings = FALSE)

sum_df <- read.csv("results/bounds_summary.csv", stringsAsFactors = FALSE, row.names = NULL)
mc_df  <- read.csv("results/table_2_significance.csv", stringsAsFactors = FALSE, row.names = NULL)

raw_lines <- readLines("results/bounds_detail.csv")
raw_lines <- raw_lines[nzchar(trimws(raw_lines))]
split_lines <- strsplit(raw_lines[-1], ",")

det_df <- data.frame(
  language = sapply(split_lines, `[`, 1),
  N        = as.numeric(sapply(split_lines, `[`, 2)),
  E        = as.numeric(sapply(split_lines, `[`, 3)),
  x_obs    = as.numeric(sapply(split_lines, `[`, 4)),
  bound    = sapply(split_lines, `[`, 5),
  order    = as.numeric(sapply(split_lines, `[`, 6)),
  p_bound  = as.numeric(sapply(split_lines, `[`, 7)),
  value    = suppressWarnings(as.numeric(sapply(split_lines, `[`, 8))),
  stringsAsFactors = FALSE
)

lang_order <- sum_df$language[order(sum_df$p_new_best, decreasing = TRUE)]

top_df <- subset(det_df,
                 (bound == "cantelli_m2") |
                   (bound == "old_markov") |
                   (bound == "moment_markov" & order %in% c(2, 3, 4))
)

top_df$method <- factor(
  ifelse(top_df$bound == "cantelli_m2", "Single-Pair Cantelli (m2)",
         ifelse(top_df$bound == "old_markov", "1st-Order Markov (s = 1)",
                ifelse(top_df$bound == "moment_markov" & top_df$order == 2, "Non-Edge Markov (s = 2)",
                       ifelse(top_df$bound == "moment_markov" & top_df$order == 3, "Non-Edge Markov (s = 3)",
                              "Non-Edge Markov (s = 4)")))),
  levels = c(
    "Single-Pair Cantelli (m2)",
    "1st-Order Markov (s = 1)",
    "Non-Edge Markov (s = 2)",
    "Non-Edge Markov (s = 3)",
    "Non-Edge Markov (s = 4)"
  )
)
top_df$language <- factor(top_df$language, levels = lang_order)

p1_top <- ggplot(top_df, aes(x = language, y = p_bound, color = method, shape = method, group = method)) +
  geom_hline(yintercept = 0.05, linetype = "dashed", color = "firebrick", linewidth = 0.65) +
  geom_line(linewidth = 0.75, alpha = 0.9) +
  geom_point(size = 2.8) +
  annotate("text", x = 1.2, y = 0.085, label = "Significance threshold alpha = 0.05 (MC resolution 1/T = 0.04)", color = "firebrick", size = 3.2, hjust = 0, fontface = "bold") +
  scale_y_continuous(limits = c(0.0, 1.0), breaks = seq(0.0, 1.0, by = 0.2)) +
  scale_color_manual(values = c(
    "Single-Pair Cantelli (m2)" = "#8c564b",
    "1st-Order Markov (s = 1)"  = "#d62728",
    "Non-Edge Markov (s = 2)"   = "#ff7f0e",
    "Non-Edge Markov (s = 3)"   = "#9467bd",
    "Non-Edge Markov (s = 4)"   = "#1f77b4"
  )) +
  labs(
    title = "A. Single-Pair Moment Bounds (Linear Scale: p in [0, 1])",
    subtitle = "Higher moments (s = 1 to 4) tighten the bound down to ~0.24 but stall above alpha = 0.05 due to Jensen's inequality",
    x = NULL,
    y = "Upper Bound on p-value",
    color = NULL,
    shape = NULL
  ) +
  theme_bw(base_size = 10.5) +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    legend.position = "top",
    plot.title = element_text(face = "bold", size = 11),
    plot.margin = margin(t = 6, r = 10, b = 2, l = 10)
  ) +
  guides(color = guide_legend(nrow = 1), shape = guide_legend(nrow = 1))

bot_df <- subset(det_df, bound == "path_chebyshev")
bot_df$language <- factor(bot_df$language, levels = lang_order)
bot_df$cutoff_label <- paste0("L=", bot_df$order)

p1_bot <- ggplot(bot_df, aes(x = language, y = log10(p_bound), group = 1)) +
  geom_line(color = "#2ca02c", linewidth = 0.9) +
  geom_point(aes(shape = cutoff_label), color = "#2ca02c", fill = "#2ca02c", size = 3.2) +
  geom_text(aes(label = sprintf("%.1e", p_bound)), vjust = -1.0, size = 2.9, color = "#1b611b", fontface = "bold") +
  scale_y_continuous(
    limits = c(-8.1, -5.0),
    breaks = seq(-8, -5, by = 1),
    labels = function(x) paste0("10^", x)
  ) +
  scale_shape_manual(values = c("L=3" = 17, "L=4" = 15)) +
  labs(
    title = "B. Global Path-Intersection Chebyshev Bound (Log10 Scale: p in [10^-8, 10^-5])",
    subtitle = "Accounting for O(1/M^2) negative covariance of edge-disjoint paths in G(N, E) proves p << 10^-5 across all languages",
    x = "Language (ordered by path-Chebyshev bound)",
    y = "Chebyshev Bound [log10]",
    shape = "Optimal Path Cutoff"
  ) +
  theme_bw(base_size = 10.5) +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1, face = "bold", size = 9.5),
    legend.position = "bottom",
    legend.title = element_text(face = "bold"),
    plot.title = element_text(face = "bold", size = 11),
    plot.margin = margin(t = 2, r = 10, b = 6, l = 10)
  )

p1_combined <- arrangeGrob(p1_top, p1_bot, ncol = 1, heights = c(1.08, 1.0))
ggsave("figures/bounds_hierarchy.png", p1_combined, width = 10.0, height = 6.8, dpi = 300)

merged_df <- merge(sum_df, mc_df[, c("language", "p_switching")], by = "language")
merged_df$language_ord <- factor(merged_df$language, levels = merged_df$language[order(merged_df$x_obs)])
merged_df$gap_label <- sprintf("+%.3f", merged_df$x_obs - merged_df$U_new)

p2_left <- ggplot(merged_df, aes(y = language_ord)) +
  geom_segment(aes(x = U_new, xend = x_obs, y = language_ord, yend = language_ord), color = "gray60", linewidth = 1.2) +
  geom_point(aes(x = U_new, color = "ER Upper Bound E[C_NH] <= U_new"), size = 3.2) +
  geom_point(aes(x = x_obs, color = "Observed Network C_obs"), size = 3.2) +
  geom_text(aes(x = (U_new + x_obs) / 2, label = gap_label), vjust = -0.6, size = 2.8, color = "gray25", fontface = "bold") +
  scale_x_continuous(limits = c(0.165, 0.355), breaks = seq(0.18, 0.34, by = 0.04)) +
  scale_color_manual(values = c("ER Upper Bound E[C_NH] <= U_new" = "#1f77b4", "Observed Network C_obs" = "#d62728")) +
  labs(
    title = "A. Empirical Closeness vs. G(N, E) Bound",
    subtitle = "Hub-driven closeness gap (Delta C) causing p_binom = 0",
    x = "Mean Closeness Centrality (C)",
    y = NULL,
    color = NULL
  ) +
  theme_bw(base_size = 11) +
  theme(
    legend.position = "bottom",
    axis.text.y = element_text(face = "bold"),
    plot.title = element_text(face = "bold", size = 11.5)
  )

merged_df$nudge_y <- 0.0042
merged_df$nudge_x <- 0.0
merged_df$nudge_y[merged_df$language == "Spanish"] <- -0.0052
merged_df$nudge_x[merged_df$language == "Spanish"] <- -1.1
merged_df$nudge_y[merged_df$language == "Italian"] <- -0.0052
merged_df$nudge_x[merged_df$language == "Italian"] <- 0.8
merged_df$nudge_y[merged_df$language == "Turkish"] <- -0.0055
merged_df$nudge_y[merged_df$language == "Polish"]  <- 0.0052
merged_df$nudge_x[merged_df$language == "Polish"]  <- 0.5
merged_df$nudge_y[merged_df$language == "Czech"]   <- 0.0062

p2_right <- ggplot(merged_df, aes(x = mean_k, y = x_obs)) +
  geom_smooth(method = "lm", formula = y ~ log(x), se = TRUE, fullrange = TRUE, color = "steelblue", fill = "lightblue", alpha = 0.25, linetype = "dashed") +
  geom_point(aes(size = N / 1000), color = "#d62728", alpha = 0.85) +
  geom_text(aes(x = mean_k + nudge_x, y = x_obs + nudge_y, label = language), size = 3.2, fontface = "bold") +
  scale_x_continuous(limits = c(4.0, 23.2), breaks = seq(5, 20, by = 5)) +
  scale_y_continuous(limits = c(0.252, 0.355)) +
  labs(
    title = "B. Closeness vs. Mean Degree <k>",
    subtitle = "Morphological typology and vocabulary size (N)",
    x = "Mean Degree <k> = 2E / N",
    y = "Observed Closeness Centrality (C_obs)",
    size = "Vertices N (x10^3)"
  ) +
  theme_bw(base_size = 11) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 11.5)
  )

p2_combined <- arrangeGrob(p2_left, p2_right, ncol = 2)
ggsave("figures/closeness_structural_analysis.png", p2_combined, width = 11.8, height = 5.4, dpi = 300)