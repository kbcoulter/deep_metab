#!/usr/bin/env Rscript

# HILIC_Knowns_Comp.R -> Generates Predicted vs Observed RT plot for HILIC

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
})

# PATHS (EDIT HERE)
knowns_csv <- "HILIC_KNOWNS.csv"
preds_csv <- "40631942.csv"
ambig_csv <- "40703382.csv"
output_plot <- "Pred_vs_Obs.svg"

knowns <- read.csv(knowns_csv)
preds <- read.csv(preds_csv)
ambig <- read.csv(ambig_csv)


knowns <- knowns[knowns$smiles != "" & !is.na(knowns$smiles), ]
ambig <- ambig[ambig$SMILES != "" & !is.na(ambig$SMILES), ]

if (length(knowns$smiles) != length(preds$SMILES)) {
  message("Length mismatch between predictions and inputs in Known HILIC... Halting.")
  stopifnot(length(knowns$smiles) != length(preds$SMILES))}

knowns <- knowns %>% select(-RT)
knowns$rt <- preds$True.RT
knowns$preds <- preds$Predicted.RT
message("Predictions paired back to input. Continuing...")

message("Generating Figure: Predicted vs Observed RT...")
line_df <- data.frame(x = c(0, 1300), y = c(0, 1300), Type = "Perfect Prediction")
legend_order <- factor(c("Known", "Unknown Mean", "Perfect Prediction"), 
                       levels = c("Known", "Unknown Mean", "Perfect Prediction"))

plot <- ggplot() +
  geom_point(data = knowns, aes(x = rt, y = preds, color = factor("Known", levels = levels(legend_order))), alpha = 0.7) +
  geom_point(data = ambig, aes(x = True.RT, y = Predicted.RT, color = factor("Unknown Mean", levels = levels(legend_order))), alpha = 0.3) +
  geom_line(data = line_df, aes(x = x, y = y, linetype = Type, color = Type), linetype = "dashed", size = 1.2) +
  theme_bw() +
  xlab("OBSERVED RT (s)") +
  ylab("PREDICTED RT (s)") +
  theme(
    axis.title = element_text(size = 14),
    axis.text  = element_text(size = 12)) +
  scale_color_manual(
    name = "HILIC Dataset Source",
    values = c("Known" = "red", "Unknown Mean" = "cornflowerblue", "Perfect Prediction" = "black")) +
  scale_x_continuous(limits = c(0, 1300), breaks = c(0, 325, 650, 975, 1300), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1300), breaks = c(0, 325, 650, 975, 1300), expand = c(0, 0))

print(plot)

MAE_known <- abs(knowns$rt - knowns$preds)
MAE_ambig <- abs(ambig$Predicted.RT - ambig$True.RT)
test <- wilcox.test(MAE_known, MAE_ambig, exact = FALSE)

ggsave(filename = output_plot, plot = plot, height = 6, width = 8)