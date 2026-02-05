#!/usr/bin/env Rscript

# Finetuning_Residuals.R -> Generates a Residual Plot from HILIC Finetuning

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(patchwork)
  library(tidymodels)
})

# Paths (EDIT HERE)
input_csv <- "~/Desktop/ToGit/Finetuning 1.csv"
output_plot <- "res.svg"
sig <- 145.874


dat <- read.csv(input_csv)
data <- dat %>% 
  mutate(
    PreRes = Pretraining.Prediction - True.RT,
    PosRes = Posttraining.Prediction - True.RT,
    abs_resid = abs(PosRes - PreRes)
  )

res <- ggplot(data) +
  geom_hline(yintercept = 0, linetype = "dashed", size = 1, color = "firebrick2", alpha = 0.8) +
  geom_rect(aes(xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = -2*sig),
            fill = "grey70", alpha = 0.01) +
  geom_rect(aes(xmin = -Inf, xmax = Inf, ymin = 2*sig, ymax = Inf),
            fill = "grey70", alpha = 0.01) +
  geom_point(aes(x = True.RT, y = PreRes), color = "black", size = 1.5) +
  geom_segment(aes(x = True.RT, xend = True.RT,
                   y = PreRes, yend = PosRes,
                   color = abs(PosRes) < abs(PreRes)),
               arrow = arrow(length = unit(0.2, "cm"))) +
  scale_color_manual(values = c(`TRUE` = "springgreen4", `FALSE` = "black"),
                     guide = "none") +
  theme_bw(base_size = 16) +
  theme(panel.grid.major = element_line(color = "grey", size = 0.4),
        panel.grid.minor = element_line(color = "grey", size = 0.4),
        axis.title.x = element_text(face = "bold"),
        axis.title.y = element_text(face = "bold")) +
  labs(x = "Validated Observed RT (s)", y = "Predicted RT Residual (s)")


ggsave(file = output_plot, plot = res)
cat("Mean PreRes:", mean(data$PreRes), "\n")
cat("Mean PosRes:", mean(data$PosRes), "\n")