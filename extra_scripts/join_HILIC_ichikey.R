#!/usr/bin/env Rscript

# Join_Hilic_Ichikey.R
library(dplyr)

left <- read.csv("LD.csv") %>%
  rename("inchikey" = "inchi_key", "min" = "rt_min", "max" = "rt_max") %>%
  filter(if_all(everything(), ~ !is.na(.))) %>%
  group_by(inchikey, compound_name) %>%
  summarise(
    min = min(min, na.rm = TRUE),
    max = max(max, na.rm = TRUE),
    .groups = "drop"
  )

right <- read.csv("RD.csv") %>%
  distinct(inchikey, .keep_all = TRUE)

joined <- left_join(left, right, by = "inchikey")
joined$RT <- round((joined$min + joined$max) / 2, 2)

write.csv(joined, "HILIC_KNOWNS.csv", row.names = FALSE)