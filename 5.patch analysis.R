library(imcRtools)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(ggpubr)
library(ggbeeswarm)
library(plyr)
library(tidyr)
library(tidyverse)
library(viridis)
library(foreach)
library(doParallel)
set.seed(100)


output.dir<-'./Output'
if(! dir.exists(output.dir)) {dir.create(output.dir)}

patch_cells <- 'B'
expend <- 1

create_dir <- function(dir) {
  if (!dir.exists(dir)) {dir.create(dir, recursive = TRUE)}
}

output.dir1 <- paste(output.dir,'5_patch analysis', sep = '/')
create_dir(output.dir1)


spe <- readRDS("./Output//1_spe.info.rds")
spe <- buildSpatialGraph(spe, img_id = "roi_id", type = "knn", k = 20)
spe <- aggregateNeighbors(spe, colPairName = "knn_interaction_graph", aggregate_by = "metadata", count_by = "celltype")

spe <- patchDetection(spe,
                      patch_cells = (spe$celltype %in% patch_cells),
                      img_id = "roi_id",
                      expand_by = expend,
                      min_patch_size = 30,
                      colPairName = "knn_interaction_graph")
patch_size <- patchSize(spe, "patch_id")
saveRDS(spe, paste(output.dir1, "1_spe.rds", sep = '/'))
saveRDS(patch_size, paste(output.dir1, "1_patch_size.rds", sep = '/'))



output.dir1.2 <- paste(output.dir1 , '1.2_patch', sep = '/')
create_dir(output.dir1.2)



spatial.data <- spatial.data <- data.frame(colData(spe), spatialCoords(spe)) %>%
  select(c("roi_id", "CellID", "patch_id", "Pos_X", "Pos_Y"))
splits <- split(spatial.data, spatial.data$roi)
roi_ids <- names(splits)


col1 <- ggthemes_data$tableau$`color-palettes`$regular$`Tableau 20`$value
col2 <- ggthemes_data$tableau$`color-palettes`$regular$`Miller Stone`$value
cols_clst_base <- c(col1, rev(col2))

g <- colPair(spe, "knn_interaction_graph")
coords <- spatialCoords(spe)
edge_df <- as.data.frame(g)
edge_df$distance <- sqrt(
  (coords[edge_df$from, 1] - coords[edge_df$to, 1])^2 +
    (coords[edge_df$from, 2] - coords[edge_df$to, 2])^2
)
avg_distance_df <- edge_df %>% select(-to) %>%
  group_by(from) %>%
  dplyr::summarise_all(mean)



detectCores <- detectCores() - 5
num_cores <- if (detectCores > 2) detectCores else 2  
cl <- makeCluster(num_cores)
registerDoParallel(cl)

foreach(roi = roi_ids, .packages = c('dplyr', 'ggplot2', 'ggthemes')) %dopar% {
  plotdata <- splits[[roi]]
  plotdata$patch_id[is.na(plotdata$patch_id)] <- "undefined"
  unique_patch_ids <- unique(plotdata$patch_id)
  cols_clst <- cols_clst_base[1:length(unique_patch_ids)]
  names(cols_clst) <- unique_patch_ids
  cols_clst["undefined"] <- 'grey'

  p <- ggplot(plotdata) +
    geom_point(aes(x = Pos_X, y = Pos_Y, color = patch_id, fill = patch_id), size = 1) +
    scale_fill_manual(values = cols_clst) +
    scale_color_manual(values = cols_clst) +
    guides(color = guide_legend(override.aes = list(size = 2)),
           fill = guide_legend(override.aes = list(size = 2), ncol = 2)) +
    theme(
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      axis.title = element_blank(),
      legend.title = element_blank(),
      legend.text = element_text(size = 8),
      legend.key = element_rect(fill = "transparent"),
      panel.background = element_blank(),
      plot.title = element_text(vjust = 0.5, hjust = 0.5)
    ) +
    labs(title = roi)
  ggsave(file.path(output.dir1.2, paste0(roi, '.png')), p, height = 3, width = 4.5, dpi = 720)
  ggsave(file.path(output.dir1.2, paste0(roi, '.pdf')), p, height = 3, width = 4.5, dpi = 720)
}
stopCluster(cl)


output.dir1.3 <- paste(output.dir1, "1.3_patch size", sep = '/')
create_dir(output.dir1.3)

groups <- c("group1", "group2", "group3", "group4")
labels <- list(
  c("Tumoral", "Peritumoral"),
  c("G1", "G2"),
  c("CS1", "CS2"),
  c("Survival Short", "Survival Long")
)

group.stat.list <- setNames(lapply(labels, function(label) {
  list(stat = 't.test', stat.label = label)
}), groups)

ttest_colors <- c(
  "Peritumoral" = "#e4796b",
  "Tumoral" = "#dd2d22",
  "G2" = "#e79d5b",
  "G1" = "#67157d",
  "CS2" = "#7dfb50",
  "CS1" = "#4d9a32",
  "Survival Long" = "#df3598",
  "Survival Short" = "#480e31"
)

patch.metadata <- colData(spe) %>% as_tibble() %>% filter(!is.na(patch_id)) %>%
  select(patch_id, contains('group')) %>% unique()
patch_cell_counts <- table(spe$patch_id) %>% as.data.frame()
colnames(patch_cell_counts) <- c('patch_id', 'cell_count')

patch_data <- patch_size %>%  as_tibble() %>% select(patch_id, size) %>%
  left_join(patch.metadata, by = "patch_id") %>%
  left_join(patch_cell_counts, by = "patch_id")

group.num <- colnames(colData(spe))[grep('group', colnames(colData(spe)))]

for (group_column in group.num) {
  temp <- patch_data %>% select(patch_id, size, cell_count, group = !!sym(group_column)) %>%
    filter(group != '')

  p <- ggplot(temp, aes(x = factor(group), y = log10(size))) +
    geom_boxplot(aes(color = factor(group)), outlier.alpha = 0, alpha = 0, notch = FALSE) +
    ggbeeswarm::geom_quasirandom(aes(color = factor(group), fill = factor(group)), shape = 21,
                                 size = 1.5, dodge.width = 0.75, alpha = 1, show.legend = TRUE) +
    geom_signif(comparisons = list(group.stat.list[[group_column]][['stat.label']]), test = 't.test', vjust = 1.5,
                textsize = 3, step_increase = 0.5) +
    theme_classic() +
    labs(y = 'Log10(size)', x = 'Group') +
    guides(color = guide_legend(ncol = 1, byrow = FALSE)) +
    scale_color_manual(name = "", values = ttest_colors) +
    scale_fill_manual(name = "", values = ttest_colors) +
    theme(strip.background = element_blank(),
          strip.text = element_text(angle = 0, size = 10),
          legend.title = element_blank(),
          axis.text = element_text(size = 10))
  p
  ggsave(file.path(output.dir1.3, paste0(group_column, "_patch_ttest.png")), p, height = 3, width = 3.5)
  ggsave(file.path(output.dir1.3, paste0(group_column, "_patch_ttest.pdf")), p, height = 3, width = 3.5)
}

output.dir1.4 <- paste(output.dir1, "1.4_cell freq", sep = '/')
create_dir(output.dir1.4)

ttest.data <- colData(spe) %>% as.tibble() %>% filter(!is.na(patch_id)) %>% 
  group_by(patch_id) %>% select(celltype) %>%
  table() %>% normalize() %>% as.data.frame()  %>% left_join(patch.metadata)


calculate_t_test <- function(data, celltypes, group_name, exclude_celltype) {
  group_labels <- group.stat.list[[group_name]][['stat.label']]
  
  p.out <- lapply(celltypes, function(ct) {
    temp <- data %>% filter(celltype == ct)
    g0 <- temp %>% filter(group == group_labels[1]) %>% pull(Freq)
    g1 <- temp %>% filter(group == group_labels[2]) %>% pull(Freq)
    
    p.val <- if (ct %in% exclude_celltype) 1 else t.test(g0, g1)$p.value
    label <- if (mean(g0) > mean(g1)) group_labels[1] else group_labels[2]
    
    data.frame(celltype = ct, group = group_name, group.label = label, p.val = p.val)
  })
  
  do.call(rbind, p.out)
}

celltypes <- levels(spe$celltype)
ttest_result <- data.frame()

for (g in group.num) {
  print(g)
  stat.dir <- file.path(output.dir1.4, g)
  create_dir(stat.dir)
  
  sub.ttest.data <- ttest.data %>%
    select(patch_id:Freq, group = !!sym(g)) %>%
    filter(group != "")

  filter_data <- sub.ttest.data %>%
    pivot_wider(names_from = celltype, values_from = Freq) %>%
    select(-patch_id) %>%
    group_by(group) %>%
    summarise_all(mean) %>%
    select(-group)
  
  select_cols <- colnames(filter_data)[apply(filter_data >= 0.05, 2, any)]
  # exclude_cols <- setdiff(colnames(filter_data),select_cols)
  exclude_cols <- NULL
  
  p.out <- calculate_t_test(sub.ttest.data, celltypes, g,exclude_cols)
  ttest_result <- bind_rows(ttest_result, p.out)
  # do.call(rbind, lapply(p.out, function(x) do.call(rbind, x)))
}


ttest_result <- ttest_result %>%
  mutate(
    p.val = as.numeric(p.val),
    Sig = case_when(
      p.val < 0.001 ~ "p<0.001",
      p.val < 0.05 ~ "p<0.05",
      TRUE ~ "NotSignificant" )
  )
ttest_result$Sig = factor(ttest_result$Sig, levels = c("NotSignificant", "p<0.05", "p<0.001"))
ttest_result$celltype <- factor(ttest_result$celltype,levels = levels(spe$celltype))


group.num <- colnames(ttest.data)[grep('group', colnames(ttest.data))]
for (g in group.num) {
  stat.dir <- paste(output.dir1.4, g, sep = '/')
  create_dir(stat.dir)
  
  sub.ttest.data <- ttest.data %>% select(patch_id:Freq, group = g) %>% filter(group!='')
  
  for (ct in unique(ttest.data$celltype)) {
    print(ct)
    ttest.data_i <- sub.ttest.data %>% filter(celltype == ct)
    
    p <- ggplot(ttest.data_i, aes(
      x = factor(group),
      y = Freq,
      color = group,
      fill = group
    )) +
      # geom_boxplot(aes(color = factor(group)), 
      #              outliers = FALSE, 
      #              staplewidth = 0.5,
      #              width = 0.75, 
      #              alpha = 0.75) +
      # ggbeeswarm::geom_quasirandom(aes(, 
      #                                  fill = factor(group)),
      #                              shape = 21, size=1.5, 
      #                              dodge.width = .75, alpha=1, 
      #                              show.legend = F) +
      geom_jitter(width = 0.15, shape=21, size=3, color="#252a32") +
      geom_boxplot(staplewidth = 0.5, width=0.75,
                   outliers = FALSE, alpha=0.75)+
      geom_signif(comparisons = list(group.stat.list[[g]][["stat.label"]]),
                  test = 't.test', textsize = 3, step_increase = 0.1) +
      theme_classic()+
      labs(x = '', y = 'Proportion of total cells (%)' ) +
      scale_color_manual(values = ttest_colors)+
      scale_fill_manual(values = ttest_colors)+
      theme(
        axis.text.x = element_text(size = 8, angle = 0, hjust = 0.5, vjust = 0.5),
        axis.text.y = element_text(size = 8, angle = 0, hjust = 0.5, vjust = 0.5),
        # legend.title = element_blank(),
        legend.position = "none",
        strip.background = element_blank(),
        strip.text = element_text(size = 10, angle = 0),
        plot.title = element_text(size = 8, hjust = 0.5, vjust = 0.5))
    p
    ggsave(paste(stat.dir,  paste0('2.5_freq_cluster_',ct,'.png'), sep = '/'), p, height = 2.1, width=2.2,dpi = 720)
    ggsave(paste(stat.dir,  paste0('2.5_freq_cluster_',ct,'.pdf'), sep = '/'), p, height = 2.1, width=2.2,dpi = 720)
  }
  
}

### ################## 1.5 function marker ##################
output.dir1.5 <-  paste(output.dir1, '1.5_t(function marker)', sep = '/')
create_dir(output.dir1.5)

idx <- which(!is.na(spe$patch_id))
spe.sub <- spe[, idx]
dim(spe.sub)

celltypes <- levels(spe.sub$celltype)
group.num <- colnames(ttest.data)[grep('group', colnames(ttest.data))]


function_marker <- c("CD69", "CD103", "GATA3", "IL6", "IL1b", "TNFa")
mat <- t(assay(spe.sub, "exprs"))[, rownames(rowData(spe.sub)) %in% function_marker] %>% 
  as_tibble()

g <- "group1"
stat.dir <- file.path(output.dir1.5, g)
data <- cbind(mat, roi_id = spe.sub$roi_id, group = as.factor(colData(spe.sub)[[g]]))

data <- data %>% filter(group != "")
data$group <- droplevels(data$group)

cols <- colnames(data)[1:(ncol(data) - 2)]
plot_data <- data %>%
  group_by(roi_id, group) %>%
  summarise_all(median) %>%
  pivot_longer(cols = cols, names_to = "Marker", values_to = "Expression")

plot_data$Marker <- factor(plot_data$Marker,levels = c("CD69", "CD103", "GATA3", "CD45RO", "IL6", "IL1b", "TNFa"))
plot_data$group <- factor(plot_data$group,levels = c("Tumoral","Peritumoral"))


library(introdataviz)
# tmp_colors <- c(
#   "Peritumoral" = "#df3598",
#   "Tumoral" = "#480e31"
# )
p <- ggplot(plot_data, aes(x = Marker, y = Expression, fill = group)) +
  introdataviz::geom_split_violin(trim = FALSE, alpha = 0.5) +  
  geom_boxplot(width = 0.2, alpha = 0.5, outlier.shape = NA, show.legend = FALSE,fatten = NULL) +  
  stat_summary(fun.data = mean_se, geom = "pointrange",
               position = position_dodge(.175), show.legend = FALSE) + 
  
  stat_compare_means(method = "t.test", label = "p.signif", size = 6) +  
  
  # scale_fill_manual(values = tmp_colors) +
  scale_fill_brewer(palette = "Dark2", name = "Language group") +
  theme_bw() +
  theme(
    axis.title = element_text(size = 12),
    axis.text.x = element_text(size = 12, angle = 0, hjust = 1, vjust = 1),
    axis.text.y = element_text(size = 12),
    legend.title = element_blank(),
    legend.text = element_text(size = 12),
    strip.background = element_blank()
  ) +
  scale_y_continuous(name = "Reaction time (ms)",
                     breaks = seq(-0.1, 0.6, 0.1), 
                     limits = c(-0.1, 0.5)) +
  ylab("Expression") + xlab("Marker")

p
ggsave(file.path(stat.dir, "plot_data.pdf"), plot = p, height = 3, width = 12)

