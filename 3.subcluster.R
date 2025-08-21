library(RColorBrewer)
library(ggplot2)
library(ComplexHeatmap)
library(ggthemes)
library(ggpubr)
library(plyr)
library(ggbeeswarm)
library(argparser)
library(scales)
library(SingleCellExperiment)
library(imcRtools)
library(diffcyt)
library(cowplot)
library(dittoSeq)
library(viridis)
library(bluster)
library(BiocParallel)
library(tidyverse)
library(scater)
library(harmony)
library(FastPG)
library(glue)
set.seed(100)


marker.file <- './Input/marker.csv'
panel <- read.csv(marker.file,check.names = FALSE, fileEncoding = "GBK")

cluster.all <- c("Myeloid")
rds <- './Output/2_CellType/2_spe.rds'
spe <- readRDS(rds)
idx <- which(spe$celltype %in% cluster.all)

spe.sub <- spe[, idx]
dim(spe.sub)

rowData(spe.sub)$use_channel <- rownames(spe.sub) %in% (panel %>% filter(myeloid_function == '1' | myeloid_lineage == "1") %>% pull(marker))
# rowData(spe.sub)$use_channel <- rownames(spe.sub) %in% (panel %>% filter(TME_function == '1' | TME_lineage == "1") %>% pull(marker))
# rowData(spe.sub)$use_channel <- rownames(spe.sub) %in% (panel %>% filter(lymphoid_function == '1' | lymphoid_lineage == "1") %>% pull(marker))

mat <- t(assay(spe.sub, "exprs")) [,rowData(spe.sub)$use_channel]

svd_res <- svd(mat)
variance_explained <- (svd_res$d^2)/sum(svd_res$d^2)
npc <- which(cumsum(variance_explained) >= 0.95)[1]
if (npc < 2) {npc <- 2}

if(length(unique(spe.sub$batch_id)) < 2) {
  # PCA: after without correction
  harmony_emb <- prcomp(t(mat)) ### batch_id
  reducedDim(spe.sub, "harmony") <- harmony_emb$rotation[, 1:npc]
  spe.sub <- runUMAP(spe.sub, dimred = "harmony", name = "UMAP_harmony")
  
} else {
  # HARMONY: after with correction
  harmony_emb <- HarmonyMatrix(mat, as.factor(spe.sub$batch_id), do_pca = T, npcs = npc) ### batch_id
  reducedDim(spe.sub, "harmony") <- harmony_emb
  spe.sub <- runUMAP(spe.sub, dimred = "harmony", name = "UMAP_harmony")
}


expr.mat <- reducedDim(spe.sub, "harmony")
cluster_PhenoGraph <- FastPG::fastCluster(as.matrix(expr.mat),k = 50, num_threads = 100)
spe.sub$pg_cluster <- as.factor(cluster_PhenoGraph$communities) 

saveRDS(spe.sub, paste(output.dir1, "1_spe.sub.rds", sep = '/'))
