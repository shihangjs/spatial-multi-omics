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
library(SpatialExperiment)
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
library(glue)
source("./Code/IMC3.3/diagDivCellHeatmap.R")
set.seed(100)

# NOTE: Execution order must be "Lymphocyte" "Myeloid" "Epithelial"
cluster_label <- "Epithelial"

rds.sub <- paste('./Output', paste0("2.1_Cluster.", cluster_label, "_subclustering"), '1_spe.sub.rds', sep = '/')
spe.sub <- readRDS(rds.sub)
spe.sub %>% colData() %>% as_tibble() %>% select(pg_cluster) %>% table() %>% as.data.frame()

anno.file <- paste('./Output', paste0("2.1_Cluster.", cluster_label, "_subclustering"), 'annotation.csv', sep = '/')

output.dir <- './Output'
if(! dir.exists(output.dir)) { dir.create(output.dir) }

options(width = 180)
## 1. Annotation
output.dir1 <- paste(output.dir, paste0('2.1_CellType.', cluster_label, '_celltype_annotation'), sep = '/')
if(! dir.exists(output.dir1)) { dir.create(output.dir1) }

celltype <- read.csv(anno.file, encoding = 'UTF-8', row.names = 1)
colnames(celltype)[1:3] <- c('pg_cluster', 'new_pg_cluster', 'celltype')
celltype$pg_cluster <- as.factor(celltype$pg_cluster)

if (cluster_label == "Lymphocyte") {
  rds <- "./Output/2_CellType/2_spe.rds"
  celltype_merge <- c("Epithelial", "Mesenchymal")
  
} else if (cluster_label == "Epithelial") {
  rds <- "./Output/2_CellType_annotation_correction/2_spe.rds"
  celltype_merge <- c("Mesenchymal")
  
} else if (cluster_label == "Myeloid") {
  rds <- "./Output/2_CellType_annotation_correction/2_spe.rds"
  celltype_merge <- NULL
  celltype$celltype <- gsub("MF", "MΦ", celltype$celltype)
}

spe <- readRDS(rds)
spe.sub$celltype <- mapvalues(spe.sub$pg_cluster, from = celltype$new_pg_cluster, to = celltype$celltype)

if (length(celltype_merge) > 0) {
  
  idx <- which(spe.sub$celltype %in% celltype_merge)
  length(idx)
  temp <- spe.sub[, idx]
  
  metadata.sub <- colData(temp) %>% as_tibble() %>% select(roi_id, CellID, celltype)
  metadata.all <- colData(spe) %>% as_tibble() %>% select(roi_id, CellID, celltype)
  metadata.new <- merge(metadata.all, metadata.sub, by = c("roi_id", "CellID"), all.x = TRUE)
  metadata.new$celltype <- ifelse(is.na(metadata.new$celltype.y), as.character(metadata.new$celltype.x), as.character(metadata.new$celltype.y))
  
  spe$celltype <- metadata.new$celltype
  cols_to_clear <- c("pg_cluster", "new_pg_cluster", "celltype_umap", "pg_cluster_umap")
  colData(spe) <- colData(spe)[, !colnames(colData(spe)) %in% cols_to_clear]
  colnames(colData(spe))
  
  if ("Lineage-" %in% unique(spe$celltype)) {
    idx <- which(spe$celltype %in% c("Lineage-"))
    length(idx)
    spe$celltype[idx] <- "Mesenchymal"
  }
  spe %>% saveRDS("./Output/2_CellType_annotation_correction/2_spe.rds")
}

spe.sub$celltype <- factor(spe.sub$celltype, levels = unique(celltype$celltype))
cols_to_clear <- c("new_pg_cluster", "celltype_umap", "pg_cluster_umap")
colData(spe.sub) <- colData(spe.sub)[, !colnames(colData(spe.sub)) %in% cols_to_clear]

if ("rm" %in% unique(spe.sub$celltype)) {
  idx <- which(spe.sub$celltype %in% c("rm", celltype_merge))
} else {
  idx <- which(spe.sub$celltype %in% c(celltype_merge))
}
length(idx)

if (length(idx) > 0) {
  spe.sub.rm <- spe.sub[, -idx]
  spe.sub.rm$celltype <- droplevels(spe.sub.rm$celltype)
} else {
  spe.sub.rm <- spe.sub
}

if ("rm" %in% unique(celltype$celltype)) {
  idx <- which(celltype$celltype %in% c("rm", celltype_merge))
} else {
  idx <- which(celltype$celltype %in% c(celltype_merge))
}
if (length(idx) > 0) {
  celltype.rm <- celltype[-idx, ]
} else {
  celltype.rm <- celltype
}

celltype_number <- unique(celltype.rm$celltype) %>% length
celltype_umap <- paste0(c(1:celltype_number), ':', unique(celltype.rm$celltype))
spe.sub.rm$celltype_umap <- mapvalues(spe.sub.rm$celltype, from = unique(celltype.rm$celltype), to = celltype_umap)
spe.sub.rm$celltype_umap <- factor(spe.sub.rm$celltype_umap, levels = celltype_umap[!grepl('rm', celltype_umap)])

saveRDS(spe.sub.rm, paste(output.dir1, "2_spe.sub.rm.rds", sep = '/'))

##  2. Annotation (all) 
output.dir2 <- paste(output.dir, '2_CellType_merged_cell_subcluster_annotation', sep = '/')
if(! dir.exists(output.dir2)) { dir.create(output.dir2) }

if (cluster_label == "Lymphocyte") {
  spe <- readRDS("./Output/2_CellType_annotation_correction/2_spe.rds")
} else {
  spe <- readRDS(paste(output.dir2, "2_spe.rds", sep = '/'))
}

metadata.sub <- colData(spe.sub) %>% as_tibble() %>% select(roi_id, CellID, celltype)
metadata.all <- colData(spe) %>% as_tibble() %>% select(roi_id, CellID, celltype)
metadata.new <- merge(metadata.all, metadata.sub, by = c("roi_id", "CellID"), all.x = TRUE)
metadata.new$celltype <- ifelse(is.na(metadata.new$celltype.y),
                                as.character(metadata.new$celltype.x),
                                as.character(metadata.new$celltype.y))
spe$celltype <- metadata.new$celltype
cols_to_clear <- c("pg_cluster", "new_pg_cluster", "celltype_umap", "pg_cluster_umap")
colData(spe) <- colData(spe)[, !colnames(colData(spe)) %in% cols_to_clear]
colnames(colData(spe))

# NOTE: "rm" exists in spe.sub
spe <- spe[, !spe$celltype == 'rm']
spe$celltype %>% unique()

saveRDS(spe, paste(output.dir2, "2_spe.rds", sep = '/'))


