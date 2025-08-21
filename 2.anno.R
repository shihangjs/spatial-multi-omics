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
set.seed(100)

rds <- './Output/1_Cluster/1_spe.rds'
spe <- readRDS(rds)

anno.file <- './Input/anno.csv'

celltype <- read.csv(anno.file)
print(celltype)

spe$celltype <- mapvalues(spe$pg_cluster, from = celltype$new_pg_cluster, to = celltype$celltype)
spe <- spe[, !spe$celltype == 'rm']
spe$celltype <- factor(spe$celltype, levels = unique(celltype$celltype)[which(unique(celltype$celltype) != 'rm')])

spe$new_pg_cluster <- mapvalues(spe$pg_cluster, from = celltype$pg_cluster, to = celltype$new_pg_cluster)
celltype_number <- unique(celltype$celltype)[which(unique(celltype$celltype) != 'rm')] %>% length()
print(celltype_number)

celltype_umap <- paste0(c(1:celltype_number), ':', unique(celltype$celltype)[which(unique(celltype$celltype) !=                                                                                 'rm')])
spe$celltype_umap <- mapvalues(spe$celltype, from = unique(celltype$celltype)[which(unique(celltype$celltype) != 'rm')],to = celltype_umap)
spe$celltype_umap <- factor(spe$celltype_umap, levels = celltype_umap[!grepl('rm', celltype_umap)])
