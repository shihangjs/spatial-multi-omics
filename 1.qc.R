library(RColorBrewer)
library(ggplot2)
library(ComplexHeatmap)
library(ggthemes)
library(ggpubr)
library(plyr)
library(ggbeeswarm)
library(scales)
library(cowplot)
library(viridis)
library(scater)
library(BiocParallel)
library(harmony)
library(SingleCellExperiment)
library(SpatialExperiment)
library(tidyverse)
library(imcRtools)
library(diffcyt)
library(dittoSeq)
library(bluster)
library(argparser)
library(FastPG)
library(glue)
set.seed(100)


q.nor <- function(x,n) {
  x[is.na(x)] <- 0
  q.values <- quantile(x,probs = c(100/(n+1) * (0:(n+1)))/100, na.rm = T)
  # x[x < q.values[2]] <- sample(x[x > q.values[2] & x < q.values[3]], length(x[x < q.values[2]]), replace = T )
  x[x > q.values[n+1]] <- sample(x[x > q.values[n] & x < q.values[n+1]], length(x[x > q.values[n+1]]),replace = T )
  scale(x)
}

minMax <- function(x) {(x - min(x)) / (max(x) - min(x))}
normalize <- function(x) {x / rowSums(x)}
z.score <- function(x) {(x-mean(x))/sd(x)}


platform<-'IMC'
csv.dir<-'./Input/csv'
info.file<-'./Input/info.csv'
marker.file<-'./Input/marker.csv'
output.dir<-'./Output'
if(! dir.exists(output.dir)) {dir.create(output.dir)}
cofactor<-'1'
q.nor.n=101


panel <- read.csv(marker.file,check.names = FALSE, fileEncoding = "GBK")

files <- list.files(csv.dir , pattern = '.csv$', full = TRUE)
csv.list <- lapply(files, function(x) {
  temp <- read.csv(x)
  col.temp <- colnames(temp)[grep('X[0-9]+', colnames(temp))]
  col.channel <- str_split(col.temp , pattern = '_', simplify = T)[, 1] %>% str_sub(2, nchar(.))
  colnames(temp)[grep('X[0-9]+', colnames(temp))]  <- panel$marker[match(col.channel, panel$channel)]
  temp <- temp[, !is.na(colnames(temp))]
  temp
})
names(csv.list) <- files


for(x in files) {
  temp <- csv.list[[x]] %>% colnames()
  temp <- any(str_detect(temp, '[.]1$'))
  if(temp == TRUE) {stop( c(x,'colnames are uncorrect'))}
  if(is.na(temp)) {stop( c('Panel marker file is uncorrect'))}
}
cur_features <- do.call('rbind.fill', csv.list)


info <- read.csv(info.file, check.names = FALSE, fileEncoding = "GBK")
colnames(info)[1] <- 'roi_id'


marker <- panel  %>% pull(marker)
counts <- cur_features %>% select(marker) 

meta <- cur_features %>% select(roi, CellID) %>%
  mutate(roi_id = roi, .keep = 'unused') %>%
  left_join(info) 

meta$batch_id <- str_split(meta$roi, '_', simplify = T)[, 1]
meta$sample_id <- str_split(meta$roi, '_', simplify = T)[, 1]
coords <- cur_features %>% select(contains('position'))
colnames(coords) <- c("Pos_X", "Pos_Y")

spe <- SpatialExperiment(assays = list(counts = t(counts)),
                         colData = meta, 
                         sample_id = as.character(meta$sample_id),
                         image_id = as.character(meta$roi_id),
                         spatialCoords = as.matrix(coords))
colnames(spe) <- paste0(spe$roi_id, ".", spe$CellID)

rowData(spe)$use_channel <- rownames(spe) %in% (panel %>% filter(anno == '1') %>% pull(marker))

counts.tsf <- asinh(counts(spe) / as.numeric(cofactor))

counts.nor <- counts.tsf
for (i in 1:nrow(counts.nor)) {
  print(i)
  counts.nor[i, ] <- q.nor(x = counts.nor[i, ], n = q.nor.n)
  counts.nor[i, ] <- minMax(counts.nor[i, ])
}
assay(spe, "exprs") <- counts.nor

set.seed(220225)


spe <- runUMAP(spe, subset_row = rowData(spe)$use_channel, exprs_values = "exprs")
mat <- t(assay(spe, "exprs")) [,rowData(spe)$use_channel]
harmony_emb <- HarmonyMatrix(mat, as.factor(spe$batch_id), do_pca = T, npcs = npc) ### batch_id
reducedDim(spe, "harmony") <- harmony_emb
spe <- runUMAP(spe, dimred = "harmony", name = "UMAP_harmony")
spe$batch_id <- factor(spe$batch_id, levels = unique(spe$batch_id))
expr.mat <- reducedDim(spe, "harmony")
cluster_PhenoGraph <- FastPG::fastCluster(as.matrix(expr.mat),k = 30, num_threads = 100)
spe$pg_cluster <- as.factor(cluster_PhenoGraph$communities)

meta.info <- colData(spe) %>% as_tibble() %>% select(roi_id, contains('group')) %>% unique()
ttest.data <- t(assay(spe, "exprs")) %>% as.tibble() %>% mutate(roi_id = spe$roi_id) %>%
  group_by(roi_id) %>% summarise_all(median) %>% left_join(meta.info) 

celllabel <- data.frame(
  pg_cluster = levels(spe$pg_cluster),
  new_pg_cluster = rep('', length(levels(spe$pg_cluster))),
  celltype = rep('', length(levels(spe$pg_cluster)))
)
write.csv(celllabel, file = paste0(output.dir1, '/annotation.csv'))

saveRDS(spe, paste(output.dir1, "1_spe.rds", sep = '/'))
