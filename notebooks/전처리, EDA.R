setwd("/Volumes/Disk/Projects/StrokeCare/ GSE16561/data") 

library(mgcv)
library(nlme)
library(KernSmooth)

# 데이터 불러오기
raw_data <- read.delim("GSE16561_RAW.txt", header = TRUE, sep = "\t", check.names = FALSE)
bgx <- read.delim("GPL6883_HumanRef-8_V3_0_R0_11282963_A.bgx",
                  header = TRUE,
                  sep = "\t",
                  skip = 8,  # 텍스트 편집기로 확인 => "[Probes]" 다음 줄부터 읽기
                  comment.char = "#",
                  quote = "",
                  row.names = NULL)

colnames(raw_data)[1] # ID_REF
colnames(bgx)

# 데이터 병합
merged <- merge(raw_data, bgx, by.x = "ID_REF", by.y = "Probe_Id")
dim(merged)
head(merged[, c("ID_REF", "Symbol")])

# Symbol 기준 같은 유전자에 속하는 probe 평균값으로 요약
library(dplyr)

# 분석에 사용할 샘플만 선택(ID_REF, Symbol, 발현값들만)
colnames(merged)[1:20]
sapply(merged, class)
# 샘플 열만 명시적으로 추출(ID_REF, Symbol 제외 + numeric만)
expr_by_gene <- merged %>%
  select(Symbol, matches("_Stroke|_Control")) %>%
  group_by(Symbol) %>%
  summarise(across(everything(), mean, na.rm = TRUE))
# Symbol을 rownames로 지정
expr_mat <- as.data.frame(expr_by_gene)
rownames(expr_mat) <- expr_mat$Symbol
expr_mat <- expr_mat[, -1] # Symbol 열 제거

dim(expr_mat)
head(expr_mat[, 1:5])

# 결측치
sum(is.na(expr_mat)) # 0

# log2 변환 여부 확인
summary(as.numeric(as.matrix(expr_mat)))
boxplot(expr_mat, las = 2, cex.axis = 0.5, outline = FALSE,
        main = "Boxplot of Expression Values", ylab = "Intensity")
hist(as.numeric(as.matrix(expr_mat)), breaks = 100, 
     main = "Expression Value Histogram", xlab = "Value")
# log2 + quantile normalization 안되어있음을 확인할 수 있었음

# 전처리
# log2 변환(pseudocount 추가로 0 방지)
log_expr <- log2(expr_mat + 1)

# quantile normalization
library(preprocessCore)
norm_expr <- normalize.quantiles(as.matrix(log_expr))
rownames(norm_expr) <- rownames(log_expr)
colnames(norm_expr) <- colnames(log_expr)
expr_mat <- as.data.frame(norm_expr)

# 전처리 확인
summary(as.numeric(as.matrix(expr_mat)))
boxplot(expr_mat, las = 2, cex.axis = 0.5, outline = FALSE,
        main = "Normalized Expression", ylab = "log2 Intensity")
hist(as.numeric(as.matrix(expr_mat)), breaks = 100,
     main = "Histogram After Log2 + Quantile Normalization")

# 군집 정보 만들기
sample_names <- colnames(expr_mat)
group <- ifelse(grepl("Control", sample_names), "Control", "Stroke")
pheno <- data.frame(sample = sample_names, group = group)
head(pheno)

# EDA
# 샘플별 분포
# boxplot
boxplot(expr_mat, las = 2, outline = FALSE,
        cex.axis = 0.5, main = "Boxplot per Sample", ylab = "log2 intensity")

# density plot
matplot(density(as.numeric(expr_mat[,1]))$x,
        sapply(expr_mat, function(x) density(x)$y),
        type = "l", lty = 1, col = rgb(0, 0, 0, 0.1),
        xlab = "log2 Intensity", ylab = "Density", main = "Density plot per Sample")

# IQR 계산
iqr_vals <- apply(expr_mat, 1, IQR)
# filtering ( IQR >= 0.5 )
expr_filtered <- expr_mat[iqr_vals >= 0.07, ]
cat("남은 유전자 수: ", nrow(expr_filtered), "/n")
# 18470

# PCA
pca <- prcomp(t(expr_filtered), scale. = TRUE)
# 컬러 지정
group_colors <- ifelse(pheno$group == "Stroke", "red", "blue")
# PCA plot
plot(pca$x[,1], pca$x[,2],
     col = group_colors, pch = 19, cex = 1.2,
     xlab = "PC1", ylab = "PC2",
     main = "PCA of Samples")
legend("topright", legend = c("Stroke", "Control"),
       col = c("red", "blue"), pch = 19)
# 확실한 분리 경향 보임

# limma로 DEG
# 1. Design matrix 생성
group <- factor(pheno$group)
design <- model.matrix(~0 + group)
colnames(design) <- levels(group)

# 2. limma 분석 진행
library(limma)
# voom 안 해도 됨 (이미 정규화, log2 완료된 microarray)
fit <- lmFit(expr_filtered, design)

# 비교: Stroke vs Control
contrast <- makeContrasts(Stroke - Control, levels = design)
fit2 <- contrasts.fit(fit, contrast)
fit2 <- eBayes(fit2)

# 결과 테이블 추출
deg_results <- topTable(fit2, adjust = "fdr", number = Inf)

# summary
summary(decideTests(fit2))

# 시각화
# volcano plot
with(deg_results, {
  plot(logFC, -log10(adj.P.Val),
       pch = 20, main = "Volcano Plot",
       xlab = "log2 Fold Change", ylab = "-log10 FDR")
  
  points(logFC[adj.P.Val < 0.05 & abs(logFC) > 1],
         -log10(adj.P.Val)[adj.P.Val < 0.05 & abs(logFC) > 1],
         col = "red", pch = 20)
  abline(h = -log10(0.05), col = "blue", lty = 2)
  abline(v = c(-1, 1), col = "blue", lty = 2)
})

# heatmap
library(pheatmap)
# 상위 DEG 50개 추출
top_deg_genes <- rownames(deg_results)[1:50]
# 해당 유전자에 대한 발현값
heatmap_data <- expr_filtered[top_deg_genes, ]
# 그룹 색상 매핑
annotation <- data.frame(Group = pheno$group)
rownames(annotation) <- pheno$sample

pheatmap(heatmap_data, scale = "row", 
         annotation_col = annotation, 
         show_rownames = TRUE, show_colnames = FALSE,
         color = colorRampPalette(c("navy", "white", "firebrick3"))(100),
         main = "Top 50 Differentially Expressed Genes")

# 저장
write.csv(deg_results, "DEG_results.csv")

