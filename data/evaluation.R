rm(list=ls())

import.data <- function(folder) {
  data <- data.frame()
  for (file in list.files(folder)) {
    data <- rbind(data, read.csv(paste(folder, file, sep=""), col.names=c("id", "seqNo", "label", "prob")))
  }; rm(file)
  return(data)
}
data <- import.data("resnet_slowfast/eval/")

frame_tp_1 <- nrow(data[data$label==1 & data$prob>=0.5,])
frame_fn_1 <- nrow(data[data$label==1 & data$prob<0.5,])
frame_rec_1 <- frame_tp_1/(frame_tp_1+frame_fn_1); rm(frame_tp_1); rm(frame_fn_1)
frame_tp_0 <- nrow(data[data$label==0 & data$prob<0.5,])
frame_fn_0 <- nrow(data[data$label==0 & data$prob>=0.5,])
frame_rec_0 <- frame_tp_0/(frame_tp_0+frame_fn_0); rm(frame_tp_0); rm(frame_fn_0)
frame_uar <- (frame_rec_1 + frame_rec_0)/2

# Collapse labels
collapse <- function(labels) {
  idx_p <- which(labels==1)
  p_d <- diff(idx_p)
  p_start <- which(p_d>1)+1
  p_end <- p_start-1
  p_start <- c(1, p_start)
  p_end <- c(p_end, sum(labels))
  idx <- data.frame(start=idx_p[p_start], end=idx_p[p_end])
  idx$mid <- apply(idx, 1, function(x) floor((x[1]+x[2])/2))
  return(sapply(seq(1, length(labels)), function(x) if(x%in%idx$mid) return(1) else return(0)))
}
data$label_coll <- collapse(data$label)

# Local maximum search in thresholded probs (Kyritsis, 2019)
maxsearch <- function(probs, threshold, mindistance) {
  probs <- sapply(probs, function(x) if(x>threshold) return(x) else return(0))
  idx_p <- which(probs>0)
  p_d <- diff(idx_p) - 1
  p_start <- which(p_d>0)+1
  p_end <- p_start-1
  p_start <- c(1, p_start)
  p_end <- c(p_end, sum(sapply(probs, function(x) if(x>threshold) return(1) else return(0))))
  idx <- data.frame(start=idx_p[p_start], end=idx_p[p_end])
  idx$max <- apply(idx, 1, function(x) x[1]-1+which.max(probs[seq(x[1], x[2])]))
  max_diff <- c(mindistance, diff(idx$max))
  carry <- 0
  rem_i <- c()
  for (i in seq(1, nrow(idx))) {
    if (max_diff[i] + carry < mindistance) {
      rem_i <- c(rem_i, i)
      carry <- carry + max_diff[i]
    } else {
      carry <- 0
    }
  }
  idx <- idx[-rem_i,]
  return(sapply(seq(1, length(probs)), function(x) if(x%in%idx$max) return(1) else return(0)))
}

# Calculating split indices
split.idx <- function(label) {
  idx_t <- which(label==1)
  t_d <- diff(idx_t)-1
  t_start <- which(t_d>0)+1
  t_end <- t_start-1
  t_start <- c(1, t_start)
  t_end <- c(t_end, sum(label))
  idx <- data.frame(start=idx_t[t_start], end=idx_t[t_end])
  return(apply(idx, 1, function(x) seq(x[[1]], x[[2]])))
}

# Calculate f1
data$max_coll <- maxsearch(data$prob, 0.987, 16)
idxs.t <- split.idx(data$label)
idxs.f <- which(data$label==0)
split.t <- lapply(idxs.t, function(x) data$max_coll[x]); rm(idxs.t)
split.f <- data$max_coll[idxs.f]; rm(idxs.f)
tp <- sum(as.numeric(lapply(split.t, function(x) if(sum(x)>0) 1 else 0)))
fn <- sum(as.numeric(lapply(split.t, function(x) if(sum(x)>0) 0 else 1)))
fp_1 <- sum(as.numeric(lapply(split.t, function(x) if(sum(x)>1) sum(x)-1 else 0)))
fp_2 <- sum(split.f); rm(split.t); rm(split.f)
prec <- tp/(tp+fp_1+fp_2)
rec <- tp/(tp+fn)
f1 <- 2*prec*rec/(prec+rec)

# Grid search
f1.grid.search <- function(data, t_start, t_end, t_step, n_f) {
  t_vals <- seq(from=t_start, to=t_end, by=t_step)
  f1_calc <- function(data, t, n_f) {
    max_coll <- maxsearch(data$prob, t, n_f)
    idxs.t <- split.idx(data$label)
    idxs.f <- which(data$label==0)
    split.t <- lapply(idxs.t, function(x) max_coll[x])
    split.f <- max_coll[idxs.f]
    tp <- sum(as.numeric(lapply(split.t, function(x) if(sum(x)>0) 1 else 0)))
    fn <- sum(as.numeric(lapply(split.t, function(x) if(sum(x)>0) 0 else 1)))
    fp <- sum(as.numeric(lapply(split.t, function(x) if(sum(x)>1) sum(x)-1 else 0)))
    fp <- fp + sum(split.f)
    prec <- tp/(tp+fp)
    rec <- tp/(tp+fn)
    f1 <- 2*prec*rec/(prec+rec)
    return(f1)
  }
  f1_vals <- sapply(t_vals, function(x) f1_calc(data, x, n_f))
  data.frame(t=t_vals, f1=f1_vals)
}

f1.results <- f1.grid.search(data, t_start=.5, t_end=.999, t_step=.001, n_f=16)

# Best thresholds computed on eval
# Settings oreba_2d_cnn_flow: threshold = 0.793; mind = 16
# Settings oreba_2d_cnn_frame: threshold = 0.957; mind = 16
# Settings oreba_3d_cnn: threshold = 0.997; mind = 16
# Settings oreba_cnn_lstm: threshold = 0.983; mind = 16
# Settings oreba_two_stream: threshold = 0.973; mind = 16
# Settings oreba_slowfast: threshold = 0.996; mind = 16
# Settings resnet_2d_cnn_flow: threshold = 0.865; mind = 16
# Settings resnet_2d_cnn_frame: threshold = 0.964; mind = 16
# Settings resnet_3d_cnn: threshold = 0.992; mind = 16
# Settings resnet_cnn_lstm: threshold = 0.996; mind = 16
# Settings resnet_two_stream: threshold = 0.997; mind = 16
# Settings resnet_slowfast: threshold = 0.987; mind = 16
