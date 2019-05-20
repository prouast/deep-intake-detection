
rm(list=ls())

import <- function(folders) {
  result <- data.frame()
  for (file in list.files(folders[1])) {
    result <- rbind(result, read.csv(paste(folders[1], "/", file, sep=""), col.names=c("id", "seq", "label", paste("prob", folders[1], sep="."))))
  }
  for (folder in folders[-1]) {
    result.2 <- data.frame()
    for (file in list.files(folder)) {
      result.2 <- rbind(result.2, read.csv(paste(folder, "/", file, sep=""), col.names=c("id", "seq", "label", paste("prob", folder, sep="."))))
    }
    result <- merge(result, result.2, by=c("id", "seq", "label"))
  }; rm(file)
  return(result)
}
write <- function(data, id, from, to, filename) {
  data <- data[data$id==id,]
  data <- data[from:to,]
  data$seq <- seq(nrow(data))
  write.csv(data, filename, row.names=FALSE)
}

# Examples for (single) motion vs appearance
#models <- c("oreba_2d_cnn_flow/eval", "oreba_2d_cnn_frame/eval", "resnet_2d_cnn_flow/eval", "resnet_2d_cnn_frame/eval")
#id <- 1010; from <- 1970; to <- 2000 # 1010_2 Using cutlery
#id <- 1015; from <- 3945; to <- 3995 # 1015_1 Preparing intake
#id <- 1036; from <- 2410; to <- 2465 # 1036_1 Lasagne intake
#id <- 1107; from <- 2480; to <- 2545 # 1107_1 Using cutlery

# Examples for single vs sequence
#models <- c("oreba_2d_cnn_frame/eval", "resnet_2d_cnn_frame/eval", "oreba_3d_cnn/eval", "resnet_3d_cnn/eval")
#id <- 1020; from <- 2907; to <- 2947 # 1020_2 Raised fork
#id <- 1043; from <- 1400; to <- 1480 # 1043_2 Passing butter
#id <- 1075; from <- 2285; to <- 2315 # 1075_1 Blowing nose
#id <- 1086; from <- 4560; to <- 4605 # 1086_1 Regular bite

# Examples for mistakes for all models
models <- c("resnet_3d_cnn/eval", "resnet_cnn_lstm/eval", "resnet_slowfast/eval")
#id <- 1004; from <- 2635; to <- 2665 # 1004_1 Bread (false negative)
#id <- 1048; from <- 0580; to <- 0625 # 1048_4 Chin drum (false positive type 2)
#id <- 1054; from <- 0674; to <- 0724 # 1054_1 Long sip (false positive type 1)
#id <- 1060; from <- 1546; to <- 1561 # 1060_2 Licking finger (false negative)
#id <- 1081; from <- 3834; to <- 3890 # 1081_1 Long sip (false pos type 1)
id <- 1112; from <- 0223; to <- 0263 # 1112_1 Food too hot (false positive type 2)

data <- import(models); data <- data[order(data$id, data$seq),]; rownames(data) <- NULL
write(data, id, from, to, paste(id, ".csv", sep=""))

prepare.probs <- function(data, id, from, to, names) {
  data <- data[data$id==id,]
  data <- data[from:to,]
  result <- data.frame()
  for (name in names) {
    result.1 <- data[, c("seq", paste("prob", gsub("/", ".", name), sep="."))]
    colnames(result.1) <- c("t", "p")
    result.1$model <- name; result.1$t <- seq(1:nrow(result.1))
    result <- rbind(result, result.1)
  }
  result$model <- factor(result$model, levels=names)
  return(result)
}
prepare.gt <- function(data, id, from, to) {
  data <- data[data$id==id,]
  data <- data[from:to,]
  start <- which(c(TRUE, tail(data$label, -1) != head(data$label, -1)) & data$label == 1)
  end <- which(c(tail(data$label, -1) != head(data$label, -1), TRUE) & data$label == 1)
  data.frame(start=start, end=end)
}
prepare.cols <- function(names) {
  cols <- c()
  for (name in names) {
    if (grepl("oreba_2d_cnn_frame", name))
      cols <- c(cols, "#00FF13")
    else if (grepl("oreba_3d_cnn", name))
      cols <- c(cols, "#455BFF")
    else if (grepl("oreba_cnn_lstm", name))
      cols <- c(cols, "#FF9500")
    else if (grepl("oreba_slowfast", name))
      cols <- c(cols, "#00F5FF")
    else if (grepl("resnet_2d_cnn_frame", name))
      cols <- c(cols, "#009B0C")
    else if (grepl("resnet_3d_cnn", name))
      cols <- c(cols, "#00129B")
    else if (grepl("resnet_cnn_lstm", name))
      cols <- c(cols, "#995900")
    else if (grepl("resnet_slowfast", name))
      cols <- c(cols, "#009399")
    else if (grepl("oreba_2d_cnn_flow", name))
      cols <- c(cols, "#FFDE00")
    else if (grepl("resnet_2d_cnn_flow", name))
      cols <- c(cols, "#B29B00")
    else
      print("no color assigned!")
  }
  return(cols)
}

# Prepare the data for plot
probs <- prepare.probs(data, id, from, to, models)
gt <- prepare.gt(data, id, from, to)
cols <- prepare.cols(models)

# From/to video times [s]
(from+16)/8/60
(to+16)/8/60

# Preview plot
preview <- function(probs, gt, cols) {
  library(ggplot2)
  if (nrow(gt) > 0) {
    p <- ggplot(data=probs, aes(x=t, y=p, group=model)) +
      geom_rect(data=gt, inherit.aes=FALSE,
                aes(xmin=start,xmax=end,ymin=-Inf,ymax=Inf), fill="#FF0014", alpha=0.5) +
      geom_line(aes(color=model), size=1) +
      scale_color_manual(values=cols) +
      theme_minimal() +
      theme(axis.text.x=element_blank()) +
      scale_y_continuous(breaks=c(0,1), minor_breaks = c(0.25, 0.5, 0.75), limits=c(0,1)) +
      scale_x_continuous(breaks=c())
  } else {
    p <- ggplot(data=probs, aes(x=t, y=p, group=model)) +
      geom_line(aes(color=model), size=1) +
      scale_color_manual(values=cols) +
      theme_minimal() +
      theme(axis.text.x=element_blank()) +
      scale_y_continuous(breaks=c(0,1), minor_breaks = c(0.25, 0.5, 0.75), limits=c(0,1)) +
      scale_x_continuous(breaks=c())
  }
  p
}
preview(probs, gt, cols)

# Generate animation
generate <- function(probs, gt, cols) {
  library(gganimate)
  if (nrow(gt) > 0) {
    a <- ggplot(data=probs, aes(x=t, y=p, group=model)) +
      geom_rect(data=gt, inherit.aes=FALSE,
                aes(xmin=start,xmax=end,ymin=-Inf,ymax=Inf), fill="#FF0014", alpha=0.5) +
      geom_line(aes(color=model), size=1) +
      geom_point(aes(color=model)) +
      scale_color_manual(values=cols) +
      theme_minimal() +
      theme(axis.text.x=element_blank(), legend.position = "none") +
      scale_y_continuous(breaks=c(0,1), minor_breaks = c(0.25, 0.5, 0.75), limits=c(0,1)) +
      scale_x_continuous(breaks=c()) +
      transition_reveal(t)
  } else {
    a <- ggplot(data=probs, aes(x=t, y=p, group=model)) +
      geom_line(aes(color=model), size=1) +
      geom_point(aes(color=model)) +
      scale_color_manual(values=cols) +
      theme_minimal() +
      theme(axis.text.x=element_blank(), legend.position = "none") +
      scale_y_continuous(breaks=c(0,1), minor_breaks = c(0.25, 0.5, 0.75), limits=c(0,1)) +
      scale_x_continuous(breaks=c()) +
      transition_reveal(t)
  }
  # animate in a two step process:
  animate(a, height=140, width=160, nframes=nrow(probs)/length(cols), fps=8)
  anim_save("graph.gif")
}
generate(probs, gt, cols)
