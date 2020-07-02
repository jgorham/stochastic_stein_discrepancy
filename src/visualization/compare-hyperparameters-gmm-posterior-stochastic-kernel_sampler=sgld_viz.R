# script for plotting banana visualizations
library(ggplot2)
library(grid)
library(gridExtra)
library(plyr)
library(reshape2)
library(coda)
library(hexbin)

source('results_data_utils.R')

### ess plots
dir <- "compare-hyperparameters-gmm-posterior"
discrepancytype <- "inversemultiquadric"
L <- 100
tn <- 1
n <- 1000
d <- 2
seed <- NULL
distname <- "gmm-posterior"

sampler.dat <- concatDataList(
    dir=dir,
    distname=distname,
    discrepancytype=discrepancytype,
    sampler="sgld",
    n=n,
    thinningn=tn,
    L=L,
    d=d,
    seed=seed
)

diag.df <- ldply(sampler.dat, function(res) {
    x <- do.call(cbind, res$X)
    #ess <- effectiveSize(x)
    #ess.avg <- mean(ess)
    data.frame(
        n=nrow(x),
        seed=res$seed,
        ksd=res$objectivevalue,
        likelihoodn=res$likelihoodn,
        thinningn=res$thinningn,
        solvetime=res$solvetime,
        ncores=res$ncores,
        epsilon=res$epsilon
    )
})

summary.df <- ddply(
    diag.df,
    .(epsilon, likelihoodn, thinningn),
    function (df) {
        sd <- sd(df$ksd)
        med <- median(df$ksd)
        S <- length(df$ksd)
        c(
            "numseeds"=S,
            "median"=med,
            "mean"=mean(df$ksd),
            "sd"=sd,
            "low"=mean(df$ksd)-sd/sqrt(S),
            "high"=mean(df$ksd)+sd/sqrt(S)
        )
    })

summary.df <- subset(summary.df, numseeds >= 10)
summary.df <- summary.df[order(summary.df$epsilon, summary.df$likelihoodn, summary.df$thinningn), ]
numseeds <- median(summary.df$numseeds)

format_eps <- get_epsilon_labeller(variable="")
eps.breaks <- sort(unique(summary.df$epsilon))
eps.labels <- format_eps(eps.breaks)
# remove 10^{-j} labels
is.power.10 <- floor(abs(log10(eps.breaks)) - 0.01) != floor(abs(log10(eps.breaks)))
summary.df$epsilon <- factor(
    summary.df$epsilon,
    levels=eps.breaks,
    labels=as.char(eps.breaks))

likelihoodn.vals <- sort(unique(summary.df$likelihoodn), decreasing=TRUE)
# want 0 at the top
likelihoodn.vals <- c(0, head(likelihoodn.vals, -1))
summary.df$likelihoodn <- factor(
    summary.df$likelihoodn,
    levels=likelihoodn.vals,
    labels=c("100 (all)", as.char(tail(likelihoodn.vals, -1)))
)

pos.d <- position_dodge(width=0.75)

ksd.plt <- ggplot(data=summary.df, aes(x=epsilon, y=log(mean), color=likelihoodn)) +
  geom_point(aes(shape=likelihoodn), position=pos.d) +
  #geom_path(aes(group=likelihoodn), position=pos.d) +
  geom_errorbar(aes(ymin=log(low), ymax=log(high)), position=pos.d, width=0.75) +
  labs(x=expression(paste("Tolerance parameter, ", epsilon)), y="log(SKSD)") +
  scale_x_discrete(
      breaks=as.char(eps.breaks[!is.power.10]),
      labels=parse(text=eps.labels[!is.power.10])
  ) +
  guides(
    color = guide_legend(title="Likelihoods, m"),
    shape = guide_legend(title="Likelihoods, m")
  ) +
  #scale_y_continuous(breaks=seq(-1.0, 5.0, by=0.5)) +
  theme_bw() +
  theme(plot.margin = unit(c(0,0,0,0), "npc"),
        axis.text = element_text(size=7),
        axis.title = element_text(size=10),
        strip.background = element_rect(fill="white",
            color="black", size=1),
        legend.title = element_text(size=8),
        legend.text = element_text(size=7))

# Prepare data for contour plot
SIGMA2Y <- 2.0
SIGMA2X1 <- 10.0
SIGMA2X2 <- 1.0
GMM.logpdf <- function(x, y) {
    log.prior.lik <- dnorm(x[1], sd=sqrt(SIGMA2X1), log=T) +
        dnorm(x[2], sd=sqrt(SIGMA2X2), log=T)
    log.likelihood <- log(0.5) + log(
        dnorm(y, mean=x[1], sd=sqrt(SIGMA2Y)) +
        dnorm(y, mean=x[1] + x[2], sd=sqrt(SIGMA2Y))
    )
    log.prior.lik + sum(log.likelihood)
}
gen.posterior.df <- function(y, xlim=c(-3, 10), ylim=c(-3, -20), n=120) {
    xrange <- seq(xlim[1], xlim[2], length.out=n)
    yrange <- seq(ylim[1], ylim[2], length.out=n)
    points <- expand.grid(x1=xrange, x2=yrange)
    logdens <- apply(points, 1, function (x.point) {
      GMM.logpdf(x.point, y)
    })
    cbind(points, logdens=logdens)
}

# SAMPLER
sampler.seed <- 7
## dcast(
##     subset(trial.df, seed == sampler.seed),
##     epsilon ~ diagnostic,
##     value.var = "metric")

# SGLD
sgld.seed <- 7
## dcast(
##     subset(trial.df, seed == sgld.seed),
##     epsilon ~ diagnostic,
##     value.var = "metric")

y.sgld <- concatDataList(
    dir="compare-hyperparameters-gmm-posterior-y",
    numsamples=100,
    x="\\[0.0, 1.0\\]"
)[[1]][['y']]

sgld.examples <- concatDataList(
    dir=dir,
    distname=distname,
    discrepancytype=discrepancytype,
    sampler="sgld",
    epsilon="(5.0e-5|0\\.005|0\\.05)",
    likelihoodn=0,
    d=d,
    seed=sgld.seed
)
sgld.xs.df <- ldply(sgld.examples, function(res) {
    data.frame(
        distname=res$distname,
        x1=res$X[[1]],
        x2=res$X[[2]],
        epsilon=res$epsilon,
        seed=res$seed
    )
})

logdens.df <- gen.posterior.df(y.sgld,
                               xlim=range(sgld.xs.df$x1),
                               ylim=range(sgld.xs.df$x2))

contour.breaks <- quantile(logdens.df$logdens,
                           c(0.985, 0.99, 0.994))

format_eps <- get_epsilon_labeller(variable="")
eps.breaks <- sort(unique(sgld.xs.df$epsilon))
eps.labels <- paste0("epsilon == ", format_eps(eps.breaks))
sgld.xs.df$epsilon <- factor(
    sgld.xs.df$epsilon,
    levels=eps.breaks,
    labels=eps.labels)

scatter.plts <- ggplot() +
    geom_point(aes(x=x1, y=x2), data=sgld.xs.df, color="black", size=1) +
    stat_contour(
        aes(x=x1, y=x2, z=logdens, color=..level..),
        size=0.4,
        binwidth=20,
        breaks=contour.breaks,
        data=logdens.df) +
    facet_grid(. ~ epsilon, labeller = label_parsed) +
    labs(x=expression(x[1]), y=expression(x[2])) +
    #scale_x_continuous(breaks=seq(from=-2.5, to=3, by=0.5),
    #                   labels=c("", "-2", "", "-1", "", "0", "", "1", "", "2", "", "3")) +
    #scale_y_continuous(breaks=seq(from=-4, to=4, by=1)) +
    scale_color_gradient(name="Log posterior",
                         low="green", high="red") +
    theme_bw() +
    theme(plot.margin = unit(c(0,0.02,0,0), "npc"),
          strip.background = element_rect(fill="white"),
          legend.position = "none")


filename <- sprintf(
    "../../results/%s/figs/julia_compare-hyperparameters-gmm-posterior_stochastic_distname=%s_discrepancytype=inversemultiquadric_n=%d_seed=%s.pdf",
    dir, distname, n, sgld.seed
)
makeDirIfMissing(filename)

pdf(file=filename, width=8.5, height=2.7)
grid.arrange(
  scatter.plts, ksd.plt,
  widths = unit.c(unit(0.6, "npc"), unit(0.35, "npc")),
  nrow=1)
dev.off()
