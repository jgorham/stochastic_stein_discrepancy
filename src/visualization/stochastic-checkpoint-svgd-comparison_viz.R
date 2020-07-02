# script for plotting banana visualizations
library(ggplot2)
library(grid)
library(plyr)
library(dplyr)
library(gridExtra)
library(reshape2)

source('results_data_utils.R')

############
# Imports! #
############
dir <- "stochastic_svgd"
dataset <- "(yacht|boston|naval)"
batchsizefrac <- NULL
nhidden <- NULL
particles <- NULL

OPTIMAL.STEPSIZES <- list(
    boston = 0.001,
    naval = 0.001,
    yacht = 0.01
)
OPTIMAL.NUM.EPOCHS <- list(
    boston = 50,
    naval = 500,
    yacht = 50
)
ITERATES.PER.EPOCH <- list(
    boston = 409,
    naval = 10241,
    yacht = 249
)

raw.dat <- concatData(
    prefix="svgd",
    dir=dir,
    dataset=dataset,
    batchsizefrac=batchsizefrac,
    nhidden=nhidden,
    particles=particles
)
colnames(raw.dat) <- c(
    'dataset',
    'trial',
    'batch.size.frac',
    'nhidden',
    'particles',
    'saga',
    'stepsize',
    'iter',
    'svgd.rmse',
    'svgd.ll',
    'svgd.grad.size',
    'svgd.time'
)

base <- raw.dat %>%
    mutate(
        num.epochs = batch.size.frac * iter,
        num.likelihoods = (1 + floor(as.num(ITERATES.PER.EPOCH[dataset]) * batch.size.frac)) * iter,
        batch.size.frac = factor(batch.size.frac, levels=sort(unique(batch.size.frac))),
        particles = factor(particles, levels=sort(unique(particles))),
        svgd.ll = ifelse(is.finite(svgd.ll), svgd.ll, NaN)
    ) %>%
    filter(
        (dataset == "boston" & (num.epochs == round(num.epochs))) |
        (dataset == "yacht" & (num.epochs / 2 == round(num.epochs / 2))) |
        (dataset == "naval" & (num.epochs / 20 == round(num.epochs / 20)))
    ) %>%
    filter(
        (dataset == "yacht" & stepsize == OPTIMAL.STEPSIZES["yacht"]) |
        (dataset == "boston" & stepsize == OPTIMAL.STEPSIZES["boston"]) |
        (dataset == "naval" & stepsize == OPTIMAL.STEPSIZES["naval"])
    ) %>%
    filter(
        (dataset == "yacht" & num.epochs <= OPTIMAL.NUM.EPOCHS["yacht"]) |
        (dataset == "boston" & num.epochs <= OPTIMAL.NUM.EPOCHS["boston"]) |
        (dataset == "naval" & num.epochs <= OPTIMAL.NUM.EPOCHS["naval"])
    )


GROUP.BY.VARS <- c('dataset', 'trial', 'batch.size.frac', 'num.likelihoods', 'iter', 'num.epochs', 'nhidden', 'particles', 'stepsize', 'iter')

skinny.dat <- melt(
    base %>% select(c(GROUP.BY.VARS, c('svgd.rmse', 'svgd.ll'))),
    id.vars=c(GROUP.BY.VARS),
    value.name='stat.value',
    variable.name='stat')

skinny.dat$stat <- factor(
    skinny.dat$stat,
    levels=c('svgd.rmse', 'svgd.ll'),
    labels=c('RMSE', 'Log Likelihood')
)

sum.dat <- ddply(skinny.dat, .(dataset, batch.size.frac, stepsize, num.likelihoods, particles, stat), function (df) {
    stat.mean <- mean(df$stat.value, na.rm=T)
    stat.sd <- sd(df$stat.value, na.rm=T)
    data.frame(
        stat.mean = stat.mean,
        stat.sd = stat.sd,
        stat.lo = stat.mean - 2 * stat.sd / sqrt(length(df)),
        stat.hi = stat.mean + 2 * stat.sd / sqrt(length(df))
    )
})

pd <- position_dodge(width=0.3)
format_eps0 <- get_epsilon_labeller(variable="")
format_eps1 <- get_epsilon_labeller(variable="", dec.places=1)
boston.plot <- (
    ggplot(
        data=sum.dat %>% filter(dataset == "boston"),
        aes(x=num.likelihoods, group=batch.size.frac, color=batch.size.frac)
    ) +
    geom_point(aes(y=stat.mean), position=pd) +
    geom_errorbar(aes(ymin=stat.lo, ymax=stat.hi), position=pd, width=0.3) +
    facet_grid(stat ~ dataset, scales="free_y", switch="y") +
    xlab("") +
    ylab("") +
    scale_x_continuous(
        breaks=(0:4)*5000,
        labels=parse(text=format_eps1((0:4)*5000))
    ) +
    guides(
        color = guide_legend(title="Batch size\nfraction, m/L")
    ) +
    theme_bw() +
    theme(strip.background = element_rect(fill="white"),
          legend.text = element_text(size=8),
          legend.title = element_text(size=8),
          legend.position = "none",
          axis.title = element_text(size=9),
          axis.text = element_text(size=9),
          legend.margin = unit(c(0,0,0,0), "npc"),
          plot.margin = unit(c(0,0,0,0), "npc"),
          strip.text = element_text(size=8),
          strip.placement = "outside")
)
yacht.plot <- (
    ggplot(
        data=sum.dat %>% filter(dataset == "yacht"),
        aes(x=num.likelihoods, group=batch.size.frac, color=batch.size.frac)
    ) +
    geom_point(aes(y=stat.mean), position=pd) +
    geom_errorbar(aes(ymin=stat.lo, ymax=stat.hi), position=pd, width=0.3) +
    facet_grid(stat ~ dataset, scales="free_y") +
    xlab("# likelihood evaluations") +
    ylab("") +
    scale_x_continuous(
        breaks=(0:4)*3000,
        labels=parse(text=format_eps1((0:4)*3000))
    ) +
    guides(
        color = guide_legend(title="Batch size\nfraction, m/L")
    ) +
    theme_bw() +
    theme(strip.background = element_rect(fill="white"),
          strip.text.y = element_blank(),
          legend.position = "none",
          legend.text = element_text(size=8),
          legend.title = element_text(size=8),
          axis.title = element_text(size=9),
          axis.text = element_text(size=9),
          plot.margin = unit(c(0,0,0,0), "npc"),
          strip.text = element_text(size=8))
)
naval.plot <- (
    ggplot(
        data=sum.dat %>% filter(dataset == "naval"),
        aes(x=num.likelihoods, group=batch.size.frac, color=batch.size.frac)
    ) +
    geom_point(aes(y=stat.mean), position=pd) +
    geom_errorbar(aes(ymin=stat.lo, ymax=stat.hi), position=pd, width=0.3) +
    facet_grid(stat ~ dataset, scales="free_y") +
    xlab("") +
    ylab("") +
    scale_x_continuous(
        breaks=(0:5)*10^6,
        labels=parse(text=format_eps1((0:5)*10^6))
    ) +
    guides(
        color = guide_legend(title="Batch size\nfraction, m/L")
    ) +
    theme_bw() +
    theme(strip.background = element_rect(fill="white"),
          strip.text.y = element_blank(),
          legend.position = "right",
          legend.text = element_text(size=8),
          legend.title = element_text(size=8),
          axis.title = element_text(size=9),
          axis.text = element_text(size=9),
          plot.margin = unit(c(0,0,0,0), "npc"),
          strip.text = element_text(size=8))
)

filename <- sprintf(
    "../../results/%s/figs/julia_stochastic-checkpoint-svgd-comparison.pdf",
    dir
)
makeDirIfMissing(filename)

# Save to file
pdf(file=filename, width=10, height=3)
grid.arrange(
    boston.plot, yacht.plot, naval.plot, ncol=3,
    widths = unit.c(unit(0.315, "npc"), unit(0.295, "npc"), unit(0.38, "npc"))
)
dev.off()


## ll.plot <- (
##     ggplot(
##         data=sum.dat %>% filter(stat == "svgd.ll"),
##         aes(x=num.likelihoods, group=batch.size.frac, color=batch.size.frac)
##     ) +
##     geom_point(aes(y=stat.mean), position=pd) +
##     geom_errorbar(aes(ymin=stat.lo, ymax=stat.hi), position=pd, width=0.3) +
##     facet_wrap(~ dataset, scales="free", ncol=3) +
##     xlab("# gradient evaluations") +
##     ylab("Log likelihood") +
##     guides(
##         color = guide_legend(title="Batch size fraction")
##     ) +
##     theme_bw() +
##     theme(strip.background = element_rect(fill="white"),
##           legend.text = element_text(size=8),
##           legend.title = element_text(size=8),
##           axis.title = element_text(size=9),
##           axis.text = element_text(size=9),
##           strip.text = element_text(size=8))
## )

## rmse.plot <- (
##     ggplot(
##         data=sum.dat %>% filter(stat == "svgd.rmse"),
##         aes(x=num.likelihoods, group=batch.size.frac, color=batch.size.frac)
##     ) +
##     geom_point(aes(y=stat.mean), position=pd) +
##     geom_errorbar(aes(ymin=stat.lo, ymax=stat.hi), position=pd, width=0.3) +
##     facet_wrap(~ dataset, scales="free", ncol=3) +
##     xlab("# gradient evaluations") +
##     ylab("RMSE") +
##     guides(
##         color = guide_legend(title="Batch size fraction")
##     ) +
##     # make the legend blank
##     scale_color_discrete(
##         guide = guide_legend(override.aes = list(color = "white"))
##     ) +
##     scale_fill_discrete(
##         guide = guide_legend(override.aes = list(color = "white"))
##     ) +
##     theme_bw() +
##     theme(strip.background = element_rect(fill="white"),
##           legend.text = element_text(size=8, color="white"),
##           legend.title = element_text(size=8, color="white"),
##           legend.key = element_rect(fill = "white", color="white"),
##           axis.title = element_text(size=9),
##           axis.text = element_text(size=9),
##           strip.text = element_text(size=8))
## )
