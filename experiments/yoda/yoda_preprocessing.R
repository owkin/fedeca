library(tidyverse)
library(haven)
library(missForest)

varnames <- list(
  pid = "Patient ID",
  ecog = "ECOG",
  bpi = "BPI>1",
  bone = "Bone metastasis only",
  age = "Age",
  bmi = "BMI",
  trt = "Treatment",
  time = "Time",
  event = "Event",
  endpoint = "Endpoint",
  trial = "Trial"
)

armnames <- list(
  apa_aa_p = "Apalutamide_AA_P",
  aa_p = "AA_P",
  p = "Placebo_P"
)

outnames <- list(
  os = "OS",
  rpfs = "rPFS"
)

load_xpt <- function(rct) {
  dfs <- list()
  for (name in names(rct$vars)) {
    dfs[[name]] <- read_xpt(
      paste0(rct$path, name, ".xpt"),
      col_select = c(rct$pid, rct$vars[[name]])
    )
  }
  return(dfs)
}

get_rct1 <- function() {
  return(
    list(
      name = "NCT02257736",
      path = "X:\\Data\\JnJ-56021927PCR3001xpt3\\Data\\",
      pid = "USUBJID",
      outcomes = list(
        os = "Overall Survival (months)",
        rpfs = "Radiographic Progression-free Survival (months)"
      ),
      arms = list(
        apa_aa_p = "APALUTAMIDE+AA-P",
        aa_p = "PLACEBO+AA-P"
      ),
      vars = list(
        adsl = c(
          "ECOGBL", "BPISFBL", "BNONLFL", "AGE", "WEIGHTBL", "HEIGHTBL",
          "VISCSCRN", # for exclusion in compliance with rct2
          "ARM"
        ),
        adtteef = c("STARTDT", "CNSR", "ADY", "PARAM")
      )
    )
  )
}

get_rct2 <- function() {
  return(
    list(
      name = "NCT00887198",
      path = "X:\\Data\\JnJ-COU-AA-302xpt3\\Data\\",
      pid = "USUBJID",
      outcomes = list(
        os = "OS",
        rpfs = "IRPFS"
      ),
      arms = list(
        aa_p = "AA",
        p = "Placebo"
      ),
      vars = list(
        atrisk = c(
          "BLECOG", "BLBPICAT", "BNMET", "TRTP", "RANDOMDT",
          "TRTSTDT", "CENSOR",
          "TMTOEVNT", "ENDPT"
        ),
        demo = c("AGE", "HT_ST"),
        vslc = c("WT_ST", "DOV_")
      )
    )
  )
}

preprocess_rct1 <- function(dfs, rct) {
  dfs$adtteef <- dfs$adtteef %>%
    filter(PARAM %in% unlist(rct$outcomes, use.names = FALSE)) %>%
    mutate(
      !!varnames$event := 1 - CNSR,
      !!varnames$time := ADY / (365 / 12),
      !!varnames$endpoint := case_match(
        PARAM,
        rct$outcomes$os ~ outnames$os,
        rct$outcomes$rpfs ~ outnames$rpfs
      )
    )

  ids_to_exclude <- dfs$adsl %>%
    filter(VISCSCRN == "Presence") %>%
    pull(rct$pid)

  dfs$adsl <- dfs$adsl %>%
    filter(ARM %in% unlist(rct$arms, use.names = FALSE)) %>%
    mutate(
      !!varnames$bpi := as.numeric(BPISFBL > 1.5),
      !!varnames$bone := as.numeric(BNONLFL == "Y"),
      !!varnames$bmi := WEIGHTBL / (HEIGHTBL / 100)^2,
      !!varnames$trt := case_match(
        ARM,
        rct$arms$apa_aa_p ~ armnames$apa_aa_p,
        rct$arms$aa_p ~ armnames$aa_p
      )
    )

  df <- dfs$adsl %>%
    left_join(dfs$adtteef, by = rct$pid) %>%
    filter(!.[[rct$pid]] %in% ids_to_exclude) %>%
    mutate(
      !!varnames$trial := rct$name
    ) %>%
    rename(
      !!varnames$pid := rct$pid,
      !!varnames$ecog := ECOGBL,
      !!varnames$age := AGE
    ) %>%
    select(unlist(varnames, use.names = FALSE))

  return(df)
}

preprocess_rct2 <- function(dfs, rct) {
  df_ref <- dfs$atrisk %>%
    filter(ENDPT == rct$outcomes$os)

  df_weight <- dfs$vslc %>%
    drop_na(WT_ST) %>%
    left_join(df_ref, by = rct$pid) %>%
    mutate(
      time_diff = DOV_ - TRTSTDT
    ) %>%
    filter(time_diff <= 7 & time_diff >= -30) %>%
    group_by(across(rct$pid)) %>%
    arrange(abs(time_diff), DOV_, .by_group = TRUE) %>%
    filter(row_number() == 1) %>%
    ungroup() %>%
    select(rct$pid, "WT_ST", time_diff)

  dfs$atrisk <- dfs$atrisk %>%
    filter(ENDPT %in% unlist(rct$outcomes, use.names = FALSE)) %>%
    mutate(
      across(c(BLBPICAT, BNMET), ~ na_if(., "")),
      !!varnames$bpi := as.numeric(BLBPICAT == "2-3"),
      !!varnames$bone := as.numeric(BNMET == "Yes"),
      !!varnames$trt := case_match(
        TRTP,
        rct$arms$aa_p ~ armnames$aa_p,
        rct$arms$p ~ armnames$p
      ),
      !!varnames$event := 1 - CENSOR,
      !!varnames$time := TMTOEVNT / (365 / 12),
      !!varnames$endpoint := case_match(
        ENDPT,
        rct$outcomes$os ~ outnames$os,
        rct$outcomes$rpfs ~ outnames$rpfs
      )
    )

  df <- dfs$atrisk %>%
    left_join(dfs$demo, by = rct$pid) %>%
    left_join(df_weight, by = rct$pid) %>%
    mutate(
      !!varnames$bmi := WT_ST / (HT_ST / 100)^2,
      !!varnames$trial := rct$name
    ) %>%
    rename(
      !!varnames$pid := rct$pid,
      !!varnames$ecog := BLECOG,
      !!varnames$age := AGE
    ) %>%
    select(unlist(varnames, use.names = FALSE))

  return(df)
}

impute_covariates <- function(df) {
  non_covs <- c(
    varnames$pid,
    varnames$trt,
    varnames$time,
    varnames$event,
    varnames$endpoint,
    varnames$trial
  )
  ids <- df %>%
    filter(Endpoint == "OS") %>%
    pull(varnames$pid)

  df_cov <- df %>%
    filter(Endpoint == "OS") %>%
    select(-all_of(non_covs)) %>%
    mutate(across(c(varnames$ecog, varnames$bpi, varnames$bone), as.factor)) %>%
    as.data.frame()

  res_imputation <- df_cov %>%
    missForest()

  df_cov_imputed <- res_imputation[["ximp"]] %>%
    mutate(!!varnames$pid := ids, .before = varnames$ecog) %>%
    as.tibble() %>%
    mutate(
      across(
        c(varnames$ecog, varnames$bpi, varnames$bone),
        ~ as.numeric(as.character(.))
      )
    )

  df_imputed <- df %>%
    select(all_of(non_covs)) %>%
    left_join(df_cov_imputed, by = varnames$pid) %>%
    select(unlist(varnames, use.names = FALSE))

  return(df_imputed)
}

preprocess_all <- function(imputation = FALSE) {
  rct1 <- get_rct1()
  rct2 <- get_rct2()
  df1 <- preprocess_rct1(load_xpt(rct1), rct1)
  df2 <- preprocess_rct2(load_xpt(rct2), rct2)
  if (imputation) {
    df <- bind_rows(
      impute_covariates(df1),
      impute_covariates(df2)
    )
  } else {
    df <- bind_rows(df1, df2) %>% drop_na()
  }

  return(df)
}

df <- preprocess_all(imputation = FALSE)
set.seed(42)
df_imputed <- preprocess_all(imputation = TRUE)
