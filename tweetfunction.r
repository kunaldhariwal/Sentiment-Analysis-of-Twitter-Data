library(twitteR)
library(ROAuth)
library(tidyverse)
library(purrrlyr)
library(text2vec)
library(caret)
library(glmnet)
library(ggrepel)

conv_fun <- function(x) iconv(x, "latin1", "ASCII", "")

tweets_classified <- read_csv('training.1600000.processed.noemoticon.csv', col_names = c('sentiment', 'id', 'date', 'query', 'user', 'text')) %>%
  dmap_at('text', conv_fun) %>%
  mutate(sentiment = ifelse(sentiment == 0, 0, 1))
tweets_classified_na <- tweets_classified %>%
  filter(is.na(id) == TRUE) %>%
  mutate(id = c(1:n()))
tweets_classified <- tweets_classified %>%
  filter(!is.na(id)) %>%
  rbind(., tweets_classified_na)

set.seed(2340)
trainIndex <- createDataPartition(tweets_classified$sentiment, p = 0.8, 
                                  list = FALSE, 
                                  times = 1)
tweets_train <- tweets_classified[trainIndex, ]
tweets_test <- tweets_classified[-trainIndex, ]

prep_fun <- tolower
tok_fun <- word_tokenizer

it_train <- itoken(tweets_train$text, 
                   preprocessor = prep_fun, 
                   tokenizer = tok_fun,
                   ids = tweets_train$id,
                   progressbar = TRUE)
it_test <- itoken(tweets_test$text, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun,
                  ids = tweets_test$id,
                  progressbar = TRUE)

vocab <- create_vocabulary(it_train)
vectorizer <- vocab_vectorizer(vocab)
dtm_train <- create_dtm(it_train, vectorizer)
tfidf <- TfIdf$new()
dtm_train_tfidf <- fit_transform(dtm_train, tfidf)
dtm_test_tfidf  <- create_dtm(it_test, vectorizer) %>% 
  transform(tfidf)

# train the model
t1 <- Sys.time()
glmnet_classifier <- cv.glmnet(x = dtm_train_tfidf,
                               y = tweets_train[['sentiment']], 
                               family = 'binomial', 
                               alpha = 1,
                               type.measure = "auc",
                               nfolds = 5,
                               thresh = 1e-3,
                               maxit = 10)
print(difftime(Sys.time(), t1, units = 'mins'))

plot(glmnet_classifier)
print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))

preds <- predict(glmnet_classifier, dtm_test_tfidf, type = 'response')[ ,1]
auc(as.numeric(tweets_test$sentiment), preds)

rm(list = setdiff(ls(), c('glmnet_classifier', 'conv_fun', 'prep_fun', 'tok_fun', 'vectorizer', 'tfidf')))
save.image('image.RData')
rm(list = ls())
