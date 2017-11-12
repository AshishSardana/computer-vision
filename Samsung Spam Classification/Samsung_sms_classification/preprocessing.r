rm(list = ls())

library(text2vec)
library(data.table)
library(stringr)

path <- "/home/ashu/Hack2Innovate/Samsung_sms_classification"
setwd(path)

train <- read.csv("TRAIN_SMS.csv" , stringsAsFactors = FALSE)
test <- read.csv("DEV_SMS.csv" , stringsAsFactors = FALSE)
sample_sub <- read.csv("sample_submission.csv" , stringsAsFactors = FALSE)

train_lbl <- train$Label
test_ID <- test$RecordNo
n_test <- nrow(test)
n_train <- nrow(train)

train_sms <- train$Message
test_sms <- test$Message

sms <- c(train_sms, test_sms)
sms <- data.frame(sms , stringsAsFactors = FALSE)
sms$id <- 1:nrow(sms)

sms$sms <- apply(data.frame(sms$sms , stringsAsFactors = FALSE), 1, function(x) str_replace_all(x, "[[:punct:]]", " ") )

word_count <- sapply(gregexpr("[[:alpha:]]+", sms$sms), function(x) sum(x > 0))

setDT(sms)
setkey(sms, id)
set.seed(2017L)
all_ids = sms$id


# define preprocessing function and tokenization function
prep_fun = tolower
tok_fun = word_tokenizer
stop_words = c("a", "the")

# Creating tokens
it_train = itoken(sms$sms, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = sms$id, 
                  progressbar = TRUE)
vocab = create_vocabulary(it_train , stopwords = stop_words , ngram = c(1L, 2L))

# Pruning vocabulory

pruned_vocab = prune_vocabulary(vocab, doc_proportion_min = 0.01)
vectorizer = vocab_vectorizer(pruned_vocab)

dtm_train = create_dtm(it_train, vectorizer)

## Text frequency Inverse Document Frequency

tfidf = TfIdf$new()
# fit model to train data and transform train data with fitted model
dtm_train_tfidf = fit_transform(dtm_train, tfidf)

fin_data <- as.matrix(dtm_train_tfidf)
fin_data2 <- as.data.frame(fin_data)
fin_data2$word_count <- word_count
tr <- fin_data2[1:n_train,]
te <- fin_data2[n_train+1:nrow(fin_data2), ]
tr$label <- train_lbl
te$RecordNo <- test_ID

write.csv(tr, "ptrain.csv", row.names = F)
write.csv(te, "ptest.csv", row.names = F)
