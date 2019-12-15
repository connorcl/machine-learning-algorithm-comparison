# set random seed
set.seed(1)

# load banknote authentication data
banknote_auth_data <- read.csv("data_banknote_authentication.csv", header = FALSE)
n_rows <- dim(banknote_auth_data)[1]

# randomize the order of the data
banknote_auth_data <- banknote_auth_data[sample(seq(n_rows)),]

# split into training and validation sets
training_set_logical <- sample(c(TRUE, FALSE), size = n_rows, replace = TRUE, prob = c(0.8, 0.2))
validation_set_logical <- !training_set_logical
training_set = banknote_auth_data[training_set_logical,]
validation_set = banknote_auth_data[validation_set_logical,]

# write training and validation sets to csv files
write.table(training_set, "banknote_train.csv", quote = FALSE, sep = ",", row.names = FALSE, col.names = FALSE)
write.table(validation_set, "banknote_valid.csv", quote = FALSE, sep = ",", row.names = FALSE, col.names = FALSE)